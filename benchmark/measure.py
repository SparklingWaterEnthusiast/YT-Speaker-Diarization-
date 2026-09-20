"""Reproducible offline inference experiments; never opens the user's job DB.

Run from the repository root: .venv/Scripts/python -m benchmark.measure --help
Results and audio are deliberately separate from the production cache.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
import wave


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, help='Frozen baseline package root')
    p.add_argument('--wav', type=Path, default=Path('benchmark/work/audio/fZZXVNt1gk0/audio.wav'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--beam', type=int, default=5)
    p.add_argument('--compute', default='int8_float16')
    p.add_argument('--model', default='large-v3')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--language', default='auto')
    p.add_argument('--no-words', action='store_true')
    p.add_argument('--no-vad', action='store_true')
    p.add_argument('--batched-one', action='store_true', help='Experimental batched decoder at batch 1, for semantic isolation')
    p.add_argument('--diarize', action='store_true')
    p.add_argument('--only-diarize', action='store_true')
    p.add_argument('--diar-batch', type=int, default=32)
    p.add_argument('--margin', type=int, default=1024, help='VRAM margin MiB, including allocator budget')
    p.add_argument('--duration', type=float, default=0, help='Crop a prefix, seconds; 0 full')
    a = p.parse_args()
    if a.source_root:
        sys.path.insert(0, str(a.source_root.resolve()))
    a.output.mkdir(parents=True, exist_ok=False)
    from ytscribe import resources
    # Capture timeline independent of which application version is imported.
    import psutil
    import pynvml as nv
    nv.nvmlInit()
    handle = nv.nvmlDeviceGetHandleByIndex(0)
    process = psutil.Process()
    done = threading.Event()
    origin = time.perf_counter()
    events = open(a.output / 'timeline.jsonl', 'w', encoding='utf-8')
    lock = threading.Lock()
    def emit(kind, **fields):
        row = dict(kind=kind, elapsed=time.perf_counter()-origin, pid=os.getpid(), **fields)
        with lock:
            events.write(json.dumps(row, default=str)+'\n'); events.flush()
    def attempt(fn, *args):
        try: return fn(*args)
        except Exception: return None
    def sample():
        mem = nv.nvmlDeviceGetMemoryInfo(handle)
        util = nv.nvmlDeviceGetUtilizationRates(handle)
        torch = sys.modules.get('torch')
        metrics = {}
        if torch is not None and torch.cuda.is_initialized():
            metrics = {k: getattr(torch.cuda,k)() for k in ('memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved')}
        emit('sample', gpu=util.gpu, vram_mb=mem.used/2**20,
             temperature_c=attempt(nv.nvmlDeviceGetTemperature,handle,nv.NVML_TEMPERATURE_GPU),
             sm_mhz=attempt(nv.nvmlDeviceGetClockInfo,handle,nv.NVML_CLOCK_SM),
             power_mw=attempt(nv.nvmlDeviceGetPowerUsage,handle),
             throttle=attempt(nv.nvmlDeviceGetCurrentClocksThrottleReasons,handle),
             rss_mb=process.memory_info().rss/2**20, cpu=psutil.cpu_percent(),
             process_cpu=process.cpu_percent(), cores=psutil.cpu_percent(percpu=True),
             threads=process.num_threads(), torch=metrics)
    sample()
    def monitor():
        while not done.wait(0.5):
            try: sample()
            except Exception as exc: emit('sample_error',error=str(exc))
    monitor_thread = threading.Thread(target=monitor,daemon=True); monitor_thread.start()
    from ytscribe.config import Config
    cfg = Config(device='cuda', whisper_model=a.model, compute_type=a.compute,
                 beam_size=a.beam, language=a.language, vad_filter=not a.no_vad)
    cfg.asr_batch_size = a.batch
    cfg.asr_cpu_threads = a.threads
    cfg.word_timestamps = not a.no_words
    cfg.diarization_batch_size = a.diar_batch
    cfg.vram_margin_mb = a.margin
    with wave.open(str(a.wav),'rb') as w:
        duration = w.getnframes()/w.getframerate()
        wav = a.wav
        if a.duration:
            wav = a.output/'input.wav'
            with wave.open(str(wav),'wb') as out:
                out.setparams(w.getparams()); out.writeframes(w.readframes(int(a.duration*w.getframerate())))
            duration=min(duration,a.duration)
    battery=psutil.sensors_battery()
    manifest = dict(arguments=vars(a), duration=duration, pid=os.getpid(),
                    python=sys.version, platform=platform.platform(),
                    gpu=nv.nvmlDeviceGetName(handle), driver=nv.nvmlSystemGetDriverVersion(),
                    ac_power=None if battery is None else battery.power_plugged,
                    versions={n:importlib.metadata.version(n) for n in ('torch','ctranslate2','faster-whisper','pyannote.audio','yt-dlp')})
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
    from ytscribe.transcribe import Transcriber
    from ytscribe.diarize import Diarizer
    asr=Transcriber(cfg, lambda s: print(s,flush=True))
    if a.batched_one:
        asr.transcribe = lambda wav, duration: asr._transcribe_once(wav,duration,1,True,None,None)
    dia=Diarizer(cfg, lambda s: print(s,flush=True)) if a.diarize or a.only_diarize else None
    results=[]
    recorder = None
    if not a.source_root:
        from ytscribe.telemetry import Recorder
        recorder = Recorder(a.output/'stages.jsonl', interval=0.5)
        recorder.__enter__()
    try:
        for rep in range(a.repeats):
            row={'repeat':rep,'cold':rep==0}
            for name,model,load,call in ([] if a.only_diarize else [('asr',asr,asr._ensure_model,lambda:asr.transcribe(wav,duration))])+(
                [('diarization',dia,dia._ensure_pipeline,lambda:dia.diarize(wav))] if dia else []):
                emit('stage_start',stage=name+'_load',repeat=rep); t=time.perf_counter(); load()
                row[name+'_load_s']=time.perf_counter()-t; emit('stage_end',stage=name+'_load',seconds=row[name+'_load_s'],repeat=rep)
                emit('stage_start',stage=name,repeat=rep); t=time.perf_counter(); result=call()
                row[name+'_s']=time.perf_counter()-t; row[name+'_x']=duration/row[name+'_s']
                emit('stage_end',stage=name,seconds=row[name+'_s'],repeat=rep)
                (a.output/f'{name}-{rep}.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
            results.append(row)
            (a.output/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
            print(json.dumps(row),flush=True)
    finally:
        emit('before_unload'); asr.unload()
        if dia: dia.unload()
        emit('after_unload'); sample(); done.set(); monitor_thread.join(); events.close()
        if recorder:
            recorder.__exit__(None,None,None)


if __name__ == '__main__':
    main()
