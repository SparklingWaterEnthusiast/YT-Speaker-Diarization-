"""Compact, reproducible summaries from measured traces (no GPU inference)."""
import hashlib
import json
from pathlib import Path
import statistics

from benchmark.quality import evaluate, reference

ROOT=Path('benchmark/results')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def stats(samples):
    result={}
    for key in ('gpu','vram_mb','temperature_c','sm_mhz','power_mw','rss_mb',
                'cpu','process_cpu','threads'):
        values=[s[key] for s in samples if s.get(key) is not None]
        if values:
            result[key]={'mean':statistics.mean(values),'min':min(values),'max':max(values)}
    result['throttle_masks']=sorted(set(s['throttle'] for s in samples if s.get('throttle') is not None))
    return result


def summarize(folder,ref):
    rows=[json.loads(line) for line in (folder/'timeline.jsonl').read_text(encoding='utf-8').splitlines()]
    result={'runs':read(folder/'results.json'),'manifest':read(folder/'manifest.json'),
            'resources':stats([r for r in rows if r['kind']=='sample']),'stages':[]}
    for begin in rows:
        if begin['kind']!='stage_start': continue
        end=next(r for r in rows if r['kind']=='stage_end' and r['stage']==begin['stage'] and r['repeat']==begin['repeat'])
        samples=[r for r in rows if r['kind']=='sample' and begin['elapsed']<=r['elapsed']<=end['elapsed']]
        result['stages'].append({'stage':begin['stage'],'repeat':begin['repeat'],
                                'seconds':end['seconds'],'resources':stats(samples)})
    diar=folder/'diarization-0.json'
    result['quality']=evaluate(read(folder/'asr-0.json'),ref,result['manifest']['duration'],
                               read(diar) if diar.exists() else None)
    result['after_unload']=rows[-1] if rows[-1]['kind']=='sample' else None
    return result


def lifecycle(root):
    result={}
    for phase in ('pause','resume','queue'):
        receipt=root/f'{phase}-result.json'
        if receipt.exists(): result[phase]=read(receipt)
    external=root/'external-timeline.json'
    if external.exists():
        result['external_process_boundaries']=[r for r in read(external) if r['phase'] not in ('pause','resume')]
    result['traces']=[]
    for path in sorted((root/'cache/diagnostics').glob('*.jsonl')):
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        samples=[r for r in rows if r['kind']=='sample']
        stages=[r for r in rows if r['kind']=='span_end']
        result['traces'].append({
            'pid':rows[0]['pid'],
            'stages':[{'name':r['name'],'seconds':r['duration_s'],'cache_hit':r['cache_hit'],
                       'status':r['status'],'fields':r['fields']} for r in stages],
            'peak_vram_mib':max(s['fields']['resources']['vram_used_gb']*1024 for s in samples),
            'peak_rss_mib':max(s['fields']['resources']['process_rss_bytes']/2**20 for s in samples),
            'peak_torch_allocated':max(s['fields']['torch'].get('max_memory_allocated_bytes') or 0 for s in samples),
            'peak_torch_reserved':max(s['fields']['torch'].get('max_memory_reserved_bytes') or 0 for s in samples),
            'items':[],
        })
        for begin in (r for r in rows if r['kind']=='span_start' and r['name']=='item.total'):
            end=next((r for r in stages if r['span_id']==begin['span_id']),None)
            if end is None: continue
            batch=[s['fields']['resources'] for s in samples if begin['elapsed_s']<=s['elapsed_s']<=end['elapsed_s']]
            if not batch: continue
            result['traces'][-1]['items'].append({
                'video_id':begin['fields']['video_id'],'seconds':end['duration_s'],'status':end['status'],
                'gpu_mean':statistics.mean(s['gpu_percent'] for s in batch),
                'vram_peak_mib':max(s['vram_used_gb']*1024 for s in batch),
                'rss_peak_mib':max(s['process_rss_bytes']/2**20 for s in batch),
                'threads_end':batch[-1]['process_threads'],
                'temperature_max':max(s['temperature_c'] for s in batch),
                'thermal_samples':sum(bool(set(s['throttle_reasons'] or ()) & {'sw_thermal_slowdown','hw_thermal_slowdown'}) for s in batch),
            })
    return result


def main():
    ref=reference(Path('benchmark/reference_fZZXVNt1gk0.txt'))
    names=[name for name in ('baseline-full','batch4-full','final-batch4-full','baseline-recheck')
           if (ROOT/name/'results.json').exists()]
    names.extend(p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'matrix').iterdir())
                 if p.is_dir() and (p/'results.json').exists())
    output={'evidence':'MEASURED; one RTX 3080 Laptop, one source video. Prefix sweep not full-video quality proof.',
            'reference_note':'Whole-second imperfect reference; WER normalized, speaker agreement is not DER.',
            'runs':{name:summarize(ROOT/name,ref) for name in names},
            'restart_diagnostic_aborted':lifecycle(Path('benchmark/work/lifecycle')),
            'restart_fixed_passed':lifecycle(Path('benchmark/work/lifecycle-fixed'))}
    full_root=Path('benchmark/work/full-pipeline')
    if (full_root/'queue-result.json').exists():
        output['full_pipeline']=lifecycle(full_root)
    for name,path in {
        'integration':ROOT/'integration/result.json',
        'timestamp_proxies':ROOT/'timestamp-proxies.json',
        'hardware_probe':ROOT/'ui/hardware.json',
        'acquisition_check':Path('benchmark/work/acquisition-check/result.json'),
        'oom_recovery_runs':ROOT/'oom-budget/results.json',
    }.items():
        if path.exists(): output[name]=read(path)
    oom_path=ROOT/'oom-budget/stages.jsonl'
    if oom_path.exists():
        output['oom_recovery_events']=[json.loads(line) for line in oom_path.read_text(encoding='utf-8').splitlines()
                                       if json.loads(line)['name']=='diarization.oom']
    (ROOT/'v0.3.0-summary.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    for name,item in output['runs'].items():
        runs=item['runs']; warm=[r['asr_s'] for r in runs if not r['cold']]
        print(f"{name}: ASR {[round(r['asr_s'],2) for r in runs]}, warm mean {statistics.mean(warm):.2f}s; "
              f"WER {100*item['quality']['wer']:.2f}%; VRAM peak {item['resources']['vram_mb']['max']:.0f}MiB")


if __name__=='__main__': main()
