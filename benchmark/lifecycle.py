"""Real Qt worker/process restart and a three-item offline endurance queue.

Creates isolated data under --root. phase prepare uses only the benchmark WAV.
phase pause closes the actual MainWindow after ASR is cached; phase resume
launches a new process, reuses ASR, diarizes, then transcribes two further items.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import wave


def main():
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=('prepare','pause','resume','queue'))
    p.add_argument('--root',type=Path,default=Path('benchmark/work/lifecycle'))
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--full',action='store_true',help='Prepare one full benchmark source instead of excerpts')
    a=p.parse_args(); root=a.root.resolve(); root.mkdir(parents=True,exist_ok=True)
    from ytscribe.config import Config
    from ytscribe.db import JobStore
    cfg=Config(cache_dir=str(root/'cache'),output_dir=str(root/'outputs'),
               cache_policy='keep_forever',recognition_enabled=False,
               asr_batch_size=a.batch,profiling_enabled=True,profiling_interval=0.5,
               sleep_between_downloads_min=0,sleep_between_downloads_max=0)
    if a.phase=='prepare':
        if (root/'jobs.sqlite3').exists(): raise RuntimeError('Use a fresh benchmark root')
        source=Path('benchmark/work/audio/fZZXVNt1gk0/audio.wav')
        store=JobStore(root/'jobs.sqlite3')
        fixtures=([('bench-first',0,1706.20225)] if a.full else
                  [('bench-first',0,300),('bench-second',1176,480),('bench-third',240,600)])
        for vid,offset,length in fixtures:
            vdir=cfg.cache_path/vid; vdir.mkdir(parents=True,exist_ok=True)
            with wave.open(str(source),'rb') as w, wave.open(str(vdir/'audio.wav'),'wb') as out:
                out.setparams(w.getparams()); w.setpos(offset*w.getframerate())
                out.writeframes(w.readframes(int(length*w.getframerate())))
            meta=dict(video_id=vid,title=f'Benchmark excerpt {offset}-{offset+length}',
                      url='https://www.youtube.com/watch?v=fZZXVNt1gk0',duration=length,
                      source_offset_seconds=offset,channel='Benchmark fixtures')
            (vdir/'meta.json').write_text(json.dumps(meta),encoding='utf-8')
            store.add(vid,meta['url'],meta['title'],length)
        store.close(); return

    # All application module paths are redirected before the window constructs.
    # The production job DB, config, cache and voice profiles are never opened.
    os.environ['QT_QPA_PLATFORM']='offscreen'
    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import QTimer
    from ytscribe import pipeline, transcribe
    from ytscribe.ui import main_window
    main_window.APP_DIR=root; pipeline.APP_DIR=root
    main_window.load_config=lambda: cfg
    main_window.save_config=lambda c: None
    QMessageBox.question=lambda *args,**kwargs: QMessageBox.StandardButton.Yes
    app=QApplication([]); win=main_window.MainWindow(); win.show()
    checkpoint=threading.Event(); cached=root/'cache/bench-first/transcript.json'
    old_stage=transcribe.run_stage
    if a.phase=='pause':
        def stage(*args,**kwargs):
            result=old_stage(*args,**kwargs)
            win.run_worker.runner.pause(); checkpoint.set()
            return result
        transcribe.run_stage=stage
    elif a.phase=='resume':
        if not cached.exists(): raise RuntimeError('Pause phase must cache transcription first')
        before=hashlib.sha256(cached.read_bytes()).hexdigest()
        if win.store.get('bench-first')['status'] != 'queued':
            raise RuntimeError('Clean shutdown did not leave the interrupted item ready for Start')
    def tick():
        if a.phase=='pause' and checkpoint.is_set() and not win._closing:
            (root/'paused.json').write_text(json.dumps({'pid':os.getpid(),
                'asr_sha256':hashlib.sha256(cached.read_bytes()).hexdigest()}),encoding='utf-8')
            win.close()
        elif a.phase!='pause' and win.run_worker and not win.run_worker.isRunning():
            win.close()
    timer=QTimer(); timer.timeout.connect(tick); timer.start(100)
    QTimer.singleShot(0,win.start_queue)
    started=time.perf_counter(); app.exec()
    jobs=win.store.all_jobs()
    result={'phase':a.phase,'pid':os.getpid(),'wall_s':time.perf_counter()-started,
            'statuses':{j['video_id']:j['status'] for j in jobs},
            'worker_finished':win._workers_finished()}
    if a.phase=='resume':
        result['cached_asr_unchanged']=hashlib.sha256(cached.read_bytes()).hexdigest()==before
    (root/f'{a.phase}-result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (root/f'{a.phase}-ui-log.txt').write_text(win.log.toPlainText(),encoding='utf-8')
    print(json.dumps(result),flush=True)
    win.store.close(); win.voice_db.close()
    if not result['worker_finished']: return 2
    if a.phase!='pause' and any(j['status']!='done' for j in jobs): return 3
    return 0


if __name__=='__main__': sys.exit(main() or 0)
