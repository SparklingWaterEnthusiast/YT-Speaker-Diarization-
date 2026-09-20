"""Offline compatibility/export checks using actual measured model artifacts.

Uses a fresh isolated root, never the production voice or job database.
Recognition checks here are in-sample compatibility checks, not accuracy evals.
"""
import json
from pathlib import Path
import shutil

import numpy as np

from ytscribe.config import Config
from ytscribe.db import JobStore
from ytscribe.pipeline import QueueRunner
from ytscribe import voices


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    root=Path('benchmark/results/integration')
    root.mkdir(parents=True,exist_ok=False)
    vid='fZZXVNt1gk0'; vdir=root/'cache'/vid; vdir.mkdir(parents=True)
    measured=Path('benchmark/results/final-batch4-full')
    for source,target in ((measured/'asr-0.json','transcript.json'),
                          (measured/'diarization-0.json','diarization.json'),
                          (Path('benchmark/work/audio')/vid/'meta.json','meta.json')):
        shutil.copyfile(source,vdir/target)
    cfg=Config(cache_dir=str(root/'cache'),output_dir=str(root/'output'),
               recognition_enabled=False,profiling_enabled=True)
    store=JobStore(root/'jobs.sqlite3')
    store.add(vid,f'https://www.youtube.com/watch?v={vid}',duration=1706.20225)
    QueueRunner(cfg,store).run_single(vid)
    files=list(cfg.output_path.iterdir())
    assert len(files)==1 and files[0].suffix=='.md'
    assert not (vdir/'audio.wav').exists()
    baseline=read(Path('benchmark/results/baseline-full/diarization-0.json'))
    final=read(vdir/'diarization.json')
    matches={}; cosines={}
    profile_path=root/'voices.sqlite3'; vdb=voices.VoiceDB(profile_path)
    alignment=voices.align_to_reference(baseline,voices.parse_reference(Path('benchmark/reference_fZZXVNt1gk0.txt')))
    for label,info in alignment.items():
        vdb.add_sample(info['name'],baseline['embeddings'][label],f'baseline:{label}')
    vdb.close(); vdb=voices.VoiceDB(profile_path)
    for label,vec in final['embeddings'].items():
        matches[label]=vdb.match(vec,cfg.recognition_threshold)
        old=np.asarray(baseline['embeddings'][label]); new=np.asarray(vec)
        cosines[label]=float(np.dot(old,new)/(np.linalg.norm(old)*np.linalg.norm(new)))
    for label,info in alignment.items():
        assert matches[label][0]==info['name']
    for label in set(final['speakers'])-set(alignment):
        assert matches[label] is None
    cfg.export_formats=['md','json','txt','srt','vtt']
    QueueRunner(cfg,store).run_single(vid)
    assert {f.suffix for f in cfg.output_path.iterdir()}=={'.md','.json','.txt','.srt','.vtt'}
    for label,info in alignment.items():
        voices.apply_rename(cfg,vdir,label,info['name'],vdb,print)
    for file in cfg.output_path.iterdir():
        text=file.read_text(encoding='utf-8')
        assert all(info['name'] in text for info in alignment.values())
    result={'markdown_only':True,'all_optional_exports':True,'rename_updates_all_formats':True,
            'no_audio_or_models_needed':True,'profile_reopen':True,'matches':matches,
            'embedding_cosines':cosines,'raw_turns_equal':baseline['turns']==final['turns'],
            'exclusive_turns_equal':baseline['exclusive']==final['exclusive'],
            'limitation':'Same-video profile compatibility; not independent cross-video recognition accuracy.'}
    (root/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    store.close(); vdb.close()


if __name__=='__main__': main()
