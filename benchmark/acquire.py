"""Acquire only the approved benchmark, in a fresh isolated cache."""
import argparse
import json
from pathlib import Path
import time

from ytscribe.config import Config
from ytscribe import media
from ytscribe.telemetry import Recorder


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('benchmark/work/acquisition-check'))
    a=p.parse_args(); a.root.mkdir(parents=True,exist_ok=False)
    cfg=Config(); vid='fZZXVNt1gk0'; vdir=a.root/vid
    with Recorder(a.root/'trace.jsonl',disk_path=a.root):
        start=time.perf_counter()
        audio=media.download_audio(f'https://www.youtube.com/watch?v={vid}',vdir,cfg)
        downloaded=time.perf_counter()
        wav=media.to_wav(audio,cfg.resolve_ffmpeg())
        result={'download_and_metadata_s':downloaded-start,
                'conversion_and_ffmpeg_resolution_s':time.perf_counter()-downloaded,
                'duration_s':media.wav_duration(wav),'source_bytes':audio.stat().st_size,
                'wav_bytes':wav.stat().st_size}
    (a.root/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
