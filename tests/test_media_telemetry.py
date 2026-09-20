"""Downloader timing hooks must not change file selection or metadata."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ytscribe import media
from ytscribe.config import Config
from ytscribe.telemetry import Recorder


class DownloadTelemetry(unittest.TestCase):
    def test_transfer_and_residual_are_explicit_not_pure_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); trace=root/'trace.jsonl'; vdir=root/'video'
            class Downloader:
                def __init__(self,opts): self.opts=opts
                def __enter__(self): return self
                def __exit__(self,*args): pass
                def extract_info(self,url,download):
                    self.opts['progress_hooks'][0]({'status':'finished','elapsed':0.01,'downloaded_bytes':3})
                    (vdir/'audio.m4a').write_bytes(b'abc')
                    return {'id':'video','title':'Test','duration':2,'webpage_url':url}
            with patch.object(media.yt_dlp,'YoutubeDL',Downloader), Recorder(trace,interval=60):
                audio=media.download_audio('https://example.invalid',vdir,Config())
            self.assertEqual(audio.name,'audio.m4a')
            self.assertEqual(media.load_meta(vdir)['duration'],2)
            rows=[json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
            transfer=next(r for r in rows if r['name']=='download.transfer')
            residual=next(r for r in rows if r['name']=='acquisition.preparation_residual')
            self.assertEqual(transfer['fields']['duration_s'],0.01)
            self.assertIn('not pure metadata',residual['fields']['note'])
            self.assertGreaterEqual(residual['fields']['duration_s'],0)


if __name__=='__main__': unittest.main()
