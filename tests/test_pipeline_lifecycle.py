"""Regression coverage for cache-only work, cancellation and GPU ownership."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from ytscribe.config import Config
from ytscribe.db import JobStore
from ytscribe import pipeline, transcribe, inference
from types import SimpleNamespace
from ytscribe.worker_lock import processing_lease


class PipelineLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.cfg=Config(cache_dir=str(self.root/'cache'),output_dir=str(self.root/'output'),
                        recognition_enabled=False,sleep_between_downloads_min=0,
                        sleep_between_downloads_max=0)
        self.store=JobStore(self.root/'jobs.sqlite3')
        self.runner=pipeline.QueueRunner(self.cfg,self.store)

    def tearDown(self):
        self.runner.close(); self.store.close(); self.temp.cleanup()

    def cached(self, vid='test'):
        self.store.add(vid,'https://example.invalid',title='Test',duration=10)
        vdir=self.cfg.cache_path/vid; vdir.mkdir(parents=True)
        for name, obj in {
            'meta':{'video_id':vid,'title':'Test','duration':10,'url':'https://example.invalid'},
            'transcript':{'segments':[{'start':0,'end':1,'text':'Hello','words':[]} ]},
            'diarization':{'speakers':['SPEAKER_00'],'exclusive':[], 'turns':[], 'embeddings':{}},
        }.items():
            (vdir/f'{name}.json').write_text(json.dumps(obj),encoding='utf-8')
        return vdir

    def test_cached_reexport_needs_no_audio_models_or_ffmpeg(self):
        self.cached()
        with patch.object(pipeline.media,'download_audio',side_effect=AssertionError('network')), \
             patch.object(Config,'resolve_ffmpeg',side_effect=AssertionError('ffmpeg')), \
             patch.object(pipeline.transcribe,'Transcriber',side_effect=AssertionError('ASR')), \
             patch.object(pipeline.diarize,'Diarizer',side_effect=AssertionError('diarization')):
            self.runner.run_single('test')
        self.assertEqual(self.store.get('test')['status'],'done')
        self.assertEqual(len(list(self.cfg.output_path.glob('*.md'))),1)

    def test_cancelled_transcription_is_not_failed(self):
        self.store.add('cancel','u'); self.store.set_queue_active(1,True)
        self.runner._process=Mock(side_effect=InterruptedError('transcription cancelled'))
        result=self.runner.run()
        self.assertEqual(result['failed'],0)
        self.assertEqual(self.store.get('cancel')['status'],'cancelled')

    def test_config_is_immutable_for_running_worker(self):
        self.cfg.whisper_model='small'
        self.assertEqual(self.runner.cfg.whisper_model,'large-v3')

    def test_invalid_cli_configuration_fails_before_processing(self):
        self.cfg.asr_batch_size=0
        with self.assertRaisesRegex(ValueError,'asr_batch_size'):
            pipeline.QueueRunner(self.cfg,self.store)

    def test_close_while_paused_queues_current_stage_for_restart(self):
        self.store.add('paused','u'); self.store.set_queue_active(1,True)
        def interrupted(*args):
            self.store.set_stage('paused','transcribe')
            self.runner.stop()
            raise InterruptedError('application closing')
        self.runner._process=interrupted
        self.runner.run()
        self.assertEqual(self.store.get('paused')['status'],'queued')
        self.assertEqual(self.store.get('paused')['stage'],'transcribe')

    def test_completed_cache_does_not_inflate_processing_speed(self):
        self.cached(); self.store.set_queue_active(1,True)
        result=self.runner.run()
        self.assertEqual(result['completed'],1)
        self.assertEqual(result['avg_rtf'],0)

    def test_prefetch_remains_owned_across_priority_change(self):
        gate=threading.Event(); entered=threading.Event()
        def download(*args,**kwargs):
            entered.set(); gate.wait(2); return self.root/'audio.m4a'
        with patch.object(pipeline.media,'download_audio',side_effect=download) as call:
            self.runner._start_prefetch({'video_id':'a','url':'u'})
            self.assertTrue(entered.wait(1))
            self.runner._start_prefetch({'video_id':'b','url':'u'})
            self.assertEqual(list(self.runner._prefetch),['a'])
            gate.set(); self.runner.close()
            self.assertEqual(call.call_count,1)
        self.assertFalse(any(t.name=='prefetch-a' for t in threading.enumerate()))

    def test_keep_forever_protects_owned_audio(self):
        vdir=self.cached(); audio=vdir/'audio.wav'; audio.write_bytes(b'audio')
        self.runner.cfg.cache_policy='keep_forever'; self.runner._delete_audio(vdir)
        self.assertTrue(audio.exists())

    def test_prefetch_reuses_existing_wav_without_source_download(self):
        vdir=self.cached()
        (vdir/'transcript.json').unlink()
        (vdir/'audio.wav').write_bytes(b'0'*100)
        with patch.object(pipeline.media,'download_audio',side_effect=AssertionError('network')):
            self.runner._start_prefetch({'video_id':'test','url':'u'})
            self.assertFalse(self.runner._prefetch)

    def test_housekeeping_does_not_delete_user_temporary_files(self):
        vdir=self.cached()
        (vdir/'personal.tmp.notes').write_text('keep',encoding='utf-8')
        (vdir/'transcript.tmp').write_text('incomplete',encoding='utf-8')
        self.runner.run_single('test')
        self.assertTrue((vdir/'personal.tmp.notes').exists())
        self.assertFalse((vdir/'transcript.tmp').exists())

    def test_audio_cleanup_preserves_artifacts_samples_and_user_files(self):
        vdir=self.cached()
        for name in ('audio.wav','audio.m4a','audio.m4a.part','audio.notes.txt','personal.wav'):
            (vdir/name).write_bytes(b'x')
        (vdir/'samples').mkdir(); (vdir/'samples'/'SPEAKER_00.wav').write_bytes(b'x')
        self.runner._delete_audio(vdir)
        self.assertFalse((vdir/'audio.m4a.part').exists())
        self.assertTrue((vdir/'audio.notes.txt').exists())
        self.assertTrue((vdir/'personal.wav').exists())
        self.assertTrue((vdir/'transcript.json').exists())
        self.assertTrue((vdir/'samples'/'SPEAKER_00.wav').exists())

    def test_queue_cleanup_includes_previous_completed_items(self):
        vdir=self.cached(); (vdir/'audio.wav').write_bytes(b'x')
        self.store.mark_done('test',1)
        self.runner.cfg.cache_policy='delete_after_queue'
        self.runner._maybe_finalize_queue(1)
        self.assertFalse((vdir/'audio.wav').exists())

    def test_processing_lease_rejects_second_worker_and_releases(self):
        with processing_lease(self.cfg.cache_path):
            with self.assertRaisesRegex(RuntimeError,'Another YTScribe'):
                with processing_lease(self.cfg.cache_path): pass
        with processing_lease(self.cfg.cache_path): pass

    def test_changing_cache_does_not_bypass_database_worker_lease(self):
        with processing_lease(self.root/'cache-a',self.root/'database'):
            with self.assertRaisesRegex(RuntimeError,'Another YTScribe'):
                with processing_lease(self.root/'cache-b',self.root/'database'): pass


class OOMRecovery(unittest.TestCase):
    def make(self):
        worker=transcribe.Transcriber.__new__(transcribe.Transcriber)
        worker.cfg=Config(asr_batch_size=8,oom_retries=2)
        worker.device='cpu'; worker.log=Mock(); worker._ensure_model=Mock()
        worker._model=None
        return worker

    def test_oom_retries_halve_work_without_partial_output(self):
        worker=self.make()
        worker._transcribe_once=Mock(side_effect=[RuntimeError('CUDA out of memory'),
                                                  RuntimeError('CUDA out of memory'),{'segments':[]}])
        with patch.object(transcribe,'release_unused') as release:
            self.assertEqual(worker.transcribe(Path('fake'),10),{'segments':[]})
            self.assertEqual(release.call_count,2)
        self.assertEqual([c.args[2] for c in worker._transcribe_once.call_args_list],[8,4,2])

    def test_non_oom_is_not_retried(self):
        worker=self.make(); worker._transcribe_once=Mock(side_effect=RuntimeError('invalid CUDA DLL'))
        with self.assertRaisesRegex(RuntimeError,'invalid CUDA DLL'):
            worker.transcribe(Path('fake'),10)
        self.assertEqual(worker._transcribe_once.call_count,1)

    def test_retries_are_bounded(self):
        worker=self.make(); worker._transcribe_once=Mock(side_effect=RuntimeError('CUDA out of memory'))
        with patch.object(transcribe,'release_unused'), self.assertRaisesRegex(RuntimeError,'out of memory'):
            worker.transcribe(Path('fake'),10)
        self.assertEqual(worker._transcribe_once.call_count,3)


class TorchBudget(unittest.TestCase):
    def test_budget_accounts_for_reserved_pool_and_restores_prior_limit(self):
        cuda=Mock()
        cuda.is_initialized.return_value=True
        cuda.get_per_process_memory_fraction.return_value=1.0
        cuda.get_device_properties.return_value=SimpleNamespace(total_memory=8*2**30)
        cuda.memory_allocated.return_value=100*2**20
        cuda.memory_reserved.return_value=200*2**20
        nv=Mock()
        nv.nvmlDeviceGetMemoryInfo.return_value=SimpleNamespace(free=4*2**30)
        with patch.dict('sys.modules',{'torch':SimpleNamespace(cuda=cuda),'pynvml':nv}), \
             patch.object(inference,'release_unused'):
            with self.assertRaisesRegex(RuntimeError,'failure'):
                with inference.torch_budget(1024,'cuda',Mock()):
                    raise RuntimeError('failure')
        fractions=[c.args[0] for c in cuda.set_per_process_memory_fraction.call_args_list]
        self.assertAlmostEqual(fractions[0],(4096+200-1024)/8192)
        self.assertEqual(fractions[1],1.0)
        nv.nvmlShutdown.assert_called_once()

    def test_cpu_budget_never_probes_or_initializes_cuda(self):
        torch=Mock()
        with patch.dict('sys.modules',{'torch':torch}):
            with inference.torch_budget(1024,'cpu',Mock()): pass
        torch.cuda.is_initialized.assert_not_called()


if __name__=='__main__': unittest.main()
