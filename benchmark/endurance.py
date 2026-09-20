"""Observe real GUI process exit/restart externally, without a CUDA context."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import pynvml as nv


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('benchmark/work/lifecycle'))
    a=p.parse_args(); root=a.root.resolve()
    nv.nvmlInit(); h=nv.nvmlDeviceGetHandleByIndex(0)
    rows=[]; origin=time.perf_counter()
    def sample(phase,pid=None):
        rows.append(dict(phase=phase,pid=pid,elapsed_s=time.perf_counter()-origin,
                         vram_mb=nv.nvmlDeviceGetMemoryInfo(h).used/2**20,
                         gpu_percent=nv.nvmlDeviceGetUtilizationRates(h).gpu))
    for phase in ('pause','resume'):
        sample('before_'+phase)
        with (root/f'{phase}.log').open('w',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-u','-m','benchmark.lifecycle',phase,
                                    '--root',str(root)],stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                sample(phase,child.pid); time.sleep(0.5)
        sample('after_'+phase,child.pid)
        (root/'external-timeline.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(f'{phase}: PID {child.pid}, exit {child.returncode}',flush=True)
        if child.returncode:
            raise RuntimeError(f'{phase} failed; inspect {root / (phase+".log")}')
        # A small bounded settling window records driver release after exit.
        for _ in range(4):
            time.sleep(0.5); sample('settled_'+phase)
    (root/'external-timeline.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    nv.nvmlShutdown()


if __name__=='__main__': main()
