"""Serial, isolated-process experiments. Never run concurrent GPU benchmarks."""
import json
from pathlib import Path
import subprocess
import sys
import time

CASES = {
    'sequential': [],
    'batched-one': ['--batched-one'],
    'batch2': ['--batch','2'],
    'batch4': ['--batch','4'],
    'batch8': ['--batch','8'],
    'beam1': ['--beam','1'],
    'float16': ['--compute','float16'],
    'threads8': ['--threads','8'],
    'english': ['--language','en'],
    'no-words': ['--no-words'],
    'no-vad': ['--no-vad'],
    'turbo': ['--model','large-v3-turbo'],
    'distil': ['--model','distil-large-v3','--language','en'],
}


def main():
    root=Path(sys.argv[1] if len(sys.argv)>1 else 'benchmark/results/matrix')
    root.mkdir(parents=True,exist_ok=True)
    summary={}
    for name, options in CASES.items():
        output=root/name
        if output.exists():
            print('SKIP existing',name,flush=True); continue
        print('START',name,flush=True); t=time.perf_counter()
        with (root/f'{name}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-u','-m','benchmark.measure',
                                   '--output',str(output),'--duration','180','--repeats','3',*options],
                                  stdout=log,stderr=subprocess.STDOUT)
        summary[name]={'exit_code':result.returncode,'process_seconds':time.perf_counter()-t}
        (root/'matrix.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print('END',name,summary[name],flush=True)


if __name__=='__main__': main()
