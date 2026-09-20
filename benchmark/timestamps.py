"""Timestamp-regression proxies, NOT alignment error against ground truth."""
import difflib
import json
from pathlib import Path
import statistics

from benchmark.quality import words


def flatten(path):
    data=json.loads(path.read_text(encoding='utf-8'))
    return [w for s in data['segments'] for w in s.get('words',[])]


def main():
    old=flatten(Path('benchmark/results/baseline-full/asr-0.json'))
    new=flatten(Path('benchmark/results/final-batch4-full/asr-0.json'))
    a=[' '.join(words(w['word'])) for w in old]
    b=[' '.join(words(w['word'])) for w in new]
    differences=[]
    for block in difflib.SequenceMatcher(None,a,b,autojunk=False).get_matching_blocks():
        for i in range(block.size):
            differences.append(abs(old[block.a+i]['start']-new[block.b+i]['start']))
    differences.sort()
    result={
        'evidence':'MEASURED consistency proxy, not timestamp accuracy or DER',
        'matched_words':len(differences),
        'median_start_difference_s':statistics.median(differences),
        'p95_start_difference_s':differences[int(.95*(len(differences)-1))],
        'invalid_word_intervals':sum(w['start']<0 or w['end']<w['start'] or w['end']>1706.20225+1 for w in new),
        'zero_duration_words':sum(w['end']==w['start'] for w in new),
        'limitations':'Matched-word timing can differ legitimately; reference lacks word-level timing. Zero-duration words are upstream alignment artifacts.'}
    path=Path('benchmark/results/timestamp-proxies.json')
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
