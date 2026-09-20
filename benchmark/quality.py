"""WER on provided reference coverage and coarse time-weighted speaker checks.

Reference timestamps are whole seconds and sometimes zero-duration; this is
not a collar/overlap-aware DER scorer. No text is used to condition inference.
"""
import argparse
import json
from pathlib import Path
import re

LINE = re.compile(r'^(.*?) \((\d+):(\d+)-(\d+):(\d+)\):\s*(.*)$')


def reference(path):
    return [dict(speaker=m[1], start=int(m[2])*60+int(m[3]),
                 end=int(m[4])*60+int(m[5]), text=m[6])
            for line in path.read_text(encoding='utf-8').splitlines()
            if (m := LINE.match(line))]


def words(text):
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1',text)
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?",text.lower().replace('’',"'"))


def edit_distance(ref, hyp):
    # rapidfuzz is already a transitive dependency. Fallback needs only O(n) space.
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(ref,hyp)
    except ImportError:
        previous=list(range(len(hyp)+1))
        for i,r in enumerate(ref,1):
            current=[i]
            for j,h in enumerate(hyp,1):
                current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(r!=h)))
            previous=current
        return previous[-1]


def evaluate(asr, ref, duration=None, diar=None):
    end=min(max(r['end'] for r in ref),duration or float('inf'))
    selected=[r for r in ref if r['start'] < end and r['end'] <= end]
    # Compare identical coverage, excluding a reference turn cut by the crop.
    end=max(r['end'] for r in selected)
    ref_tokens=words(' '.join(r['text'] for r in selected))
    hyp_tokens=words(' '.join(w['word'] for s in asr['segments'] for w in s.get('words',[])
                            if w['start'] < end))
    if not hyp_tokens:
        hyp_tokens=words(' '.join(s['text'] for s in asr['segments'] if s['start'] < end))
    result={'reference_words':len(ref_tokens),'hypothesis_words':len(hyp_tokens),
            'end_seconds':end,'word_errors':edit_distance(ref_tokens,hyp_tokens)}
    result['wer']=result['word_errors']/len(ref_tokens)
    if diar:
        overlap={}
        for t in diar['exclusive']:
            for r in selected:
                d=max(0,min(t['end'],r['end'])-max(t['start'],r['start']))
                if d:
                    counts=overlap.setdefault(t['speaker'],{})
                    counts[r['speaker']]=counts.get(r['speaker'],0)+d
        mapping={label:max(counts,key=counts.get) for label,counts in overlap.items()}
        total=sum(sum(c.values()) for c in overlap.values())
        correct=sum(c[mapping[label]] for label,c in overlap.items())
        result.update(speaker_mapping=mapping, overlapping_reference_seconds=total,
                      speaker_time_agreement=correct/total if total else None,
                      note='Post-hoc label mapping, approximate reference-time agreement; not DER or independent voice recognition.')
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('asr',type=Path)
    p.add_argument('--diar',type=Path); p.add_argument('--duration',type=float)
    p.add_argument('--reference',type=Path,default=Path('benchmark/reference_fZZXVNt1gk0.txt'))
    a=p.parse_args()
    result=evaluate(json.loads(a.asr.read_text(encoding='utf-8')),reference(a.reference),a.duration,
                    json.loads(a.diar.read_text(encoding='utf-8')) if a.diar else None)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
