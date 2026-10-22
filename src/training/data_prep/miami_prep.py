"""Bangor Miami CHAT -> training segments.

Parses each .cha, cleans CHAT markup to plain verbatim text (keeps repeats/false starts and
filler words, drops events/comments/pauses, drops utterances with unintelligible xxx), then
merges consecutive same-speaker utterances into <= MAX_S clips. Writes a manifest and,
with --cut, the 16 kHz mono wav clips.
"""
import argparse
import collections
import json
import re
import subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--cut', action='store_true')
ap.add_argument('--max_s', type=float, default=20.0)
ap.add_argument('--gap_s', type=float, default=2.0)
args = ap.parse_args()

M = Path('/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/raw/miami')
OUT = Path('/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/miami_prep')
OUT.mkdir(parents=True, exist_ok=True)
BULLET = re.compile('\x15(\\d+)_(\\d+)\x15')
UNINTEL = re.compile(r'(^|\s)(xxx|xx|yyy|www)(@s:[\w&+]+)?(?=\s|$)')


def clean(text):
    t = text
    t = re.sub(r'&=\S+', ' ', t)                      # &=laughs events
    t = re.sub(r'\[[^\]]*\]', ' ', t)                  # [/] [//] [?] [= x] [* x] comments/retrace markers
    t = re.sub(r'\+[/."^<,!?]+', ' ', t)               # +... +//. +" +^ trailing markers
    t = re.sub(r'\(\.{1,3}\)', ' ', t)                 # (.) pauses
    t = re.sub(r'@s:[\w&+]+', '', t)                   # language tags
    t = re.sub(r'@\w+', '', t)                         # other @ tags
    t = re.sub(r'(^|\s)&([A-Za-z])', r'\1\2', t)       # &uh fillers -> uh
    t = re.sub(r'(^|\s)0\S*', ' ', t)                  # omitted words 0word
    t = t.replace('<', ' ').replace('>', ' ')
    t = re.sub(r'\((\w+)\)', r'\1', t)                 # (be)cause -> because
    t = re.sub(r'(\w):+', r'\1', t)                    # lengthening ye:s
    t = re.sub(r'[#^~+↑↓]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    t = re.sub(r'\s+([.?!,])', r'\1', t)
    return t


stats = collections.Counter()
segments = []
for cha in sorted(M.glob('chat/*.cha')):
    conv = cha.stem
    utts = []
    cur = None
    for line in open(cha, encoding='utf-8', errors='replace'):
        line = line.rstrip('\n')
        if line.startswith('*'):
            if cur:
                utts.append(cur)
            spk, txt = line[1:].split(':', 1)
            cur = [spk.strip(), txt.strip()]
        elif line.startswith('\t') and cur:
            cur[1] += ' ' + line.strip()
        else:
            if cur:
                utts.append(cur)
            cur = None
    if cur:
        utts.append(cur)
    for spk, raw in utts:
        stats['utts'] += 1
        m = BULLET.search(raw)
        if not m:
            stats['no_timing'] += 1
            continue
        s, e = int(m.group(1)) / 1000, int(m.group(2)) / 1000
        body = BULLET.sub(' ', raw)
        if UNINTEL.search(body):
            stats['unintelligible'] += 1
            continue
        t = clean(body)
        words = re.sub(r'[.?!,]', '', t).split()
        if not words or e - s < 0.3 or e - s > 40:
            stats['empty_or_bad_len'] += 1
            continue
        segments.append({'conv': conv, 'spk': spk, 'start': s, 'end': e, 'text': t})
        stats['kept_utts'] += 1

# merge consecutive same-speaker utterances
clips = []
i = 0
while i < len(segments):
    a = dict(segments[i])
    j = i + 1
    while j < len(segments) and segments[j]['conv'] == a['conv'] and segments[j]['spk'] == a['spk'] \
            and segments[j]['start'] - a['end'] <= args.gap_s and segments[j]['end'] - a['start'] <= args.max_s:
        a['end'] = segments[j]['end']
        a['text'] += ' ' + segments[j]['text']
        j += 1
    a['duration'] = round(a['end'] - a['start'], 3)
    a['id'] = f"{a['conv']}_{a['spk']}_{int(a['start'] * 1000):08d}"
    clips.append(a)
    i = j

durs = [c['duration'] for c in clips]
words = sum(len(c['text'].split()) for c in clips)
print(dict(stats))
print(f"clips {len(clips)}  hours {sum(durs) / 3600:.2f}  mean {sum(durs) / len(durs):.1f}s  words {words}  convs {len(set(c['conv'] for c in clips))}  speakers {len(set((c['conv'], c['spk']) for c in clips))}")
print('duration bins:', {b: sum(1 for d in durs if lo <= d < hi) for b, (lo, hi) in {'<3s': (0, 3), '3-10': (3, 10), '10-20': (10, 20), '>20': (20, 99)}.items()})
for c in clips[100:106]:
    print(f"  [{c['duration']:.1f}s] {c['text'][:150]}")
with open(OUT / 'manifest.jsonl', 'w', encoding='utf-8') as f:
    for c in clips:
        f.write(json.dumps(c, ensure_ascii=False) + '\n')

if args.cut:
    (OUT / 'clips').mkdir(exist_ok=True)
    by_conv = collections.defaultdict(list)
    for c in clips:
        by_conv[c['conv']].append(c)
    import numpy as np
    import soundfile as sf
    for conv, cs in by_conv.items():
        wav = subprocess.run(['ffmpeg', '-v', 'quiet', '-i', str(M / 'audios' / f'{conv}.mp3'), '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'],
                             capture_output=True).stdout
        x = np.frombuffer(wav, dtype=np.float32)
        for c in cs:
            seg = x[int(c['start'] * 16000):int(c['end'] * 16000)]
            sf.write(OUT / 'clips' / f"{c['id']}.wav", seg, 16000)
        print(conv, len(cs), flush=True)
    print('CUT_DONE')
