"""Regenerate the Miami clip texts with the competition's false-start convention: a CHAT phonological fragment
'&gro' becomes 'gro...' (the dev references write cut-off words this way; the original miami_prep.py turned it into a
bare 'gro'). Vowel-only '&e' / '&a' are hesitation fillers in this corpus and become 'eh' / 'ah'; other '&x' single
vowels stay bare as before. Everything else is identical to miami_prep.py, so the clip ids and boundaries match.
Writes miami_prep/manifest_frag.jsonl and refs_frag.csv next to every miami_kfold/*/refs.csv, and checks that
each new text equals the old one once the markers are removed."""
import collections
import csv
import json
import re
from pathlib import Path

M = Path('/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/raw/miami')
P = Path('/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/miami_prep')
K = Path('/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/miami_kfold')
MAX_S, GAP_S = 20.0, 2.0
BULLET = re.compile('\x15(\\d+)_(\\d+)\x15')
UNINTEL = re.compile(r'(^|\s)(xxx|xx|yyy|www)(@s:[\w&+]+)?(?=\s|$)')
FILLER = {'uh', 'um', 'mm', 'hmm', 'mhm', 'eh', 'ah', 'er', 'em', 'hm'}
VOWEL_FILLER = {'e': 'eh', 'a': 'ah'}
# variant 2 (argv[1] == "frag2"): the CHAT vowel hesitations '&e' / '&a' are DROPPED instead of written as eh/ah.
# Measured on dev 2026-09-23: writing them as "eh" doubled the wrongly inserted "eh" of every system trained on
# those targets (Voxtral 3 -> 10, Whisper 4 -> 9, refiner 6 -> 13); the competition annotator does not transcribe
# these faint hesitations (the user listened to sastre9_KAY_00090942: no audible filler where CHAT has '&e').
import sys as _sys
VARIANT = _sys.argv[1] if len(_sys.argv) > 1 else 'frag'
stats = collections.Counter()


def amp(m):
    w = m.group(2)
    lw = w.lower()
    if lw in FILLER:
        stats['filler'] += 1
        return m.group(1) + w
    if lw in VOWEL_FILLER:
        stats['vowel_filler'] += 1
        return m.group(1) + '§' if VARIANT == 'frag2' else m.group(1) + VOWEL_FILLER[lw]   # placeholder keeps the utterance alive, stripped after merging
    if re.fullmatch(r'[aeiouáéíóú]+', lw):
        stats['vowel_bare'] += 1
        return m.group(1) + '§' if VARIANT == 'frag2' else m.group(1) + w
    stats['fragment'] += 1
    return m.group(1) + w + '...'


def clean(text):
    t = text
    t = re.sub(r'&=\S+', ' ', t)
    t = re.sub(r'\[[^\]]*\]', ' ', t)
    t = re.sub(r'\+[/."^<,!?]+', ' ', t)
    t = re.sub(r'\(\.{1,3}\)', ' ', t)
    t = re.sub(r'@s:[\w&+]+', '', t)
    t = re.sub(r'@\w+', '', t)
    t = re.sub(r'(^|\s)&([A-Za-z][A-Za-z\'áéíóúñü]*)', amp, t)     # <- the only change vs miami_prep.py
    t = re.sub(r'(^|\s)&([A-Za-z])', r'\1\2', t)
    t = re.sub(r'(^|\s)0\S*', ' ', t)
    t = t.replace('<', ' ').replace('>', ' ')
    t = re.sub(r'\((\w+)\)', r'\1', t)
    t = re.sub(r'(\w):+', r'\1', t)
    t = re.sub(r'[#^~+↑↓]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    t = re.sub(r'\s+([.?!,])', r'\1', t)
    return t


segments = []
for cha in sorted(M.glob('chat/*.cha')):
    conv = cha.stem
    utts, cur = [], None
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
        m = BULLET.search(raw)
        if not m:
            continue
        s, e = int(m.group(1)) / 1000, int(m.group(2)) / 1000
        body = BULLET.sub(' ', raw)
        if UNINTEL.search(body):
            continue
        t = clean(body)
        words = re.sub(r'[.?!,]', '', t).split()
        if not words or e - s < 0.3 or e - s > 40:
            continue
        segments.append({'conv': conv, 'spk': spk, 'start': s, 'end': e, 'text': t})

clips, i = [], 0
while i < len(segments):
    a = dict(segments[i])
    j = i + 1
    while j < len(segments) and segments[j]['conv'] == a['conv'] and segments[j]['spk'] == a['spk'] \
            and segments[j]['start'] - a['end'] <= GAP_S and segments[j]['end'] - a['start'] <= MAX_S:
        a['end'] = segments[j]['end']
        a['text'] += ' ' + segments[j]['text']
        j += 1
    a['duration'] = round(a['end'] - a['start'], 3)
    a['id'] = f"{a['conv']}_{a['spk']}_{int(a['start'] * 1000):08d}"
    clips.append(a)
    i = j

if VARIANT == 'frag2':   # drop the hesitation placeholders now that clip boundaries are fixed
    for c in clips:
        x = re.sub(r'§', '', c['text'])
        x = re.sub(r'\s+([.?!,])', r'', re.sub(r'\s+', ' ', x)).strip()
        c['text'] = x
old = {c['id']: c for c in map(json.loads, open(P / 'manifest.jsonl', encoding='utf-8'))}
new = {c['id']: c for c in clips}
assert set(old) == set(new), (len(old), len(new), len(set(old) ^ set(new)))
strip = lambda t: re.sub(r'\.\.\.', '', t)
bad = 0
for k in old:
    o, n = old[k]['text'], new[k]['text']
    n2 = strip(n)
    # the only other change: 'e'->'eh', 'a'->'ah' for vowel fillers, so compare after undoing that is not exact; compare fragments only
    if strip(re.sub(r'\b(eh|ah)\b', lambda m: m.group(1)[0], n)) != o and n2 != o:
        bad += 1
        if bad <= 5:
            print('DIFF', k, '\n  old:', o[:120], '\n  new:', n[:120])
print(f'{len(clips)} clips; ids identical; texts differing beyond markers: {bad}; marker stats {dict(stats)}')
n_frag = sum(len(re.findall(r'\S\.\.\.(?=\s|$)', c['text'])) for c in clips)
n_words = sum(len(c['text'].split()) for c in clips)
print(f'fragments in targets: {n_frag} / {n_words} words = {n_frag / n_words:.4%}')
with open(P / f'manifest_{VARIANT}.jsonl', 'w', encoding='utf-8') as f:
    for c in clips:
        f.write(json.dumps(c, ensure_ascii=False) + '\n')
for d in sorted(K.glob('miami_*')):
    if not (d / 'refs.csv').exists():
        continue
    rows = list(csv.DictReader(open(d / 'refs.csv', encoding='utf-8')))
    with open(d / f'refs_{VARIANT}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['audio_filename', 'transcript'])
        for r in rows:
            w.writerow([r['audio_filename'], new[r['audio_filename'][:-4]]['text']])
    nf = sum(len(re.findall(r'\S\.\.\.(?=\s|$)', new[r['audio_filename'][:-4]]['text'])) for r in rows)
    print(f'  {d.name}: {len(rows)} rows, {nf} fragments -> refs_{VARIANT}.csv')
print('FRAG_REFS_DONE')
