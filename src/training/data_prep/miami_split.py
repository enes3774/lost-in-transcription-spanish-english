"""Speaker-disjoint split of Bangor Miami into A / B / T.

Speakers are identified by pseudonym name from @Participants (CHAT codes are reused, e.g. MAR = María and
Marcela). Unnamed roles (non_participant, Other_Speaker, Investigator, Unidentified) cannot be matched across
recordings, so a conversation's unnamed speakers are treated as local to it. Conversations that share a named
speaker are merged into one component (e.g. all maria* recordings, herring2+herring12 via Miguel); components
are then assigned whole, so no named speaker appears in two groups. Target: T ~10 %, A and B ~45 % of hours.
"""
import collections
import json
import random
import re
import unicodedata
from pathlib import Path

M = Path("/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/raw/miami")
P = Path("/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/miami_prep")
UNNAMED = {"non_participant", "other_speaker", "other", "investigator", "unidentified"}


def norm(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


speakers = {}
for cha in sorted(M.glob("chat/*.cha")):
    line = next(l for l in open(cha, encoding="utf-8", errors="replace") if l.startswith("@Participants"))
    named = set()
    for part in line.split("\t", 1)[1].split(","):
        toks = part.split()
        if len(toks) >= 2 and norm(toks[1]) not in UNNAMED:
            named.add(norm(toks[1]))
    speakers[cha.stem] = named

# union-find over conversations sharing a named speaker
parent = {c: c for c in speakers}


def find(c):
    while parent[c] != c:
        parent[c] = parent[parent[c]]
        c = parent[c]
    return c


by_spk = collections.defaultdict(list)
for c, s in speakers.items():
    for n in s:
        by_spk[n].append(c)
for n, cs in by_spk.items():
    for c in cs[1:]:
        parent[find(c)] = find(cs[0])

clips = [json.loads(l) for l in open(P / "manifest.jsonl", encoding="utf-8")]
ok = [c for c in clips if not re.search(r"\b(xxx|xx|yyy|www)\b", c["text"]) and (P / "clips" / f"{c['id']}.wav").exists()]
hours = collections.Counter()
for c in ok:
    hours[c["conv"]] += c["duration"] / 3600
comps = collections.defaultdict(list)
for c in speakers:
    comps[find(c)].append(c)
comp_list = sorted(comps.values(), key=lambda cs: -sum(hours[c] for c in cs))
total = sum(hours.values())
print(f"{len(speakers)} conversations, {len(by_spk)} named speakers, {len(comp_list)} components, {total:.1f} h usable")
print("multi-conversation components:", [(sorted(cs)[:3] + (["..."] if len(cs) > 3 else []), round(sum(hours[c] for c in cs), 2)) for cs in comp_list if len(cs) > 1])

# T: random single-conversation components until ~10 % of hours; then greedy-balance A/B by hours
rng = random.Random(2026)
singles = [cs for cs in comp_list if len(cs) == 1]
rng.shuffle(singles)
fold = {}
t_hours = 0.0
for cs in singles:
    if t_hours >= 0.10 * total:
        break
    for c in cs:
        fold[c] = "T"
    t_hours += sum(hours[c] for c in cs)
load = {"A": 0.0, "B": 0.0}
for cs in comp_list:
    if cs[0] in fold:
        continue
    g = min(load, key=load.get)
    for c in cs:
        fold[c] = g
    load[g] += sum(hours[c] for c in cs)

# verification: no named speaker in two groups
spk_groups = collections.defaultdict(set)
for c, s in speakers.items():
    for n in s:
        spk_groups[n].add(fold[c])
leaks = {n: g for n, g in spk_groups.items() if len(g) > 1}
assert not leaks, leaks
split = {"conv_fold": fold, "note": __doc__}
for g in ("A", "B", "T"):
    cs = sorted(c for c in fold if fold[c] == g)
    ids = [c["id"] for c in ok if fold[c["conv"]] == g]
    split[g] = ids
    print(f"{g}: {len(cs):2d} conversations, {sum(hours[c] for c in cs):5.2f} h, {len(ids):5d} clips, "
          f"{len({n for c in cs for n in speakers[c]})} named speakers | {cs}")
print("speaker leaks across groups:", len(leaks))
json.dump(split, open(P / "split_spk3.json", "w"), indent=1)
print("wrote", P / "split_spk3.json")
