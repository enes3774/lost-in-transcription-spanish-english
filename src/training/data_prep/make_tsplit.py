"""Split the held-out T conversations into a selection half and an untouched test half.
Tsel  = herring16, sastre12, zeledon4  -> miami_Tsel600 (200 clips >= 1 s per conversation) for early stopping
Ttest = herring3,  sastre13, zeledon6  -> miami_Ttest (all clips) — never used for any choice.
T conversations are single-conversation speaker components, so the halves share no named speaker."""
import csv
import json
import os
import random
from pathlib import Path

P = Path("/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/miami_prep")
K = Path("/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/miami_kfold")
SEL = ["herring16", "sastre12", "zeledon4"]
TEST = ["herring3", "sastre13", "zeledon6"]
split = json.load(open(P / "split_spk3.json"))
assert all(split["conv_fold"][c] == "T" for c in SEL + TEST)
man = {}
for l in open(P / "manifest.jsonl", encoding="utf-8"):
    r = json.loads(l)
    man[r["id"]] = r
ck = next(k for k in ("conv", "conversation", "file") if k in next(iter(man.values())))
refs = {r["audio_filename"]: r["transcript"] for r in csv.DictReader(open(K / "miami_T" / "refs.csv", encoding="utf-8"))}
meta = {r["audio_filename"]: r for r in csv.DictReader(open(K / "miami_T" / "test_metadata.csv"))}


def write(name, files):
    d = K / name
    d.mkdir(exist_ok=True)
    if not (d / "clips").exists():
        os.symlink(P / "clips", d / "clips")
    with open(d / "test_metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(meta.values())).keys()))
        w.writeheader()
        for n in files:
            w.writerow(meta[n])
    with open(d / "refs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio_filename", "transcript"])
        for n in files:
            w.writerow([n, refs[n]])
    hrs = sum(float(meta[n]["file_duration_seconds"]) for n in files) / 3600
    print(f"{name}: {len(files)} clips, {hrs:.2f} h")


by_conv = {}
for n in refs:
    by_conv.setdefault(man[n[:-4]][ck], []).append(n)
rng = random.Random(0)
sel = []
for c in SEL:
    pool = sorted(n for n in by_conv[c] if float(meta[n]["file_duration_seconds"]) >= 1.0 and refs[n].strip())
    rng.shuffle(pool)
    sel += sorted(pool[:200])
write("miami_Tsel600", sel)
write("miami_Ttest", sorted(n for c in TEST for n in by_conv[c]))
spk = lambda convs: {man[n[:-4]].get("speaker_name") or man[n[:-4]].get("spk") for c in convs for n in by_conv[c]}
print("conversations per half:", {c: len(by_conv[c]) for c in SEL + TEST})
