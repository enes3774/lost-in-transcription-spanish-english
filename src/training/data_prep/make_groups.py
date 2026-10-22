"""Stage-compatible data dirs for Miami groups A, B, T and a fixed 600-clip early-stopping sample of T.
Each dir: test_metadata.csv (audio_filename=<id>.wav), refs.csv, clips -> miami_prep/clips (symlink)."""
import csv
import json
import os
import random
from pathlib import Path

P = Path("/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/miami_prep")
K = Path("/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/miami_kfold")
split = json.load(open(P / "split_spk3.json"))
man = {c["id"]: c for c in map(json.loads, open(P / "manifest.jsonl", encoding="utf-8"))}


def write(name, ids):
    d = K / name
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "clips").exists():
        os.symlink(P / "clips", d / "clips")
    ids = sorted(ids)
    with open(d / "test_metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio_filename", "file_duration_seconds", "language"])
        for i in ids:
            w.writerow([f"{i}.wav", round(man[i]["duration"], 2), "enspa"])
    with open(d / "refs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio_filename", "transcript"])
        for i in ids:
            w.writerow([f"{i}.wav", man[i]["text"]])
    print(name, len(ids), "clips", round(sum(man[i]["duration"] for i in ids) / 3600, 2), "h")


for g in ("A", "B", "T"):
    write(f"miami_{g}", split[g])
# early-stopping sample: 100 clips of >=1 s from each of the 6 T conversations
rng = random.Random(11)
by_conv = {}
for i in split["T"]:
    if man[i]["duration"] >= 1.0:
        by_conv.setdefault(man[i]["conv"], []).append(i)
val = [i for cs in by_conv.values() for i in rng.sample(sorted(cs), min(100, len(cs)))]
write("miami_Tval600", val)
