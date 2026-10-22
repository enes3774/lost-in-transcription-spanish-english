"""Stress set: glue dev clips into long recordings (targets ~60, ~120, ~220 s) to mimic long recordings. Reference = concatenated references.
Every dev clip is used exactly once per target length, so WER is comparable with the original dev score."""
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf

DEV = Path("/kaggle/working/omni_rt/dev")
OUT = Path("/kaggle/working/omni_rt/dev_long")
(OUT / "clips").mkdir(parents=True, exist_ok=True)
refs = {r["audio_filename"]: r["transcript"] for r in csv.DictReader(open(DEV / "refs.csv", encoding="utf-8"))}
dur = {r["audio_filename"]: float(r["file_duration_seconds"]) for r in csv.DictReader(open(DEV / "test_metadata.csv"))}
names = sorted(refs)
random.Random(0).shuffle(names)
meta, out_refs = [], []
for target in (60, 120, 220):
    group, total, gi = [], 0.0, 0
    def flush():
        global group, total, gi
        if not group:
            return
        wavs = []
        for n in group:
            w, sr = sf.read(DEV / "clips_wav" / (n[:-4] + ".wav"), dtype="float32", always_2d=True)
            assert sr == 16000, sr
            wavs += [w.mean(1), np.zeros(int(0.4 * sr), dtype="float32")]   # 0.4 s pause between messages
        name = f"long{target}_{gi:02d}.wav"
        sf.write(OUT / "clips" / name, np.concatenate(wavs[:-1]), 16000)
        meta.append((name, round(sum(len(x) for x in wavs[:-1]) / 16000), "enspa"))
        out_refs.append((name, " ".join(refs[n].strip() for n in group)))
        group, total, gi = [], 0.0, gi + 1
    for n in names:
        group.append(n); total += dur[n]
        if total >= target:
            flush()
    flush()
with open(OUT / "test_metadata.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["audio_filename", "file_duration_seconds", "language"]); w.writerows(meta)
with open(OUT / "refs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["audio_filename", "transcript"]); w.writerows(out_refs)
for t in (60, 120, 220):
    d = [m[1] for m in meta if m[0].startswith(f"long{t}_")]
    print(f"target {t:3d}s: {len(d)} clips, {min(d)}-{max(d)} s")
