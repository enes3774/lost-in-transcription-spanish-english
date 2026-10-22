"""v9 test-time augmentation audio: a speed-perturbed copy of the data dir (Kaldi-style: resample by the factor, so
tempo and pitch change; 1.1 = 10 % faster). Same file names (WAV content; the loaders read by header), the manifest
gets the real new durations so main.py cuts clips over 30 s exactly as for the validation run (kfold/make_tta.py).
usage: stage_tta.py <data_dir> <out_dir> <factor>
"""
import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import load_audio, log, read_manifest  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import soxr  # noqa: E402

src, dst, F = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
(dst / "clips").mkdir(parents=True, exist_ok=True)
rows = read_manifest(src)


def one(r):
    w = load_audio(src / "clips" / r["audio_filename"])
    y = soxr.resample(w, 16000 * F, 16000).astype(np.float32)      # play back F x faster
    sf.write(str(dst / "clips" / r["audio_filename"]), y, 16000, format="WAV", subtype="PCM_16")
    return r["audio_filename"], len(y) / 16000


with ThreadPoolExecutor(16) as ex:
    dur = dict(ex.map(one, rows))
assert len(dur) == len(rows)
with open(dst / "test_metadata.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["audio_filename", "file_duration_seconds", "language"])
    for r in rows:
        w.writerow([r["audio_filename"], f"{dur[r['audio_filename']]:.2f}", r.get("language", "enspa")])
log(f"tta x{F}: {len(rows)} clips, {sum(dur.values()) / 3600:.2f} h")
