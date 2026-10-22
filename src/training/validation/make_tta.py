"""Speed-perturbed copies of lab_all (dev + Miami half + long, 1,662 clips) for test-time-augmentation voters.
Kaldi-style speed perturbation: resample by the factor, so tempo AND pitch change (factor 1.1 = 10 % faster, shorter).
Same file names (WAV content; libsndfile reads by header), test_metadata durations = the real new durations.
usage: make_tta.py <factor> -> $T/tta/lab_all_s<factor*100>/"""
import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

sys.path.insert(0, "/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track/submission_v8_code")
from common import load_audio  # noqa: E402  (same loader as every stage: 16 kHz mono float32)

F = float(sys.argv[1])
SRC = Path("/kaggle/working/omni_rt/lab_all")
DST = Path(f"/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track/tta/lab_all_s{int(round(F * 100)):03d}")
(DST / "clips").mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open(SRC / "test_metadata.csv", encoding="utf-8")))


def one(r):
    w = load_audio(SRC / "clips" / r["audio_filename"])
    y = soxr.resample(w, 16000 * F, 16000).astype(np.float32)      # play back F x faster
    sf.write(str(DST / "clips" / r["audio_filename"]), y, 16000, format="WAV", subtype="PCM_16")
    return r["audio_filename"], len(y) / 16000


with ThreadPoolExecutor(8) as ex:
    dur = dict(ex.map(one, rows))
with open(DST / "test_metadata.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["audio_filename", "file_duration_seconds", "language"])
    for r in rows:
        w.writerow([r["audio_filename"], f"{dur[r['audio_filename']]:.2f}", r.get("language", "enspa")])
print(f"factor {F}: {len(rows)} clips -> {DST}, total {sum(dur.values()) / 3600:.2f} h")
