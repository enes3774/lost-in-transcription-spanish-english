"""Make long clips safe for models that break on them (Whisper's chunk stitching, Qwen3-ASR truncation,
conformer position-table bug): cut every clip longer than LIMIT at the quietest point inside each window,
decode the pieces with the unchanged stage script, then join the texts back in order.

  seg_long.py split <data_dir> <out_dir> [limit_s=30] [max_piece_s=28] [min_piece_s=12]
  seg_long.py merge <out_dir> <pieces_pred.csv> <merged_pred.csv>

<out_dir> gets test_metadata.csv, clips/ (wav pieces; short clips are converted once to wav too) and mapping.json.
Pure numpy + soundfile; no model needed.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

SR = 16000


def load(path):
    import soundfile as sf
    try:
        w, sr = sf.read(str(path), dtype="float32", always_2d=True)
        w = w.mean(1)
    except Exception:
        import librosa
        w, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != SR:
        import soxr
        w = soxr.resample(w, sr, SR)
    return np.ascontiguousarray(w, dtype=np.float32)


def cut_points(w, max_piece, min_piece):
    """Greedy: from the current start, look in [min_piece, max_piece] and cut at the lowest-energy 0.3 s window."""
    hop = int(0.02 * SR)
    frames = len(w) // hop
    e = np.array([float(np.mean(w[i * hop:(i + 1) * hop] ** 2)) for i in range(frames)]) if frames else np.zeros(0)
    k = 15                                                     # 0.3 s smoothing
    es = np.convolve(e, np.ones(k) / k, mode="same") if len(e) >= k else e
    cuts, start = [], 0
    while (len(w) - start) / SR > max_piece:
        lo = start + int(min_piece * SR)
        hi = start + int(max_piece * SR)
        a, b = lo // hop, min(hi // hop, len(es) - 1)
        c = (a + int(np.argmin(es[a:b]))) * hop if b > a else hi
        cuts.append(c)
        start = c
    return cuts


def split(data_dir, out_dir, limit=30.0, max_piece=28.0, min_piece=12.0):
    import soundfile as sf
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    (out_dir / "clips").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(data_dir / "test_metadata.csv", encoding="utf-8")))
    mapping, meta, n_long = {}, [], 0
    for r in rows:
        name = r["audio_filename"]
        w = load(data_dir / "clips" / name)
        stem = Path(name).stem
        if len(w) / SR <= limit:
            pieces = [w]
        else:
            n_long += 1
            idx = [0] + cut_points(w, max_piece, min_piece) + [len(w)]
            pieces = [w[a:b] for a, b in zip(idx, idx[1:]) if b - a > int(0.2 * SR)]
        names = []
        for i, p in enumerate(pieces):
            pn = f"{stem}__p{i:02d}.wav"
            sf.write(out_dir / "clips" / pn, p, SR)
            names.append(pn)
            meta.append({"audio_filename": pn, "file_duration_seconds": max(1, round(len(p) / SR)), "language": r.get("language", "enspa")})
        mapping[name] = names
    with open(out_dir / "test_metadata.csv", "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=["audio_filename", "file_duration_seconds", "language"])
        wri.writeheader(); wri.writerows(meta)
    json.dump(mapping, open(out_dir / "mapping.json", "w"))
    longest = max(float(m["file_duration_seconds"]) for m in meta)
    print(f"{len(rows)} clips ({n_long} longer than {limit:.0f}s) -> {len(meta)} pieces, longest piece {longest:.0f}s")


def merge(out_dir, pieces_csv, merged_csv):
    mapping = json.load(open(Path(out_dir) / "mapping.json"))
    pred = {r["audio_filename"]: r["transcript"] for r in csv.DictReader(open(pieces_csv, encoding="utf-8"))}
    with open(merged_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio_filename", "transcript"])
        for name, parts in mapping.items():
            w.writerow([name, " ".join(" ".join((pred.get(p) or "").split()) for p in parts).strip()])
    print(f"merged {len(mapping)} clips -> {merged_csv}")


if __name__ == "__main__":
    if sys.argv[1] == "split":
        split(sys.argv[2], sys.argv[3], *[float(x) for x in sys.argv[4:7]])
    else:
        merge(sys.argv[2], sys.argv[3], sys.argv[4])
