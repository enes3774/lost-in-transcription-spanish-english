"""Shared helpers for the k-fold trainers."""
import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/kaggle/working/omni_rt")
from score import normalize_text  # noqa: E402

P = Path("/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/miami_prep")
K = Path("/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/miami_kfold")
RUNS = Path("/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track/kfold")
XXX = re.compile(r"\b(xxx|xx|yyy|www)\b")


def log(m):
    print(time.strftime("%H:%M:%S ") + m, flush=True)


def manifest():
    return {c["id"]: c for c in map(json.loads, open(P / _os.environ.get("KF_MANIFEST", "manifest.jsonl"), encoding="utf-8"))}


def group_ids(g):
    sp = json.load(open(P / "split_spk3.json"))
    return sp["A"] + sp["B"] if g == "AB" else sp[g]


def val_rows():
    import os
    d = K / os.environ.get("KF_VAL", "miami_Tval600")
    rows = list(csv.DictReader(open(d / "test_metadata.csv")))
    refs = {r["audio_filename"]: r["transcript"] for r in csv.DictReader(open(d / os.environ.get("KF_REFS", "refs.csv"), encoding="utf-8"))}
    return d, rows, refs


def wer(refs, preds, names):
    import jiwer
    fix = lambda t: re.sub(r"\bo_k\b", "okay", t or "")
    return jiwer.wer([normalize_text(fix(refs[n])) for n in names], [normalize_text(fix(preds.get(n, ""))) for n in names])


class EarlyStop:
    """Stop when validation WER has not improved for `patience` consecutive evaluations."""

    def __init__(self, out_dir, patience):
        self.best, self.best_step, self.bad, self.patience = 9.9, None, 0, patience
        self.hist_path = Path(out_dir) / "val_history.json"
        self.hist = []

    def update(self, step, wer_value):
        self.hist.append({"step": step, "val_wer": round(wer_value, 5)})
        json.dump({"history": self.hist, "best_step": self.best_step if wer_value >= self.best else step},
                  open(self.hist_path, "w"), indent=1)
        if wer_value < self.best - 1e-4:
            self.best, self.best_step, self.bad = wer_value, step, 0
            return True, False
        self.bad += 1
        return False, self.bad >= self.patience


# ---- optional external-data mix (CIEMPIESS) -------------------------------------------------------
import os as _os
import random as _random
EXTRA_MANIFEST = Path(_os.environ.get("KF_EXTRA_MANIFEST", "/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/ciempiess_prep/manifest_light.jsonl"))
TAG = _os.environ.get("KF_TAG", "")
EXTRA_FRAC = float(_os.environ.get("KF_EXTRA_FRAC", "0"))


def training_items(G, min_s, max_s, seed=0):
    """Miami group G clips, plus CIEMPIESS clips making up EXTRA_FRAC of the final list (shuffled)."""
    man = manifest()
    items = [{"key": i, "wav": str(P / "clips" / f"{i}.wav"), "text": man[i]["text"], "duration": man[i]["duration"], "extra": False}
             for i in group_ids(G) if min_s <= man[i]["duration"] <= max_s and not XXX.search(man[i]["text"])]
    if EXTRA_FRAC > 0:
        ex = [json.loads(l) for l in open(EXTRA_MANIFEST, encoding="utf-8")]
        ex = [e for e in ex if min_s <= e["duration"] <= max_s]
        _random.Random(seed + 1).shuffle(ex)
        n = int(len(items) * EXTRA_FRAC / (1 - EXTRA_FRAC))
        items += [{"key": e["id"], "wav": e["wav"], "text": e["text"], "duration": e["duration"], "extra": True} for e in ex[:n]]
    _random.Random(seed).shuffle(items)
    log(f"training items: {sum(not x['extra'] for x in items)} Miami + {sum(x['extra'] for x in items)} CIEMPIESS (tag '{TAG}')")
    return items


# ---- phone-style augmentation (KF_AUG=1) ----------------------------------------------------------
AUG = _os.environ.get("KF_AUG", "0") == "1"
SR = 16000


def augment(wav, rng):
    """Simulate the recording path of the evaluation audio. Input/output: float32 mono at 16 kHz."""
    import numpy as np
    import soxr
    if not AUG or wav.size < SR // 4:
        return wav
    w = wav.astype("float32").copy()
    if rng.random() < 0.5:                                  # tempo / pitch via resampling
        f = rng.uniform(0.9, 1.1)
        w = soxr.resample(w, SR, int(SR * f)).astype("float32")
    if rng.random() < 0.6:                                  # telephone band via 8 kHz round trip
        w = soxr.resample(soxr.resample(w, SR, 8000), 8000, SR).astype("float32")
    if rng.random() < 0.3:                                  # coarser codec-ish round trip
        r = rng.choice([11025, 12000, 22050])
        w = soxr.resample(soxr.resample(w, SR, r), r, SR).astype("float32")
    if rng.random() < 0.6:                                  # additive noise, white or pink
        snr = rng.uniform(8.0, 30.0)
        n = rng.standard_normal(len(w)).astype("float32")
        if rng.random() < 0.5:
            n = np.cumsum(n).astype("float32")
            n -= n.mean()
        p_s = float(np.mean(w ** 2)) + 1e-12
        p_n = float(np.mean(n ** 2)) + 1e-12
        w = w + n * float(np.sqrt(p_s / (p_n * 10 ** (snr / 10))))
    if rng.random() < 0.25:                                 # short room reverb
        t = rng.uniform(0.08, 0.3)
        ir = (rng.standard_normal(int(SR * t)) * np.exp(-np.arange(int(SR * t)) / (SR * t / 4))).astype("float32")
        ir[0] = 1.0
        w = np.convolve(w, ir / (np.abs(ir).sum() + 1e-9), mode="same").astype("float32")
    if rng.random() < 0.3:                                  # packet loss / dropouts
        for _ in range(rng.integers(1, 4)):
            n = int(SR * rng.uniform(0.02, 0.06))
            if len(w) > n + 1:
                i = int(rng.integers(0, len(w) - n))
                w[i:i + n] *= rng.uniform(0.0, 0.2)
    g = rng.uniform(0.3, 1.6)                               # gain, with occasional clipping
    w = np.clip(w * g, -1.0, 1.0).astype("float32")
    m = float(np.abs(w).max())
    if m > 1e-6:
        w = (w / m * 0.95).astype("float32")
    return np.ascontiguousarray(w)


def augment_to_file(path, rng, tmp):
    """Voxtral takes a file path, so write the augmented copy to tmp and return its path."""
    if not AUG:
        return str(path)
    import soundfile as sf
    w, sr = sf.read(str(path), dtype="float32", always_2d=True)
    w = w.mean(1)
    if sr != SR:
        import soxr
        w = soxr.resample(w, sr, SR)
    sf.write(tmp, augment(w, rng), SR)
    return tmp
