"""Shared helpers for the submission stages."""
import csv
import os
import re
import time
from pathlib import Path

T0 = time.time()
SR = 16000
PROMPT = (
    "Transcribe this audio word for word.\n"
    "The speaker code-switches freely between English and Spanish, often inside a single "
    "sentence. Write every word in the language it was actually spoken in. "
    "Never translate between English and Spanish: if a phrase is spoken in English, "
    "keep it in English; if it is spoken in Spanish, keep it in Spanish.\n"
    "Do not summarise, do not rephrase, do not fix grammar, do not add words that were not said.\n"
    "Output only the transcript itself, with no preamble, no quotes and no commentary."
)
PREAMBLE_RE = re.compile(
    r"^\s*(here\s+is\s+(the\s+)?(verbatim\s+)?transcript(ion)?|transcript(ion)?|"
    r"aqu[ií]\s+est[aá]\s+la\s+transcripci[oó]n|transcripci[oó]n)\s*[:\-]\s*", re.IGNORECASE)


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def clean_prediction(text):
    t = PREAMBLE_RE.sub("", (text or "").strip()).strip()
    if len(t) > 1 and t[0] in '"«“' and t[-1] in '"»”':
        t = t[1:-1].strip()
    return " ".join(t.split())


def load_audio(path):
    import numpy as np
    try:
        import soundfile as sf
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
    except Exception:
        import librosa
        wav, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != SR:
        import soxr
        wav = soxr.resample(wav, sr, SR)
    return np.ascontiguousarray(wav, dtype=np.float32)


def read_manifest(data_dir):
    with open(Path(data_dir) / "test_metadata.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if os.environ.get("OMNI_LIMIT"):
        rows = rows[: int(os.environ["OMNI_LIMIT"])]
    return rows


def write_csv(path, rows, preds):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio_filename", "transcript"])
        for r in rows:
            w.writerow([r["audio_filename"], preds.get(r["audio_filename"], "")])
    os.replace(tmp, path)


def read_csv(path):
    try:
        return {r["audio_filename"]: r["transcript"] for r in csv.DictReader(open(path, newline="", encoding="utf-8"))}
    except FileNotFoundError:
        return {}
