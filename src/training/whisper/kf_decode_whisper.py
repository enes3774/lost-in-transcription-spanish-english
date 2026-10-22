"""Decode a data dir with the handed-over Whisper large-v3 + Miami LoRA (merged), matching their settings:
HF pipeline, fp16, chunk_length_s=30, stride (5,5), task=transcribe, language auto.
usage: kf_decode_whisper.py <model_dir> <data_dir> <out_csv> [batch]
"""
import csv
import os
import sys
import time
from pathlib import Path

import torch
from transformers import pipeline

sys.path.insert(0, "/kaggle/working/omni_rt/sub")
from main import load_audio  # noqa: E402  soundfile/librosa reader, resamples to 16 kHz (handles mp3)

MODEL, DATA, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
BS = int(sys.argv[4]) if len(sys.argv) > 4 else 16
rows = list(csv.DictReader(open(DATA / "test_metadata.csv", encoding="utf-8")))
done = {}
if OUT.exists():
    done = {r["audio_filename"]: r["transcript"] for r in csv.DictReader(open(OUT, encoding="utf-8"))}
todo = [r for r in rows if r["audio_filename"] not in done]
print(f"{len(rows)} clips, {len(todo)} to decode, batch {BS}", flush=True)

asr = pipeline("automatic-speech-recognition", model=MODEL, device=0, torch_dtype=torch.float16,
               chunk_length_s=30, stride_length_s=(5, 5))
t0 = time.time()
for i in range(0, len(todo), BS * 6):
    part = todo[i:i + BS * 6]
    inputs = [{"raw": load_audio(DATA / "clips" / r["audio_filename"]), "sampling_rate": 16000} for r in part]
    out = asr(inputs, batch_size=BS, generate_kwargs={"task": "transcribe"})
    for r, o in zip(part, out):
        done[r["audio_filename"]] = " ".join((o["text"] or "").split())
    tmp = OUT.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio_filename", "transcript"])
        for r in rows:
            w.writerow([r["audio_filename"], done.get(r["audio_filename"], "")])
    os.replace(tmp, OUT)
    n = min(i + len(part), len(todo))
    print(f"{time.strftime('%H:%M:%S')} {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
print("WHISPER_DECODE_DONE", flush=True)
