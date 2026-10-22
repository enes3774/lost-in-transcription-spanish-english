"""Whisper large-v3 (Miami LoRA merged) stage. HF pipeline, fp16, language/task from the model's generation config.
usage: stage_whisper.py <model_dir> <data_dir> <out_csv> [batch]

main.py hands this stage a data dir whose clips are already <= 28 s (seg_long.py), because the pipeline's own
30-second chunk stitching drops ~20% of the words on long recordings. chunk_length_s stays set only as a safety net.
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")            # the runtime image ships TensorFlow/Keras 3; importing TF Whisper crashes
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_audio, log, read_manifest, write_csv  # noqa: E402

import torch  # noqa: E402
from transformers import pipeline  # noqa: E402

model_dir, data_dir, out_csv = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
bs = int(sys.argv[4]) if len(sys.argv) > 4 else 16
rows = read_manifest(data_dir)
order = sorted(rows, key=lambda r: -float(r.get("file_duration_seconds") or 0))
preds = {}
write_csv(out_csv, rows, preds)
t0 = time.time()
asr = pipeline("automatic-speech-recognition", model=model_dir, framework="pt", device=0, torch_dtype=torch.float16,
               chunk_length_s=30, stride_length_s=(5, 5))
log(f"whisper ready in {time.time() - t0:.0f}s, {len(rows)} clips, batch {bs}")
step = bs * 6
for k in range(0, len(order), step):
    part = order[k:k + step]
    inputs = [{"raw": load_audio(data_dir / "clips" / r["audio_filename"]), "sampling_rate": 16000} for r in part]
    out = asr(inputs, batch_size=bs, generate_kwargs={"task": "transcribe"})   # STRICT_V8: no smaller-batch retry
    for r, o in zip(part, out):
        preds[r["audio_filename"]] = " ".join((o.get("text") or "").split())
    write_csv(out_csv, rows, preds)
    log(f"whisper {min(k + step, len(order))}/{len(order)}")
log(f"whisper stage done in {time.time() - t0:.0f}s")
