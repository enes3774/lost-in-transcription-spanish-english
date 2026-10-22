"""Qwen3-ASR-1.7B (Miami-merged) stage via the qwen_asr package (HF backend), optional beam search.
usage: stage_qwen3asr.py <model_dir> <data_dir> <out_csv> [num_beams]
"""
import functools
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import clean_prediction, log, read_manifest, write_csv  # noqa: E402

import torch  # noqa: E402
from qwen_asr import Qwen3ASRModel  # noqa: E402

model_dir, data_dir, out_csv = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
beams = int(sys.argv[4]) if len(sys.argv) > 4 else 4
rows = read_manifest(data_dir)
order = sorted(rows, key=lambda r: -float(r.get("file_duration_seconds") or 0))
preds = {}
write_csv(out_csv, rows, preds)
t0 = time.time()
qm = Qwen3ASRModel.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda:0", max_inference_batch_size=8)
qm.model.eval()
if beams > 1:
    qm.model.generate = functools.partial(qm.model.generate, num_beams=beams, do_sample=False)
log(f"qwen3asr ready (beams={beams})")
chunk = 64
with torch.no_grad():
    for ci in range(0, len(order), chunk):
        part = order[ci:ci + chunk]
        res = qm.transcribe(audio=[str(data_dir / "clips" / r["audio_filename"]) for r in part])
        for r, x in zip(part, res):
            preds[r["audio_filename"]] = clean_prediction(getattr(x, "text", None) or "")
        write_csv(out_csv, rows, preds)
        if (ci // chunk) % 5 == 0:
            log(f"qwen3asr chunk {ci // chunk + 1}/{(len(order) + chunk - 1) // chunk}")
write_csv(out_csv, rows, preds)
log(f"qwen3asr stage done in {time.time() - t0:.0f}s")
