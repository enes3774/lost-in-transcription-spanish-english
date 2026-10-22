"""Merge a Whisper LoRA checkpoint into the base model in fp32 (bf16 merging rounds the LoRA update away) so the HF pipeline can serve it.
usage: whisper_merge.py <run_dir> <out_dir>   -- picks the epoch-2 checkpoint (second-smallest step), as in the full recipe
"""
import re
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration

BASE = "/home/jovyan/work/Enes/lost_in_transcription/models/checkpoints/whisper-large-v3"
run, out = Path(sys.argv[1]), Path(sys.argv[2])
cks = sorted(run.glob("checkpoint-*"), key=lambda p: int(re.search(r"(\d+)$", p.name).group(1)))
assert len(cks) >= 2, cks
ck = cks[int(sys.argv[3]) if len(sys.argv) > 3 else 1]
print("merging", ck)
model = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(model, str(ck)).merge_and_unload()
model.generation_config.language = "es"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.config.forced_decoder_ids = None
out.mkdir(parents=True, exist_ok=True)
model.save_pretrained(out, safe_serialization=True)
for f in Path(BASE).iterdir():
    if f.suffix in (".json", ".txt") and not (out / f.name).exists() and "model" not in f.name:
        shutil.copy(f, out / f.name)
print("MERGED", out)
