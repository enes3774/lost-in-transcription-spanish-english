"""Save the fp32 Cohere Transcribe checkpoint that cohere_train.py writes (<run>/best) as bf16, which is the dtype the
inference stage loads anyway; this halves the file (8.2 GB -> 4.1 GB). The shipped models/cohere is
cohere/AB_frag_lr1e5/best saved this way. usage: save_bf16.py <best_dir> <out_dir>"""
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
m = AutoModelForSpeechSeq2Seq.from_pretrained(str(src), trust_remote_code=True, local_files_only=True, dtype=torch.float32)
m = m.to(torch.bfloat16)
dst.mkdir(parents=True, exist_ok=True)
m.save_pretrained(str(dst), safe_serialization=True)
for f in src.iterdir():                      # tokenizer, processor and the remote-code .py files travel with the weights
    if f.is_file() and f.name not in ("model.safetensors", "config.json") and not (dst / f.name).exists():
        shutil.copy(f, dst / f.name)
print("saved", sorted(p.name for p in dst.iterdir()))
