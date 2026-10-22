"""Save the merged Whisper as fp16 (the stage loads it in fp16 anyway, so this only halves the package size)."""
import shutil, sys
from pathlib import Path
import torch
from transformers import WhisperForConditionalGeneration
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
m = WhisperForConditionalGeneration.from_pretrained(str(src), torch_dtype=torch.float32)
gc = m.generation_config
m = m.half()
m.generation_config = gc
dst.mkdir(parents=True, exist_ok=True)
m.save_pretrained(dst, safe_serialization=True, max_shard_size="5GB")
for f in src.iterdir():
    if f.is_file() and "safetensors" not in f.name and not (dst / f.name).exists():
        shutil.copy(f, dst / f.name)
print("language/task:", m.generation_config.language, m.generation_config.task)
print("WHISPER_FP16_DONE", sorted(p.name for p in dst.iterdir()))
