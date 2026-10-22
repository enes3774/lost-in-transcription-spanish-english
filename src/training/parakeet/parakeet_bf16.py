"""Save the trained parakeet checkpoint in bf16: the stage loads it as bf16 anyway, so fp32 on disk is
1.2 GB of wasted package space on a submission that is already close to the largest size the platform accepted."""
import shutil, sys, torch
from pathlib import Path
sys.path.insert(0, "/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track/vendor_tf")
from transformers import ParakeetForTDT
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
m = ParakeetForTDT.from_pretrained(str(src), dtype=torch.float32, local_files_only=True).to(torch.bfloat16)
dst.mkdir(parents=True, exist_ok=True)
m.save_pretrained(str(dst), safe_serialization=True)
for f in src.iterdir():
    if f.is_file() and f.name not in ("model.safetensors", "config.json"):
        shutil.copy(f, dst / f.name)
print("saved", sorted(p.name for p in dst.iterdir()))
