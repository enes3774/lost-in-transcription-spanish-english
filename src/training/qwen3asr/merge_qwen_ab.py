"""Merge the A+B Qwen3-ASR LoRA into the base model so stage_qwen3asr.py can load it directly (same as v4's build)."""
import glob, os, shutil, sys, time
import torch
from peft import PeftModel
from qwen_asr import Qwen3ASRModel
T = "/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track"
S = glob.glob(T + "/qwen3_asr_training/local_cache/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/*")[0]
A = sys.argv[1] if len(sys.argv) > 1 else T + "/kfold/qwen3asr_AB/best"
OUT = sys.argv[2] if len(sys.argv) > 2 else T + "/kfold/qwen3asr_AB/merged"
t = time.time()
qm = Qwen3ASRModel.from_pretrained(S, dtype=torch.bfloat16, device_map="cpu")
qm.model.thinker = PeftModel.from_pretrained(qm.model.thinker, A).merge_and_unload()
gc = qm.model.generation_config
for k in ("temperature", "top_p", "top_k"):
    setattr(gc, k, None)
gc.do_sample = False
qm.model.save_pretrained(OUT, safe_serialization=True, max_shard_size="5GB")
qm.processor.save_pretrained(OUT)
for f in os.listdir(S):
    if f.endswith(".json") and not os.path.exists(OUT + "/" + f):
        shutil.copy(S + "/" + f, OUT)
print("MERGED", OUT, f"{time.time() - t:.0f}s", flush=True)
