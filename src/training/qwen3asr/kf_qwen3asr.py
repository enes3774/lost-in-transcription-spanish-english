"""Qwen3-ASR-1.7B Miami LoRA on one speaker-disjoint group, early-stopped on T-val600 (greedy eval).
usage: kf_qwen3asr.py <A|B> [max_steps] [eval_every] [patience]"""
import glob
import json
import os
import random
import shutil
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_kf import AUG, RUNS, TAG, EarlyStop, P, augment, log, training_items, val_rows, wer  # noqa: E402

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from qwen_asr import Qwen3ASRModel  # noqa: E402

sys.path.insert(0, "/kaggle/working/omni_rt/sub")
import main as M  # noqa: E402  load_audio

G = sys.argv[1]
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
EVAL_EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 250
PATIENCE = int(sys.argv[4]) if len(sys.argv) > 4 else 2
ACCUM, LR = 8, 1e-4
OUT = RUNS / f"qwen3asr_{G}{TAG}"
OUT.mkdir(parents=True, exist_ok=True)
S = glob.glob("/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track/qwen3_asr_training/local_cache/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/*")[0]
TAGS = RUNS / "qwen3asr_lang_tags.json"
OLD_TAGS = "/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track/qwen3_asr_training/lora_runs/qwen3asr_r16_n6k/lang_tags.json"

qm = Qwen3ASRModel.from_pretrained(S, dtype=torch.bfloat16, device_map="cuda:0", max_inference_batch_size=16)
thinker, processor = qm.model.thinker, qm.processor
tok = processor.tokenizer
vdir, vrows, vrefs = val_rows()
import os as _os
import numpy as _np
AUG_RNG = _np.random.default_rng(1234)
items = training_items(G, 2.0, 1e9)
ids = [c["key"] for c in items if not c["extra"]]

# language tags from the base model (the target format is "language X<asr_text>text")
tags = {}
for path in (OLD_TAGS, TAGS):
    if os.path.exists(path):
        tags.update(json.load(open(path)))
todo = [i for i in ids if i not in tags]
log(f"group {G}: {len(ids)} clips, tagging {len(todo)} new")
with torch.no_grad():
    for k in range(0, len(todo), 64):
        part = todo[k:k + 64]
        for i, x in zip(part, qm.transcribe(audio=[str(P / "clips" / f"{i}.wav") for i in part])):
            tags[i] = getattr(x, "language", None) or "Spanish"
        if k % 1024 == 0:
            json.dump(tags, open(TAGS, "w"))
json.dump(tags, open(TAGS, "w"))
for c in items:
    if c["extra"]:
        tags[c["key"]] = "Spanish"


@torch.no_grad()
def evaluate(step):
    thinker.eval(); thinker.config.use_cache = True
    res = qm.transcribe(audio=[str(vdir / "clips" / r["audio_filename"]) for r in vrows])
    preds = {r["audio_filename"]: (getattr(x, "text", "") or "").strip() for r, x in zip(vrows, res)}
    thinker.config.use_cache = False; thinker.train()
    w = wer(vrefs, preds, [r["audio_filename"] for r in vrows])
    log(f"EVAL step {step}: T-val WER {w:.4f}")
    return w


stop = EarlyStop(OUT, PATIENCE)
stop.update(0, evaluate(0))
prompt = qm._build_text_prompt(context="", force_language=None)
for p in qm.model.parameters():
    p.requires_grad_(False)
pm = get_peft_model(thinker, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                         target_modules=r".*(language_model|text_model|model\.layers).*self_attn\.(q_proj|k_proj|v_proj|o_proj)"))
thinker.config.use_cache = False
lora = [p for n, p in pm.named_parameters() if "lora_" in n]
opt = torch.optim.AdamW(lora, lr=LR, weight_decay=0.0)
warm = 50
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.1, (MAX_STEPS - s) / max(1, MAX_STEPS - warm)))
pm.train()
step, micro, losses = 0, 0, []
while step < MAX_STEPS:
    for c in items:
        wav = augment(M.load_audio(c["wav"]), AUG_RNG)
        x = processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
        tgt = tok(f"language {tags[c['key']]}<asr_text>{c['text']}", add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        input_ids = torch.cat([x["input_ids"], torch.tensor([tgt])], dim=1)
        labels = input_ids.clone(); labels[:, : x["input_ids"].shape[1]] = -100
        b = {k: (v.to("cuda", torch.bfloat16) if v.dtype.is_floating_point else v.to("cuda")) for k, v in x.items() if k not in ("input_ids", "attention_mask")}
        out = pm(input_ids=input_ids.cuda(), attention_mask=torch.ones_like(input_ids).cuda(), labels=labels.cuda(), **b)
        (out.loss / ACCUM).backward(); losses.append(out.loss.item())
        micro += 1
        if micro % ACCUM:
            continue
        torch.nn.utils.clip_grad_norm_(lora, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 50 == 0:
            log(f"step {step} loss {sum(losses) / len(losses):.4f}"); losses = []
        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            improved, done = stop.update(step, evaluate(step))
            if improved:
                if (OUT / "best").exists():
                    shutil.rmtree(OUT / "best")
                pm.save_pretrained(str(OUT / "best"))
                log(f"new best at step {step}")
            if done:
                log(f"early stop at step {step}; best step {stop.best_step} WER {stop.best:.4f}")
                step = MAX_STEPS
        if step >= MAX_STEPS:
            break
if stop.best_step == 0:
    pm.save_pretrained(str(OUT / "best"))
log(f"DONE best step {stop.best_step} T-val WER {stop.best:.4f}")
