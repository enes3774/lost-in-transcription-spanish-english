"""Voxtral-Small-24B Miami LoRA on one speaker-disjoint group, early-stopped on T-val600.
usage: kf_voxtral.py <A|B> [max_steps] [eval_every] [patience]"""
import glob
import math
import os
import random
import shutil
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_kf import AUG, RUNS, TAG, EarlyStop, P, augment_to_file, log, training_items, val_rows, wer  # noqa: E402

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoProcessor, VoxtralForConditionalGeneration  # noqa: E402

G = sys.argv[1]
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
EVAL_EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 250
PATIENCE = int(sys.argv[4]) if len(sys.argv) > 4 else 2
ACCUM, LR = 8, 1e-4
OUT = RUNS / f"vox_{G}{TAG}"
OUT.mkdir(parents=True, exist_ok=True)
S = glob.glob("/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track/mistral_voxtral/local_cache/hub/models--mistralai--Voxtral-Small-24B-2507/snapshots/*")[0]
PROMPT = (
    "Transcribe this audio word for word.\n"
    "The speaker code-switches freely between English and Spanish, often inside a single "
    "sentence. Write every word in the language it was actually spoken in. "
    "Never translate between English and Spanish: if a phrase is spoken in English, "
    "keep it in English; if it is spoken in Spanish, keep it in Spanish.\n"
    "Do not summarise, do not rephrase, do not fix grammar, do not add words that were not said.\n"
    "Output only the transcript itself, with no preamble, no quotes and no commentary."
)

processor = AutoProcessor.from_pretrained(S)
tok = processor.tokenizer
conv = lambda p: [{"role": "user", "content": [{"type": "audio", "path": str(p)}, {"type": "text", "text": PROMPT}]}]
model = VoxtralForConditionalGeneration.from_pretrained(S, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
model.config.use_cache = False
for p in model.parameters():
    p.requires_grad_(False)
pm = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                       target_modules=r"language_model.*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"))
if hasattr(model, "language_model"):
    model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
vdir, vrows, vrefs = val_rows()
vrows = sorted(vrows, key=lambda r: -float(r["file_duration_seconds"]))


@torch.no_grad()
def evaluate(step):
    pm.eval(); model.config.use_cache = True
    preds = {}
    for i in range(0, len(vrows), 8):
        part = vrows[i:i + 8]
        x = processor.apply_chat_template([conv(vdir / "clips" / r["audio_filename"]) for r in part], return_tensors="pt")
        x = {k: v.to("cuda", torch.bfloat16 if v.dtype.is_floating_point else None) for k, v in x.items()}
        mx = max(32, int(max(float(r["file_duration_seconds"]) for r in part) * 20) + 64)
        out = pm.generate(**x, max_new_tokens=mx, do_sample=False)
        for r, t in zip(part, processor.batch_decode(out[:, x["input_ids"].shape[1]:], skip_special_tokens=True)):
            preds[r["audio_filename"]] = " ".join(t.split())
    model.config.use_cache = False; pm.train()
    w = wer(vrefs, preds, [r["audio_filename"] for r in vrows])
    log(f"EVAL step {step}: T-val WER {w:.4f}")
    return w


import os as _os
import numpy as _np
AUG_RNG = _np.random.default_rng(1234)
AUG_TMP = f"/dev/shm/kfaug_{_os.getpid()}.wav"
ids = training_items(G, 2.0, 1e9)
log(f"group {G}: {len(ids)} training clips (>=2 s), max_steps {MAX_STEPS}, eval every {EVAL_EVERY}, patience {PATIENCE}")
stop = EarlyStop(OUT, PATIENCE)
stop.update(0, evaluate(0))
lora = [p for n, p in pm.named_parameters() if "lora_" in n]
opt = torch.optim.AdamW(lora, lr=LR, weight_decay=0.0)
warm = 50
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.1, (MAX_STEPS - s) / max(1, MAX_STEPS - warm)))
pm.train()
step, micro, losses = 0, 0, []
while step < MAX_STEPS:
    for c in ids:
        try:
            x = processor.apply_chat_template(conv(augment_to_file(c["wav"], AUG_RNG, AUG_TMP)), return_tensors="pt")
            tgt = tok(c["text"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            input_ids = torch.cat([x["input_ids"], torch.tensor([tgt])], dim=1)
            labels = input_ids.clone(); labels[:, : x["input_ids"].shape[1]] = -100
            out = pm(input_ids=input_ids.cuda(), attention_mask=torch.ones_like(input_ids).cuda(),
                     input_features=x["input_features"].to("cuda", torch.bfloat16), labels=labels.cuda())
            (out.loss / ACCUM).backward(); losses.append(out.loss.item())
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); continue
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
    log("no improvement over the base model; saving untrained adapter as best")
    pm.save_pretrained(str(OUT / "best"))
log(f"DONE best step {stop.best_step} T-val WER {stop.best:.4f}")
