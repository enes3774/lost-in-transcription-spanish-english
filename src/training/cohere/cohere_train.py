"""Full fine-tune of Cohere Transcribe (2B conformer encoder-decoder) on Miami A+B, same protocol as the other
members: early stop on the selection half of T (miami_Tsel600), never on dev. Targets = the model's own decoder
prompt for language <lang> + the Miami text (manifest chosen by KF_MANIFEST, e.g. manifest_frag.jsonl) + EOS.
fp32 master weights, bf16 autocast, AdamW; BatchNorm statistics in the conformer are frozen (small batches).
usage: cohere_train.py <snapshot_dir> <out_dir> [--lang es --lr 2e-5 --bs 8 --accum 2 --max_steps 3000 --eval_every 250 --patience 3]
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/kaggle/working/omni_rt")
sys.path.insert(0, "/kaggle/working/omni_rt/kfold")
import jiwer  # noqa: E402
from score import normalize_text  # noqa: E402
from common_kf import training_items, val_rows  # noqa: E402
import re  # noqa: E402

fix = lambda t: re.sub(r"\bo_k\b", "okay", t or "")  # noqa: E731


def collapse(text, max_uni=4, max_ngram=2):
    """Runaway-repetition guard used on every member before the vote (same as vote.py / refine_common)."""
    w = (text or "").split()
    for n in (6, 5, 4, 3, 2, 1):
        cap = max_uni if n == 1 else max_ngram
        out, i = [], 0
        while i < len(w):
            seg = [x.lower().strip(".,?!") for x in w[i:i + n]]
            reps = 1
            while len(seg) == n and [x.lower().strip(".,?!") for x in w[i + reps * n:i + (reps + 1) * n]] == seg:
                reps += 1
            out.extend(w[i:i + min(reps, cap) * n])
            i += reps * n
        w = out
    return " ".join(w)

ap = argparse.ArgumentParser()
ap.add_argument("snapshot")
ap.add_argument("out")
ap.add_argument("--lang", default="es")
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--max_steps", type=int, default=3000)
ap.add_argument("--eval_every", type=int, default=250)
ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--group", default="AB", help="Miami training group: AB (final member) or A / B (fold model for out-of-fold hypotheses)")
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)


def log(m):
    print(time.strftime("%H:%M:%S ") + m, flush=True)


from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq  # noqa: E402

processor = AutoProcessor.from_pretrained(args.snapshot, trust_remote_code=True)
tok = processor.tokenizer
model = AutoModelForSpeechSeq2Seq.from_pretrained(args.snapshot, trust_remote_code=True, torch_dtype=torch.float32).to("cuda")
gen_cfg = json.load(open(Path(args.snapshot) / "generation_config.json"))
EOS = gen_cfg.get("eos_token_id", getattr(model.config, "eos_token_id", None))
PAD = gen_cfg.get("pad_token_id", getattr(model.config, "pad_token_id", None))
if PAD is None:
    PAD = EOS
PROMPT = model.build_prompt(language=args.lang, punctuation=True)
p_ids = tok(PROMPT, add_special_tokens=False)["input_ids"]
log(f"model {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B fp32; eos {EOS} pad {PAD}; prompt {len(p_ids)} tokens: {PROMPT}")

items = [it for it in training_items(args.group, 1.0, 35.0, seed=args.seed) if it["text"].strip()]
vdir, vrows, vrefs = val_rows()
vnames = [r["audio_filename"] for r in vrows]
log(f"{len(items)} training clips; validation {vdir.name} {len(vnames)} clips; lang {args.lang}")


def load_wav(p):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == 16000, (p, sr)
    return x


def build(batch):
    wavs = [load_wav(it["wav"]) for it in batch]
    feats = processor.feature_extractor(wavs, sampling_rate=16000, return_tensors="pt")
    seqs = []
    for it in batch:
        t = tok(it["text"].strip(), add_special_tokens=False)["input_ids"]
        seqs.append(p_ids + t + [EOS])
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L - 1), PAD, dtype=torch.long)
    lab = torch.full((len(seqs), L - 1), -100, dtype=torch.long)
    am = torch.zeros((len(seqs), L - 1), dtype=torch.long)
    for i, s in enumerate(seqs):
        n = len(s) - 1
        ids[i, :n] = torch.tensor(s[:-1])
        lab[i, :n] = torch.tensor(s[1:])
        lab[i, :len(p_ids) - 1] = -100          # predict from the last prompt token onwards
        am[i, :n] = 1
    out = {k: v.to("cuda") for k, v in feats.items() if torch.is_tensor(v)}
    out["input_ids"] = ids.to("cuda")
    out["decoder_attention_mask"] = am.to("cuda")
    out["labels"] = lab.to("cuda")
    return out


@torch.no_grad()
def evaluate(step):
    model.eval()
    wavs = [load_wav(str(vdir / "clips" / n)) for n in vnames]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        texts = model.transcribe(processor, args.lang, audio_arrays=wavs, sample_rates=[16000] * len(wavs), batch_size=32)
    preds = {n: collapse(" ".join(str(t).split())) for n, t in zip(vnames, texts)}
    n_loop = sum(1 for n, t in zip(vnames, texts) if len(str(t).split()) > 2 * len(preds[n].split()) + 3)
    with open(OUT / f"val_step{step}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio_filename", "transcript"])
        for n in vnames:
            w.writerow([n, preds[n]])
    names = [n for n in vnames if normalize_text(fix(vrefs[n])).strip()]
    w_ = jiwer.wer([normalize_text(fix(vrefs[n])) for n in names], [normalize_text(fix(preds[n])) for n in names])
    log(f"   eval step {step}: {n_loop} looping clips collapsed")
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
    return w_


model.train()
for m in model.modules():
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.eval()
w0 = evaluate(0)
log(f"step 0 (zero-shot) T-val WER {w0:.4f}")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
rng = random.Random(args.seed)
best, best_step, bad = w0, 0, 0
step, micro, losses, t1 = 0, 0, [], time.time()
pos = 0
while step < args.max_steps:
    if pos + args.bs > len(items):
        rng.shuffle(items)
        pos = 0
    batch = items[pos:pos + args.bs]
    pos += args.bs
    try:
        b = build(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**b)
        (out.loss / args.accum).backward()
        losses.append(out.loss.item())
    except torch.OutOfMemoryError:
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        log("OOM, batch skipped")
        continue
    micro += 1
    if micro % args.accum:
        continue
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    step += 1
    if step % 25 == 0:
        log(f"step {step}/{args.max_steps} loss {sum(losses) / len(losses):.4f} {(time.time() - t1) / micro:.2f}s/batch mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G")
        losses = []
    if step % args.eval_every == 0:
        w_ = evaluate(step)
        log(f"step {step} T-val WER {w_:.4f} (best {best:.4f} @ {best_step})")
        if w_ < best:
            best, best_step, bad = w_, step, 0
            model.save_pretrained(str(OUT / "best"), safe_serialization=True)
            # the processor cannot serialize its filterbank module; its files are unchanged, copy them
            for f in Path(args.snapshot).iterdir():
                if f.is_file() and f.name not in ("model.safetensors", "config.json", "README.md") and not f.name.startswith("."):
                    (OUT / "best" / f.name).write_bytes(f.read_bytes())
            log(f"new best at step {step}")
        else:
            bad += 1
            if bad >= args.patience:
                log(f"early stop at step {step}; best step {best_step} WER {best:.4f}")
                break
log(f"DONE best step {best_step} T-val WER {best:.4f}")
