"""Bangor Miami fine-tune of moondream/parakeet-ultra (ParakeetForTDT), same protocol as cohere_train.py:
fp32 master weights with bf16 autocast, BatchNorm running stats frozen, early stop on the SELECTION half of
the Miami held-out set (miami_Tsel600) with the runaway-repetition guard applied - never on the competition dev set.

The Ultra repo ships only weights + config + tokenizer, so the processor and the generation config come from
the base repo (nvidia/parakeet-tdt-0.6b-v3); the two config.json files are byte-identical.

  parakeet_train.py <out_dir> [--init moondream/parakeet-ultra] [--lr 1e-5] [--group AB]
"""
import argparse
import csv
import random
import re
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, "/kaggle/working/omni_rt/kfold")
import jiwer  # noqa: E402
from score import normalize_text  # noqa: E402
from common_kf import training_items, val_rows  # noqa: E402


def fix(t):
    return re.sub(r"\bo_k\b", "okay", t or "")


def collapse(text, max_uni=4, max_ngram=2):
    """Cap runaway repetition so a looping clip cannot make the selection metric meaningless."""
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
ap.add_argument("out")
ap.add_argument("--init", default="moondream/parakeet-ultra")
ap.add_argument("--proc", default="nvidia/parakeet-tdt-0.6b-v3")
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--max_steps", type=int, default=3000)
ap.add_argument("--eval_every", type=int, default=250)
ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_s", type=float, default=30.0)
ap.add_argument("--group", default="AB", help="Miami group: AB (final member) or A / B (fold model for OOF hypotheses)")
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)


def log(m):
    print(time.strftime("%H:%M:%S ") + m, flush=True)


from transformers import AutoProcessor, GenerationConfig, ParakeetForTDT  # noqa: E402

processor = AutoProcessor.from_pretrained(args.proc)
model = ParakeetForTDT.from_pretrained(args.init, dtype=torch.float32).to("cuda")
model.generation_config = GenerationConfig.from_pretrained(args.proc)
log("model %.2fB fp32 from %s; processor %s" % (sum(p.numel() for p in model.parameters()) / 1e9, args.init, args.proc))
log("blank %s pad %s vocab %s" % (model.config.blank_token_id, model.config.pad_token_id, model.config.vocab_size))

items = [it for it in training_items(args.group, 1.0, args.max_s, seed=args.seed) if it["text"].strip()]
vdir, vrows, vrefs = val_rows()
vnames = [r["audio_filename"] for r in vrows]
log("%d training clips; validation %s %d clips" % (len(items), vdir.name, len(vnames)))


def load_wav(p):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == 16000, (p, sr)
    return x


def to_cuda(enc):
    """Keep features in fp32: the master weights are fp32 and autocast does the bf16 conversion itself."""
    return {k: v.to("cuda") for k, v in enc.items() if torch.is_tensor(v)}


def build(batch):
    """The processor makes input_features + attention_mask + labels + decoder_input_ids (blank-prefixed) for us."""
    wavs = [load_wav(it["wav"]) for it in batch]
    texts = [it["text"].strip() for it in batch]
    return to_cuda(processor(wavs, text=texts, sampling_rate=16000, return_tensors="pt", padding=True))


def freeze_bn():
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


@torch.no_grad()
def evaluate(step, bs=16):
    model.eval()
    preds, raw = {}, {}
    for i in range(0, len(vnames), bs):
        chunk = vnames[i:i + bs]
        wavs = [load_wav(str(vdir / "clips" / n)) for n in chunk]
        enc = to_cuda(processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = model.generate(**enc)
        seq = gen.sequences if hasattr(gen, "sequences") else gen
        for n, t in zip(chunk, processor.batch_decode(seq, skip_special_tokens=True)):
            raw[n] = " ".join(str(t).split())
            preds[n] = collapse(raw[n])
    n_loop = sum(1 for n in vnames if len(raw[n].split()) > 2 * len(preds[n].split()) + 3)
    with open(OUT / ("val_step%d.csv" % step), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio_filename", "transcript"])
        for n in vnames:
            w.writerow([n, preds[n]])
    names = [n for n in vnames if normalize_text(fix(vrefs[n])).strip()]
    score = jiwer.wer([normalize_text(fix(vrefs[n])) for n in names], [normalize_text(fix(preds[n])) for n in names])
    log("   eval step %d: %d looping clips collapsed" % (step, n_loop))
    model.train()
    freeze_bn()
    return score


def save_best(step):
    d = OUT / "best"
    model.save_pretrained(str(d), safe_serialization=True)
    model.generation_config.save_pretrained(str(d))
    try:                                   # keep the saved folder self-contained for the package stage
        processor.save_pretrained(str(d))
    except Exception as e:
        log("   processor.save_pretrained failed (%s); the package stage must take it from %s" % (type(e).__name__, args.proc))
    (d / "SOURCE.txt").write_text("%s + Miami %s, lr %s, step %d\n" % (args.init, args.group, args.lr, step))


model.train()
freeze_bn()
w0 = evaluate(0)
log("step 0 (zero-shot) T-val WER %.4f" % w0)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
rng = random.Random(args.seed)


def make_batches():
    """Length-bucketed batches. The TDT loss allocates batch x frames x tokens x vocab, so a batch padded to the
    longest clip in the set is what blows up memory; grouping similar durations keeps every lattice small."""
    ordered = sorted(items, key=lambda it: it["duration"])
    groups = [ordered[i:i + args.bs] for i in range(0, len(ordered), args.bs)]
    rng.shuffle(groups)
    return groups


batches = make_batches()
log("%d length-bucketed batches of %d (max clip %.0fs)" % (len(batches), args.bs, max(it["duration"] for it in items)))
best, best_step, bad = w0, 0, 0
step, micro, losses, t1, pos = 0, 0, [], time.time(), 0
while step < args.max_steps:
    if pos >= len(batches):
        batches = make_batches()
        pos = 0
    batch = batches[pos]
    pos += 1
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
        log("step %d/%d loss %.4f %.2fs/batch mem %.1fG" % (
            step, args.max_steps, sum(losses) / max(1, len(losses)),
            (time.time() - t1) / micro, torch.cuda.max_memory_allocated() / 2 ** 30))
        losses = []
    if step % args.eval_every == 0:
        score = evaluate(step)
        log("step %d T-val WER %.4f (best %.4f @ %d)" % (step, score, best, best_step))
        if score < best:
            best, best_step, bad = score, step, 0
            save_best(step)
            log("new best at step %d" % step)
        else:
            bad += 1
            if bad >= args.patience:
                log("early stop at step %d; best step %d WER %.4f" % (step, best_step, best))
                break
log("PARAKEET_TRAIN_DONE best step %d T-val WER %.4f" % (best_step, best))
