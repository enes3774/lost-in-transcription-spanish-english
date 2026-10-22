"""Miami A+B fine-tune of Phi-4-multimodal-instruct (ZFTurbo fork) for ASR: trains the speech LoRA (r320), the audio
encoder and the speech audio projection; the 3.8B language model and everything vision stay frozen (bf16).
Trainable parts are fp32 masters under bf16 autocast. Same protocol as the other members: frag labels, early stop on
miami_Tsel600 (never dev). Saves only the trained tensors (bf16) as <out>/best/delta.safetensors.
usage: phi4_train.py <snapshot> <out> [--lr 4e-5 --bs 8 --accum 2 --max_steps 3000 --eval_every 250 --patience 3]"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from safetensors.torch import save_file

sys.path.insert(0, "/kaggle/working/omni_rt")
sys.path.insert(0, "/kaggle/working/omni_rt/kfold")
import re  # noqa: E402
import jiwer  # noqa: E402
from score import normalize_text  # noqa: E402
from common_kf import training_items, val_rows  # noqa: E402

fix = lambda t: re.sub(r"\bo_k\b", "okay", t or "")  # noqa: E731


def collapse(text, max_uni=4, max_ngram=2):
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
ap.add_argument("--lr", type=float, default=4e-5)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--max_steps", type=int, default=3000)
ap.add_argument("--eval_every", type=int, default=250)
ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--group", default="AB")
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(args.seed)
SR = 16000


def log(m):
    print(time.strftime("%H:%M:%S ") + m, flush=True)


from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: E402

processor = AutoProcessor.from_pretrained(args.snapshot, trust_remote_code=True)
tok = processor.tokenizer
model = AutoModelForCausalLM.from_pretrained(args.snapshot, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                             _attn_implementation="sdpa").to("cuda")
PROMPT = "<|user|><|audio_1|>Transcribe the audio clip into text.<|end|><|assistant|>"
SUFFIX = "<|end|><|endoftext|>"
ASSIST = tok.convert_tokens_to_ids("<|assistant|>")
END = tok.convert_tokens_to_ids("<|end|>")
TRAIN_KEYS = ("lora_A.speech", "lora_B.speech", "audio_embed.encoder", "audio_projection.speech")
train_names = []
for n, p in model.named_parameters():
    if any(k in n for k in TRAIN_KEYS):
        p.data = p.data.float()
        p.requires_grad_(True)
        train_names.append(n)
    else:
        p.requires_grad_(False)
tset = set(train_names)
params = [p for n, p in model.named_parameters() if n in tset]
assert any("lora_A.speech" in n for n in train_names) and any("audio_embed.encoder" in n for n in train_names) \
    and any("audio_projection.speech" in n for n in train_names), "trainable groups missing"
log(f"trainable {sum(p.numel() for p in params) / 1e6:.0f}M in {len(params)} tensors; assistant id {ASSIST}, end id {END}")

items = [it for it in training_items(args.group, 1.0, 35.0, seed=args.seed) if it["text"].strip()]
vdir, vrows, vrefs = val_rows()
vnames = [r["audio_filename"] for r in vrows]
vdur = {r["audio_filename"]: float(r.get("file_duration_seconds") or 0) for r in vrows}
log(f"{len(items)} training clips; validation {vdir.name} {len(vnames)} clips")


def load_wav(p):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    assert sr == SR, (p, sr)
    return x


IMAGE_KEYS = ("input_image_embeds", "image_sizes", "image_attention_mask")


def to_cuda(inp):  # the processor sends empty image tensors; in train mode they trigger a fake vision forward
    return {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in inp.items() if k not in IMAGE_KEYS}


def build(batch):
    texts = [PROMPT + it["text"].strip() + SUFFIX for it in batch]
    inp = to_cuda(processor(text=texts, audios=[(load_wav(it["wav"]), SR) for it in batch], return_tensors="pt"))
    ids, am = inp["input_ids"], inp["attention_mask"].long()
    lab = ids.clone()
    lab[am == 0] = -100
    for i in range(ids.shape[0]):
        a = (ids[i] == ASSIST).nonzero()
        assert len(a) >= 1
        lab[i, :a[-1].item() + 1] = -100            # loss only on the answer (+ <|end|><|endoftext|>)
    inp["labels"] = lab
    inp["position_ids"] = (am.cumsum(1) - 1).clamp(min=0)
    return inp


@torch.no_grad()
def evaluate(step):
    model.eval()
    preds, n_loop = {}, 0
    order = sorted(vnames, key=lambda n: -vdur[n])
    for i in range(0, len(order), 16):
        ch = order[i:i + 16]
        inp = to_cuda(processor(text=[PROMPT] * len(ch), audios=[(load_wav(str(vdir / "clips" / n)), SR) for n in ch], return_tensors="pt"))
        mx = max(64, int(max(vdur[n] for n in ch) * 12) + 64)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = model.generate(**inp, max_new_tokens=mx, do_sample=False, num_logits_to_keep=1, eos_token_id=[END, tok.eos_token_id])
        outs = tok.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for n, t in zip(ch, outs):
            raw = " ".join(str(t).split())
            preds[n] = collapse(raw)
            n_loop += len(raw.split()) > 2 * len(preds[n].split()) + 3
    with open(OUT / f"val_step{step}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio_filename", "transcript"])
        for n in vnames:
            w.writerow([n, preds[n]])
    names = [n for n in vnames if normalize_text(fix(vrefs[n])).strip()]
    w_ = jiwer.wer([normalize_text(fix(vrefs[n])) for n in names], [normalize_text(fix(preds[n])) for n in names])
    log(f"   eval step {step}: {n_loop} looping clips collapsed")
    model.train()
    return w_


w0 = evaluate(0)
log(f"step 0 (zero-shot) T-val WER {w0:.4f}")
model.train()
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
rng = random.Random(args.seed)
best, best_step, bad = w0, 0, 0
step, micro, losses, t1, pos = 0, 0, [], time.time(), 0
while step < args.max_steps:
    if pos + args.bs > len(items):
        rng.shuffle(items)
        pos = 0
    batch = items[pos:pos + args.bs]
    pos += args.bs
    b = build(batch)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**b)
    assert torch.isfinite(out.loss), f"non-finite loss at step {step}"
    (out.loss / args.accum).backward()
    losses.append(out.loss.item())
    micro += 1
    if micro % args.accum:
        continue
    torch.nn.utils.clip_grad_norm_(params, 1.0)
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
            (OUT / "best").mkdir(exist_ok=True)
            sd = dict(model.named_parameters())
            save_file({n: sd[n].detach().to(torch.bfloat16).contiguous() for n in train_names}, str(OUT / "best" / "delta.safetensors"))
            log(f"new best at step {step}")
        else:
            bad += 1
            if bad >= args.patience:
                log(f"early stop at step {step}; best step {best_step} WER {best:.4f}")
                break
log(f"DONE best step {best_step} T-val WER {best:.4f}")
