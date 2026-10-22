"""LoRA fine-tune of the Qwen3-Omni thinker's text attention on en-es dev clips (HF + peft).

usage: python omni_lora_train.py --folds 0,1,2,3,4 --run r16_e3 [--full] [--epochs 3 --lr 1e-4 --r 16]
  --folds k  : train on all dev clips with fold != k, save <run>/fold{k}.pt
  --full     : train on all 155 clips, save <run>/full.pt  (for the final submission)
The model is loaded once and the LoRA weights are re-initialised per fold.
Saved file: {"A": {hf_weight_name: A}, "B": {...}, "scale": alpha/r} (tiny, on the HDD).
"""
import argparse
import csv
import math
import os
import random
import sys
import json
import time
import types
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--folds", default="0,1,2,3,4")
ap.add_argument("--full", action="store_true")
ap.add_argument("--run", required=True)
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--prompt_job", default=None, help="lab job json whose system/prompt to train with")
ap.add_argument("--grad_ckpt", action="store_true")
ap.add_argument("--folds_csv", default="/kaggle/working/omni_rt/dev/folds.csv")
ap.add_argument("--miami", type=int, default=0, help="train on N Bangor Miami clips instead of dev folds")
ap.add_argument("--min_s", type=float, default=2.0)
ap.add_argument("--miami_names", default=None, help="file of <id>.wav names; keep only these Miami clips (A+B halves)")
ap.add_argument("--save_every", type=int, default=250)
args = ap.parse_args()

sys.path.insert(0, "/kaggle/working/omni_rt/sub")
import main as M  # noqa: E402  prompt, audio loader

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration  # noqa: E402

MODEL = "/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track/qwen3_omni/submission_vllm/model"
RUNS = Path("/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track/qwen3_omni/lora_runs")
DEV = Path("/kaggle/working/omni_rt/dev")
out_dir = RUNS / args.run
out_dir.mkdir(parents=True, exist_ok=True)


def log(m):
    print(time.strftime("%H:%M:%S ") + m, flush=True)


if args.miami:
    MIAMI = Path("/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/miami_prep")
    import re as _re
    clips = [json.loads(l) for l in open(MIAMI / "manifest.jsonl", encoding="utf-8")]
    clips = [c for c in clips if c["duration"] >= args.min_s and not _re.search(r"(xxx|xx|yyy|www)", c["text"])
             and (MIAMI / "clips" / f"{c['id']}.wav").exists()]
    if args.miami_names:
        keep = set(l.strip() for l in open(args.miami_names, encoding='utf-8') if l.strip())
        clips = [c for c in clips if c['id'] + '.wav' in keep]
        print(f'miami_names filter: {len(keep)} names -> {len(clips)} clips', flush=True)
    random.Random(args.seed).shuffle(clips)
    clips = clips[: args.miami]
    refs = {c["id"] + ".wav": c["text"] for c in clips}
    folds = {n: -1 for n in refs}
    wavs = {n: M.load_audio(MIAMI / "clips" / n) for n in refs}
    print(f"miami clips {len(clips)}, {sum(c['duration'] for c in clips) / 3600:.1f} h", flush=True)
else:
    refs = {r["audio_filename"]: r["transcript"] for r in csv.DictReader(open(DEV / "refs.csv", encoding="utf-8"))}
    folds = {r["audio_filename"]: int(r["fold"]) for r in csv.DictReader(open(args.folds_csv))}
    wavs = {n: M.load_audio(DEV / "clips" / n) for n in refs}

processor = AutoProcessor.from_pretrained(MODEL)
job = json.load(open(args.prompt_job)) if args.prompt_job else {}
SYSTEM = job.get("system", M.SYSTEM_PROMPT)
USER_PROMPT = job.get("prompt", M.PROMPT)
conv = ([{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}] if SYSTEM else []) +        [{"role": "user", "content": [{"type": "audio", "audio": "x"}, {"type": "text", "text": USER_PROMPT}]}]
PROMPT_TEXT = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)


def fast_moe_forward(self, hidden_states):
    """Same math as Qwen3OmniMoeThinkerTextSparseMoeBlock.forward, but all experts in two
    grouped matmuls instead of a Python loop over up to 128 experts."""
    b, t, h = hidden_states.shape
    x = hidden_states.reshape(-1, h)
    router_logits = self.gate(x)
    rw = F.softmax(router_logits, dim=1, dtype=torch.float)
    rw, sel = torch.topk(rw, self.top_k, dim=-1)
    if self.norm_topk_prob:
        rw = rw / rw.sum(dim=-1, keepdim=True)
    rw = rw.to(x.dtype)
    flat = sel.reshape(-1)
    order = torch.argsort(flat, stable=True)
    tok = torch.div(order, self.top_k, rounding_mode="floor")
    offs = torch.cumsum(torch.bincount(flat, minlength=self.num_experts), 0, dtype=torch.int32)
    xs = x.index_select(0, tok)
    g, u = torch._grouped_mm(xs, self.w_gu, offs=offs).split(self.inter, dim=-1)
    y = torch._grouped_mm(self.moe_act(g) * u, self.w_dn, offs=offs)
    y = y * rw.reshape(-1).index_select(0, order).unsqueeze(-1)
    out = torch.zeros_like(x).index_add_(0, tok, y)
    return out.reshape(b, t, h), router_logits


def fuse_moe_block(block):
    ex = block.experts
    inter = ex[0].gate_proj.weight.shape[0]
    block.register_buffer("w_gu", torch.stack(
        [torch.cat([e.gate_proj.weight.t(), e.up_proj.weight.t()], dim=1) for e in ex]).contiguous(), persistent=False)
    block.register_buffer("w_dn", torch.stack([e.down_proj.weight.t() for e in ex]).contiguous(), persistent=False)
    block.inter = inter
    block.moe_act = ex[0].act_fn
    return ex

t0 = time.time()
# load on CPU, fuse each MoE block's experts into stacked tensors there, then move to GPU
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="cpu", attn_implementation="sdpa", enable_audio_output=False)
thinker = model.thinker
if getattr(model, "talker", None) is not None:
    del model.talker
thinker.config.use_cache = False
for p in thinker.parameters():
    p.requires_grad_(False)
log(f"model loaded to CPU in {time.time() - t0:.0f}s")
check_block, check_experts = None, None
for i, layer in enumerate(thinker.model.layers):
    blk = layer.mlp
    old = fuse_moe_block(blk)
    if i == 0:
        check_block, check_experts = blk, old
    del blk.experts
    blk.forward = types.MethodType(fast_moe_forward, blk)
thinker.to("cuda")
log(f"fused MoE and moved to GPU at {time.time() - t0:.0f}s, gpu mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
with torch.no_grad():  # numerical check of the fused block against the original loop
    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as mq
    check_experts.to("cuda")
    ref_blk = types.SimpleNamespace(**{k: getattr(check_block, k) for k in ("num_experts", "top_k", "norm_topk_prob", "gate")})
    ref_blk.experts = check_experts
    xin = torch.randn(1, 257, thinker.config.text_config.hidden_size, device="cuda", dtype=torch.bfloat16)
    ref_out, _ = mq.Qwen3OmniMoeThinkerTextSparseMoeBlock.forward(ref_blk, xin)
    new_out, _ = check_block(xin)
    err = ((ref_out.float() - new_out.float()).abs().max() / ref_out.float().abs().max()).item()
    log(f"fused MoE check: max rel err {err:.2e}")
    assert err < 1e-2, "fused MoE mismatch"
    del check_experts, ref_blk
    torch.cuda.empty_cache()

lcfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
                  target_modules=r"model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)")
pm = get_peft_model(thinker, lcfg)
pm.print_trainable_parameters()
if args.grad_ckpt:
    thinker.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
thinker.model.train()


def build(name):
    wav = wavs[name]
    p = processor(text=[PROMPT_TEXT], audio=[wav], return_tensors="pt", padding=True)
    full = processor(text=[PROMPT_TEXT + refs[name] + "<|im_end|>"], audio=[wav], return_tensors="pt", padding=True)
    P = p["input_ids"].shape[1]
    assert torch.equal(full["input_ids"][0, :P], p["input_ids"][0]), "prompt tokens are not a prefix"
    labels = full["input_ids"].clone()
    labels[:, :P] = -100
    batch = {k: v.to("cuda") for k, v in full.items()}
    batch["input_features"] = batch["input_features"].to(torch.bfloat16)
    batch["labels"] = labels.to("cuda")
    return batch


lora_params = [p for n, p in pm.named_parameters() if "lora_" in n]


def reset_lora(seed):
    torch.manual_seed(seed)
    for n, mod in pm.named_modules():
        if hasattr(mod, "reset_lora_parameters") and hasattr(mod, "lora_A") and "default" in mod.lora_A:
            mod.reset_lora_parameters("default", True)


def save(path):
    A, B = {}, {}
    for n, mod in pm.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            key = n.replace("base_model.model.", "", 1)
            hf = "thinker." + key + ".weight"
            A[hf] = mod.lora_A["default"].weight.detach().to("cpu", torch.float32).clone()
            B[hf] = mod.lora_B["default"].weight.detach().to("cpu", torch.float32).clone()
    torch.save({"A": A, "B": B, "scale": args.alpha / args.r, "args": vars(args)}, path)


def train(train_names, tag):
    reset_lora(args.seed)
    opt = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_names) / args.accum)
    total = max(1, int(steps_per_epoch * args.epochs))
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    rng = random.Random(args.seed)
    step, micro, t_start, run_loss = 0, 0, time.time(), []
    while step < total:
        order = train_names[:]
        rng.shuffle(order)
        for name in order:
            batch = build(name)
            out = thinker(**batch)
            (out.loss / args.accum).backward()
            run_loss.append(out.loss.item())
            micro += 1
            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0 or step == total:
                    log(f"[{tag}] step {step}/{total} loss {sum(run_loss) / len(run_loss):.4f} "
                        f"{(time.time() - t_start) / micro:.2f}s/sample mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G")
                    run_loss = []
                if args.miami and step % args.save_every == 0 and step < total:
                    save(out_dir / f"{tag}_step{step}.pt")
                    log(f"[{tag}] saved step {step}")
                if step >= total:
                    break
    path = out_dir / f"{tag}.pt"
    save(path)
    log(f"[{tag}] saved {path} after {(time.time() - t_start) / 60:.1f} min")


names = sorted(refs)
if args.miami:
    train(names, "miami")
elif args.full:
    train(names, "full")
else:
    for k in [int(x) for x in args.folds.split(",")]:
        if (out_dir / f"fold{k}.pt").exists():
            log(f"fold{k} exists, skipping")
            continue
        train([n for n in names if folds[n] != k], f"fold{k}")
log("all done")
