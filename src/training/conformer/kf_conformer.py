"""NeMo enes Conformer-CTC fine-tune on one speaker-disjoint group, early-stopped on T-val600.
Plain PyTorch loop (NeMo's loaders crash on this box); run with CUDA_LAUNCH_BLOCKING=1.
Saves the best weights as a plain state dict + tokenizer for stage_conformer.py.
usage: kf_conformer.py <A|B> [max_steps] [eval_every] [patience]"""
import os
import random
import re
import shutil
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_kf import AUG, RUNS, TAG, EarlyStop, P, augment, log, training_items, val_rows, wer  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

G = sys.argv[1]
MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
EVAL_EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 500
PATIENCE = int(sys.argv[4]) if len(sys.argv) > 4 else 3
BS, LR = 16, 5e-5
OUT = RUNS / f"conformer_{G}{TAG}"
(OUT / "serve").mkdir(parents=True, exist_ok=True)
NEMO = "/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track/ctc_conformer_training/nemo/stt_enes_conformer_ctc_large_codesw.nemo"

from nemo.collections.asr.models import EncDecCTCModelBPE  # noqa: E402

model = EncDecCTCModelBPE.restore_from(NEMO, map_location="cpu").to("cuda")
with tarfile.open(NEMO) as t:
    tokname = next(n for n in t.getnames() if n.endswith("tokenizer.model"))
    with open(OUT / "serve" / "tokenizer.model", "wb") as f:
        f.write(t.extractfile(tokname).read())
vdir, vrows, vrefs = val_rows()


def norm_target(t):
    t = t.lower().replace("’", "'")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", t)).strip()


def batchify(wavs):
    L = torch.tensor([len(w) for w in wavs]); X = torch.zeros(len(wavs), int(L.max()))
    for k, w in enumerate(wavs):
        X[k, : len(w)] = torch.from_numpy(w)
    return X.cuda(), L.cuda()


@torch.no_grad()
def evaluate(step):
    model.eval()
    preds = {}
    rows = sorted(vrows, key=lambda r: -float(r["file_duration_seconds"]))
    for k in range(0, len(rows), 8):
        part = rows[k:k + 8]
        x, l = batchify([sf.read(vdir / "clips" / r["audio_filename"], dtype="float32")[0] for r in part])
        lp, el, _ = model.forward(input_signal=x, input_signal_length=l)
        hyp = model.decoding.ctc_decoder_predictions_tensor(lp, decoder_lengths=el, return_hypotheses=False)
        hyp = hyp[0] if isinstance(hyp, tuple) else hyp
        for r, h in zip(part, hyp):
            preds[r["audio_filename"]] = getattr(h, "text", h).strip()
    model.train()
    w = wer(vrefs, preds, [r["audio_filename"] for r in vrows])
    log(f"EVAL step {step}: T-val WER {w:.4f}")
    return w


def save_best():
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.endswith("num_batches_tracked")}
    torch.save(sd, OUT / "serve" / "conformer_miami.pt")


import numpy as _np
AUG_RNG = _np.random.default_rng(1234)
clips = [dict(c, tgt=norm_target(c["text"])) for c in training_items(G, 1.0, 20.0)]
clips = [c for c in clips if c["tgt"]]
log(f"group {G}: {len(clips)} training clips, max_steps {MAX_STEPS}, eval every {EVAL_EVERY}, patience {PATIENCE}")
stop = EarlyStop(OUT, PATIENCE)
stop.update(0, evaluate(0))
save_best()
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=1e-3)
warm = 200
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.05, 0.5 * (1 + np.cos(np.pi * min(1.0, (s - warm) / max(1, MAX_STEPS - warm))))))
model.train()
step, pos, losses = 0, 0, []
while step < MAX_STEPS:
    part = clips[pos:pos + BS]
    pos = (pos + BS) % len(clips)
    if len(part) < BS:
        continue
    x, l = batchify([augment(sf.read(c["wav"], dtype="float32")[0], AUG_RNG) for c in part])
    tg = [model.tokenizer.text_to_ids(c["tgt"]) for c in part]
    tl = torch.tensor([len(t) for t in tg]).cuda()
    tt = torch.zeros(len(tg), int(tl.max()), dtype=torch.long)
    for k, t in enumerate(tg):
        tt[k, : len(t)] = torch.tensor(t)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lp, el, _ = model.forward(input_signal=x, input_signal_length=l)
    loss = model.loss(log_probs=lp.float(), targets=tt.cuda(), input_lengths=el, target_lengths=tl)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    step += 1; losses.append(loss.item())
    if step % 100 == 0:
        log(f"step {step} loss {np.mean(losses):.3f}"); losses = []
    if step % EVAL_EVERY == 0 or step == MAX_STEPS:
        improved, done = stop.update(step, evaluate(step))
        if improved:
            save_best(); log(f"new best at step {step}")
        if done:
            log(f"early stop at step {step}; best step {stop.best_step} WER {stop.best:.4f}")
            break
log(f"DONE best step {stop.best_step} T-val WER {stop.best:.4f}")
