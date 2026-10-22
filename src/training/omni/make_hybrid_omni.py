"""Build the hybrid Omni checkpoint for v6: the AWQ int4 experts as shipped in v5, but the thinker's attention
projections (q/k/v/o) restored to bf16 from the original model so the refiner's attention delta can be merged in
exactly. Talker and code2wav tensors (6.6 GiB, unused by the thinker-only vLLM path) are dropped.
usage: make_hybrid_omni.py <awq_dir> <bf16_dir> <out_dir>
"""
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC, BF, DST = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
DST.mkdir(parents=True, exist_ok=True)
PACKED = re.compile(r"^thinker\.model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.(weight_packed|weight_scale|weight_shape)$")
BF16 = re.compile(r"^thinker\.model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight$")
SHARD_BYTES = 4 * 2**30


def log(m):
    print(time.strftime("%H:%M:%S ") + m, flush=True)


idx = json.load(open(SRC / "model.safetensors.index.json"))["weight_map"]
bidx = json.load(open(BF / "model.safetensors.index.json"))["weight_map"]
shard, size, n_shard, new_map, total = {}, 0, 0, {}, 0
dropped = {"talker": 0, "code2wav": 0, "attn_packed": 0}


def flush():
    global shard, size, n_shard
    if not shard:
        return
    n_shard += 1
    name = f"model-{n_shard:05d}.safetensors"
    save_file(shard, str(DST / name), metadata={"format": "pt"})
    for k in shard:
        new_map[k] = name
    log(f"wrote {name} ({size / 2**30:.2f} GiB, {len(shard)} tensors)")
    shard, size = {}, 0


def add(k, t):
    global size, total
    shard[k] = t.contiguous()
    size += t.numel() * t.element_size()
    total += t.numel() * t.element_size()
    if size >= SHARD_BYTES:
        flush()


for f in sorted(set(idx.values())):
    with safe_open(str(SRC / f), framework="pt") as fh:
        for k in fh.keys():
            if k.startswith("talker."):
                dropped["talker"] += 1
                continue
            if k.startswith("code2wav."):
                dropped["code2wav"] += 1
                continue
            if PACKED.match(k):
                dropped["attn_packed"] += 1
                continue
            add(k, fh.get_tensor(k))
    log(f"read {f}")
layers = set()
by_file = {}
for k, f in bidx.items():
    if BF16.match(k):
        by_file.setdefault(f, []).append(k)
        layers.add(int(BF16.match(k).group(1)))
n_attn = 0
for f, ks in sorted(by_file.items()):
    with safe_open(str(BF / f), framework="pt") as fh:
        for k in ks:
            t = fh.get_tensor(k)
            assert t.dtype == torch.bfloat16, (k, t.dtype)
            add(k, t)
            n_attn += 1
flush()
log(f"dropped {dropped}; added {n_attn} bf16 attention tensors for {len(layers)} layers; total {total / 2**30:.2f} GiB")
json.dump({"metadata": {"total_size": total}, "weight_map": new_map}, open(DST / "model.safetensors.index.json", "w"), indent=1)

c = json.load(open(SRC / "config.json"))
ign = c["quantization_config"]["ignore"]
extra = [f"thinker.model.layers.{i}.self_attn.{p}_proj" for i in sorted(layers) for p in "qkvo"]
c["quantization_config"]["ignore"] = ign + [x for x in extra if x not in ign]
json.dump(c, open(DST / "config.json", "w"), indent=1)
for f in SRC.iterdir():
    if f.is_file() and not f.name.startswith("model") and f.name not in ("config.json", "README.md", ".gitattributes"):
        shutil.copy(f, DST / f.name)
log(f"HYBRID_DONE {DST}: {len(new_map)} tensors, ignore list +{len(extra)}")
