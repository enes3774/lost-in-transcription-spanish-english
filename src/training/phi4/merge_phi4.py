"""Shrink the Phi-4 member without changing what the stage computes:
  1. write our Miami delta INTO the base shards (the stage copies exactly these tensors over the base at load time), so the
     1.8 GB delta file is no longer needed; a tiny delta with ONE tensor (already equal to the merged value) keeps
     stage_phi4.py and main.py's pre-flight unchanged;
  2. ZERO_VISION=1: fill the image tower and the vision LoRA (never used for audio-only input) with zeros, so they deflate to
     nothing in the zip. Names, shapes and dtypes stay as they are.
Every tensor of the result is compared bit by bit with what it must be. CPU only.
usage: merge_phi4.py <base_dir> <delta.safetensors> <out_dir> <out_delta_dir>"""
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

base, delta_path, out, out_delta = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
ZERO = os.environ.get("ZERO_VISION", "0") == "1"
out.mkdir(parents=True, exist_ok=True); out_delta.mkdir(parents=True, exist_ok=True)
delta = load_file(delta_path)
bkey = lambda k: k.replace("._checkpoint_wrapped_module", "")  # noqa: E731  (the loaded model wraps the encoder blocks)
dmap = {bkey(k): k for k in delta}
assert len(dmap) == len(delta), "delta names collide after mapping"
is_vision = lambda k: ".image_embed." in k or "lora_A.vision" in k or "lora_B.vision" in k  # noqa: E731
bits = lambda t: t.contiguous().view(torch.int16) if t.dtype == torch.bfloat16 else t  # noqa: E731
used, zeroed, zbytes, total = set(), 0, 0, 0
for f in sorted(os.listdir(base)):
    src = base / f
    if not (f.startswith("model-") and f.endswith(".safetensors")):
        if src.is_file():
            shutil.copy2(src, out / f)                     # small files (config, code, tokenizer): real copies
        elif src.is_dir():
            shutil.copytree(src, out / f)
        continue
    sd = load_file(str(src))
    new = {}
    for k, v in sd.items():
        if k in dmap:
            d = delta[dmap[k]]
            assert d.shape == v.shape and d.dtype == v.dtype, (k, d.shape, v.shape, d.dtype, v.dtype)
            assert not is_vision(k), f"delta touches a vision tensor: {k}"
            new[k] = d; used.add(k)
        elif ZERO and is_vision(k):
            new[k] = torch.zeros_like(v); zeroed += 1; zbytes += v.numel() * v.element_size()
        else:
            new[k] = v
        total += v.numel() * v.element_size()
    save_file(new, str(out / f), metadata={"format": "pt"})
    back = load_file(str(out / f))                         # bit-by-bit check of what was written
    assert set(back) == set(sd), f
    for k, v in sd.items():
        want = delta[dmap[k]] if k in dmap else (torch.zeros_like(v) if ZERO and is_vision(k) else v)
        assert back[k].dtype == v.dtype and back[k].shape == v.shape and torch.equal(bits(back[k]), bits(want)), k
    print(f"{f}: {len(sd)} tensors, {sum(1 for k in sd if k in dmap)} from the delta, {sum(1 for k in sd if ZERO and is_vision(k) and k not in dmap)} zeroed -> verified", flush=True)
assert used == set(dmap), f"{len(set(dmap) - used)} delta tensors not found in the base, e.g. {sorted(set(dmap) - used)[:2]}"
small = min(delta, key=lambda k: delta[k].numel())
save_file({small: delta[small]}, str(out_delta / "delta.safetensors"))
(out_delta / "SOURCE.txt").write_text("The Miami delta is merged into models/phi4 (merge_phi4.py); this file holds one tensor that already equals the merged value.\n")
idx = json.load(open(base / "model.safetensors.index.json"))
assert json.load(open(out / "model.safetensors.index.json")) == idx
print(f"MERGE_OK: {len(used)} delta tensors merged, {zeroed} vision tensors zeroed ({zbytes / 1e9:.2f} GB of {total / 1e9:.2f} GB), tiny delta = {small} {tuple(delta[small].shape)}")
