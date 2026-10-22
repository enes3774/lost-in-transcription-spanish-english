"""Parakeet member stage: moondream/parakeet-ultra fine-tuned on Bangor Miami (ParakeetForTDT, 0.6B).

usage: stage_parakeet.py <model_dir> <data_dir> <out_csv> [batch]

The runner's own transformers may predate ParakeetForTDT (it landed in 5.6). If a vendored library tree sits
next to the package (vendor_tf/, holding transformers + tokenizers + huggingface_hub and their dist-info),
it is put FIRST on sys.path here, so only this stage sees the newer libraries and every other member keeps
the runtime's own versions. If the runtime is already new enough, the vendor tree is simply not needed.

Like every other member the stage writes its CSV after each chunk, so a later failure still leaves partial output.
"""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = Path(os.environ.get("PARAKEET_VENDOR", HERE / "vendor_tf"))
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(HERE))
from common import load_audio, log, read_manifest, write_csv  # noqa: E402

model_dir, data_dir, out_csv = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
bs = int(sys.argv[4]) if len(sys.argv) > 4 else 16


def collapse(text, max_uni=4, max_ngram=2):
    """Same runaway-repetition guard every other member passes through."""
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


def main():
    import torch
    import transformers
    from transformers import AutoProcessor, GenerationConfig, ParakeetForTDT

    rows = read_manifest(data_dir)
    log(f"parakeet: {len(rows)} clips, batch {bs}, transformers {transformers.__version__}"
        f"{' (vendored)' if VENDOR.is_dir() else ''}")
    t = time.time()
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = ParakeetForTDT.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16).to("cuda").eval()
    model.generation_config = GenerationConfig.from_pretrained(model_dir, local_files_only=True)   # checked in pre-flight
    log(f"parakeet model ready in {time.time() - t:.0f}s")

    def decode(wavs):
        enc = processor(wavs, sampling_rate=16000, return_tensors="pt", padding=True)
        enc = {k: v.to("cuda") for k, v in enc.items() if torch.is_tensor(v)}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen = model.generate(**enc)
        seq = gen.sequences if hasattr(gen, "sequences") else gen
        return processor.batch_decode(seq, skip_special_tokens=True)

    order = sorted(rows, key=lambda r: -float(r.get("file_duration_seconds") or 0))
    preds, n_loop = {}, 0
    for ci in range(0, len(order), bs):
        part = order[ci:ci + bs]
        wavs = [load_audio(data_dir / "clips" / r["audio_filename"]) for r in part]
        texts = decode(wavs)
        assert len(texts) == len(part), (len(texts), len(part))
        for r, x in zip(part, texts):
            raw = " ".join(str(x).split())
            c = collapse(raw)
            n_loop += len(raw.split()) > 2 * len(c.split()) + 3
            preds[r["audio_filename"]] = c
        if (ci // bs) % 8 == 0:
            write_csv(out_csv, rows, preds)
            log(f"parakeet {min(ci + bs, len(order))}/{len(order)} clips, {n_loop} looping clips collapsed")
    write_csv(out_csv, rows, preds)
    log(f"parakeet done in {time.time() - t:.0f}s: {sum(1 for v in preds.values() if v.strip())}/{len(rows)} non-empty, "
        f"{n_loop} collapsed")


if __name__ == "__main__":
    main()
