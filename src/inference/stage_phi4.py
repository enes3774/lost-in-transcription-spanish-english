"""Phi-4-multimodal-instruct stage (v9): the public model (ZFTurbo fork = microsoft weights + a generate() fix) plus
our Miami A+B fine-tune of its speech LoRA, audio encoder and speech projection, shipped as delta.safetensors and
copied over the base tensors at load time (every delta name must exist in the model). Greedy, batch 8, clips sorted by
length, ASR prompt from the model card; long clips arrive already cut (> 30 s) by main.py. Remote model code needs the
small pure-python `backoff` package, vendored in vendor_phi/.
usage: stage_phi4.py <model_dir> <data_dir> <out_csv> [batch]   (delta: <model_dir>_delta/delta.safetensors)
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor_phi"))
from common import load_audio, log, read_manifest, write_csv  # noqa: E402

model_dir, data_dir, out_csv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
bs = int(sys.argv[4]) if len(sys.argv) > 4 else 8
delta_path = Path(str(model_dir) + "_delta") / "delta.safetensors"
SR = 16000
PROMPT = "<|user|><|audio_1|>Transcribe the audio clip into text.<|end|><|assistant|>"


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


def main():
    import backoff  # noqa: F401  (fail here, clearly, if the vendored package is missing)
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
    rows = read_manifest(data_dir)
    log(f"phi4: {len(rows)} clips, batch {bs}")
    t = time.time()
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True,
                                                 torch_dtype=torch.bfloat16, _attn_implementation="sdpa").to("cuda").eval()
    gcfg = GenerationConfig.from_pretrained(str(model_dir))
    dl = load_file(str(delta_path))
    sd = dict(model.named_parameters())
    missing = [k for k in dl if k not in sd]
    assert not missing, f"{len(missing)} delta tensors not in the model, e.g. {missing[:2]}"
    with torch.no_grad():
        for k, v in dl.items():
            assert sd[k].shape == v.shape, (k, sd[k].shape, v.shape)
            sd[k].copy_(v.to(sd[k].dtype))
    log(f"phi4 model ready in {time.time() - t:.0f}s, Miami delta: {len(dl)} tensors")
    dur = {r["audio_filename"]: float(r.get("file_duration_seconds") or 0) for r in rows}
    order = sorted(rows, key=lambda r: -dur[r["audio_filename"]])
    preds, n_loop = {}, 0
    for i in range(0, len(order), bs):
        part = order[i:i + bs]
        wavs = [load_audio(data_dir / "clips" / r["audio_filename"]) for r in part]
        mx = max(64, int(max(dur[r["audio_filename"]] for r in part) * 12) + 64)
        inputs = processor(text=[PROMPT] * len(part), audios=[(w, SR) for w in wavs], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            g = model.generate(**inputs, max_new_tokens=mx, generation_config=gcfg, do_sample=False, num_logits_to_keep=1)
        texts = processor.batch_decode(g[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        assert len(texts) == len(part), (len(texts), len(part))
        for r, x in zip(part, texts):
            raw = " ".join(str(x).split())
            c = collapse(raw)
            n_loop += len(raw.split()) > 2 * len(c.split()) + 3
            preds[r["audio_filename"]] = c
        if (i // bs) % 50 == 0:
            write_csv(out_csv, rows, preds)
            log(f"phi4 {min(i + bs, len(order))}/{len(order)} clips, {n_loop} looping clips collapsed")
    write_csv(out_csv, rows, preds)
    log(f"phi4 done in {time.time() - t:.0f}s: {sum(1 for v in preds.values() if v.strip())}/{len(rows)} non-empty")


if __name__ == "__main__":
    main()
