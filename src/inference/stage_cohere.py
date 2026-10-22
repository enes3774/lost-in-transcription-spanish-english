"""Cohere Transcribe (2B conformer encoder-decoder, full fine-tune on Bangor Miami) stage: whole clips, forced
Spanish prompt (the code-switched Miami targets were trained under it), built-in chunking above 35 s, then the
runaway-repetition guard (this model loops on one-word clips). Custom model code ships inside the model dir and is
loaded with trust_remote_code. usage: stage_cohere.py <model_dir> <data_dir> <out_csv> [batch] [beams]
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import load_audio, log, read_manifest, write_csv  # noqa: E402

model_dir, data_dir, out_csv = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
bs = int(sys.argv[4]) if len(sys.argv) > 4 else 16
beams = int(sys.argv[5]) if len(sys.argv) > 5 else 1
LANG = "es"


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
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    rows = read_manifest(data_dir)
    log(f"cohere: {len(rows)} clips, lang {LANG}, batch {bs}")
    t = time.time()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True,
                                                      dtype=torch.bfloat16).to("cuda").eval()
    log(f"cohere model ready in {time.time() - t:.0f}s")
    if beams > 1:   # COHERE_BEAMS_PATCH: transcribe() hard-codes num_beams=1
        _orig = model.generate

        def _gen(*a, **kw):
            return _orig(*a, **dict(kw, num_beams=beams, cache_implementation="dynamic"))
        model.generate = _gen
    log(f"cohere beams {beams}")
    order = sorted(rows, key=lambda r: -float(r.get("file_duration_seconds") or 0))
    preds, n_loop = {}, 0
    CH = 128
    for ci in range(0, len(order), CH):
        part = order[ci:ci + CH]
        wavs = [load_audio(data_dir / "clips" / r["audio_filename"]) for r in part]
        with torch.no_grad():
            texts = model.transcribe(processor, LANG, audio_arrays=wavs, sample_rates=[16000] * len(wavs), batch_size=bs)
        assert len(texts) == len(part), (len(texts), len(part))
        for r, x in zip(part, texts):
            raw = " ".join(str(x).split())
            c = collapse(raw)
            n_loop += len(raw.split()) > 2 * len(c.split()) + 3
            preds[r["audio_filename"]] = c
        write_csv(out_csv, rows, preds)
        log(f"cohere {min(ci + CH, len(order))}/{len(order)} clips, {n_loop} looping clips collapsed")
    write_csv(out_csv, rows, preds)
    log(f"cohere done in {time.time() - t:.0f}s: {sum(1 for v in preds.values() if v.strip())}/{len(rows)} non-empty")


if __name__ == "__main__":
    main()
