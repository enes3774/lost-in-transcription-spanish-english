"""Qwen3-Omni-30B-A3B-Instruct (thinker only, bf16) ASR submission via vLLM.

Layout inside submission.zip:
    main.py
    model/            config.json, *.safetensors (thinker + audio tower), tokenizer/processor files

Environment knobs (all optional):
    OMNI_DATA_DIR        data dir                          (default: <cwd>/data)
    OMNI_OUT_PATH        output csv                        (default: <cwd>/submission/submission.csv)
    OMNI_MODEL_DIR       model dir                         (default: <this file>/model)
    OMNI_GPU_UTIL        vLLM gpu_memory_utilization        (default: 0.92)
    OMNI_MAX_SEQS        vLLM max_num_seqs                  (default: 256)
    OMNI_BATCHED_TOKENS  vLLM max_num_batched_tokens        (default: 32768)
    OMNI_CHUNK           requests per generate() call      (default: 512)
    OMNI_TIME_BUDGET_S   stop starting new chunks after    (default: 6000 = 100 min; unzip runs before this clock)
    OMNI_EAGER           1 = disable CUDA graphs            (default: 0)
    OMNI_MOE_BACKEND     vLLM moe_backend                   (default: triton)
    OMNI_LIMIT           only first N clips (debug)         (default: all)
"""
import csv
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

T0 = time.time()

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")  # platform log cap: 500 lines

SR = 16000
TOKENS_PER_SECOND = 20
MIN_NEW_TOKENS = 32
TOKENS_MARGIN = 64

SYSTEM_PROMPT = "You are a verbatim speech transcription engine."
PROMPT = (
    "Transcribe this audio word for word.\n"
    "The speaker code-switches freely between English and Spanish, often inside a single "
    "sentence. Write every word in the language it was actually spoken in. "
    "Never translate between English and Spanish: if a phrase is spoken in English, "
    "keep it in English; if it is spoken in Spanish, keep it in Spanish.\n"
    "Do not summarise, do not rephrase, do not fix grammar, do not add words that were not said.\n"
    "Output only the transcript itself, with no preamble, no quotes and no commentary."
)
PREAMBLE_RE = re.compile(
    r"^\s*(here\s+is\s+(the\s+)?(verbatim\s+)?transcript(ion)?|transcript(ion)?|"
    r"aqu[ií]\s+est[aá]\s+la\s+transcripci[oó]n|transcripci[oó]n)\s*[:\-]\s*",
    re.IGNORECASE,
)


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def _patch_vllm_deepstack():
    """vLLM 0.23 bug: with image=video=0 the vision tower is skipped, but its deepstack buffers
    are still allocated on `meta` and then added to the text hidden states ("Tensor on device
    meta is not on the expected device cuda:0"). For audio-only inputs those buffers are all
    zeros, so treating a meta buffer as absent is exact.

    Runs at import time: vLLM's engine core is a spawned child that re-imports this file as
    __mp_main__ (or a forked child that inherits the patched class), so the patch reaches it.
    """
    from vllm.model_executor.models import qwen3_omni_moe_thinker as m   # STRICT_V8: must import on the runner
    cls = m.Qwen3OmniMoeThinkerForConditionalGeneration
    if getattr(cls, "_omni_deepstack_patched", False):
        return

    def live(self):
        buf = getattr(self, "deepstack_input_embeds", None)
        return bool(buf) and buf[0].device.type != "meta"

    for name in ("_get_deepstack_input_embeds", "_set_deepstack_input_embeds",
                 "_clear_deepstack_input_embeds"):
        def wrapped(self, *args, _orig=getattr(cls, name), **kwargs):
            return _orig(self, *args, **kwargs) if live(self) else None
        setattr(cls, name, wrapped)
    cls._omni_deepstack_patched = True


_patch_vllm_deepstack()


def clean_prediction(text):
    t = text.strip()
    t = PREAMBLE_RE.sub("", t).strip()
    if len(t) > 1 and t[0] in '"«“' and t[-1] in '"»”':
        t = t[1:-1].strip()
    return " ".join(t.split())


def load_audio(path):
    import numpy as np
    try:
        import soundfile as sf
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
    except Exception:
        import librosa
        wav, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != SR:
        import soxr
        wav = soxr.resample(wav, sr, SR)
    return np.ascontiguousarray(wav, dtype=np.float32)


_DELTA_DEFAULT = Path(__file__).resolve().parent / "models" / "omni_delta" / "delta.pt"
if Path(os.environ.get("OMNI_DELTA") or _DELTA_DEFAULT).is_file():
    # the delta is pushed into the running engine with collective_rpc(<function>), which needs the in-process
    # engine core (same setting as stage_refine.py and the lab); the default multiprocess core cannot serialise it
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

ATTN_RE = re.compile(r"^thinker\.model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj\.weight$")


def merge_delta(llm, model_dir, delta_pt):
    """W <- W + scale * B @ A for the attention projections, pushed into the running engine (same as stage_refine)."""
    import json
    import torch
    from safetensors import safe_open
    t = time.time()
    d = torch.load(str(delta_pt), map_location="cpu")
    idx = json.load(open(model_dir / "model.safetensors.index.json"))["weight_map"]
    by_file = {}
    for k in d["A"]:
        assert ATTN_RE.match(k), k
        by_file.setdefault(idx[k], []).append(k)
    named = []
    for f, ks in by_file.items():
        with safe_open(str(model_dir / f), framework="pt") as fh:
            for k in ks:
                w = fh.get_tensor(k)
                named.append((k, (w.float() + d["scale"] * (d["B"][k].float() @ d["A"][k].float())).to(w.dtype)))

    def push(worker, named=named):
        r = worker.model_runner
        (r.get_model() if hasattr(r, "get_model") else r.model).load_weights(iter(named))
        torch.cuda.synchronize()
        return True
    llm.collective_rpc(push)
    log(f"omni delta merged into {len(named)} attention tensors in {time.time() - t:.0f}s")


def write_csv(path, rows, preds):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audio_filename", "transcript"])
        for r in rows:
            w.writerow([r["audio_filename"], preds.get(r["audio_filename"], "")])
    os.replace(tmp, path)


def main():
    cwd = Path.cwd()
    data_dir = Path(os.environ.get("OMNI_DATA_DIR", cwd / "data"))
    out_path = Path(os.environ.get("OMNI_OUT_PATH", cwd / "submission" / "submission.csv"))
    model_dir = Path(os.environ.get("OMNI_MODEL_DIR", Path(__file__).resolve().parent / "model"))
    budget = float(os.environ.get("OMNI_TIME_BUDGET_S", 6000))
    chunk = int(os.environ.get("OMNI_CHUNK", 512))

    with open(data_dir / "test_metadata.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if os.environ.get("OMNI_LIMIT"):
        rows = rows[: int(os.environ["OMNI_LIMIT"])]
    preds = {}
    write_csv(out_path, rows, preds)  # always leave a valid file behind
    log("start")

    # Smoke test (1-minute budget) cannot load a 60 GB model: only prove I/O.
    if os.environ.get("LOST_IN_TRANSCRIPTION_IS_SMOKE", "0") == "1" and os.environ.get("OMNI_SMOKE_FULL", "0") != "1":
        log("smoke run: wrote empty transcripts")
        return

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(str(model_dir))
    conv = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "audio", "audio": "x"}, {"type": "text", "text": PROMPT}]},
    ]
    prompt_text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)

    # Longest first: the long tail is where the audio encoder and memory are stressed, so
    # failures surface early; and similar lengths inside a chunk minimise stragglers.
    order = sorted(rows, key=lambda r: -float(r.get("file_duration_seconds") or 0))

    pool = ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4))
    clip_dir = data_dir / "clips"
    chunks = [order[i:i + chunk] for i in range(0, len(order), chunk)]
    # decode the first chunk while the engine loads
    pending = pool.map(lambda r: load_audio(clip_dir / r["audio_filename"]), chunks[0]) if chunks else None

    llm = LLM(
        model=str(model_dir),
        dtype="bfloat16",
        max_model_len=int(os.environ.get("OMNI_MAX_MODEL_LEN", 8192)),
        max_num_seqs=int(os.environ.get("OMNI_MAX_SEQS", 256)),
        max_num_batched_tokens=int(os.environ.get("OMNI_BATCHED_TOKENS", 32768)),
        gpu_memory_utilization=float(os.environ.get("OMNI_GPU_UTIL", 0.92)),
        # vision tower skipped (see _patch_vllm_deepstack); image profiling is broken anyway
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 0},
        # Triton fused-MoE: prebuilt in the wheel (FlashInfer would JIT-compile, and the
        # runtime image has no CUDA toolchain).
        moe_backend=os.environ.get("OMNI_MOE_BACKEND", "triton"),
        enforce_eager=os.environ.get("OMNI_EAGER", "0") == "1",
        seed=0,
    )
    log("engine ready")

    delta_pt = Path(os.environ.get("OMNI_DELTA") or (Path(__file__).resolve().parent / "models" / "omni_delta" / "delta.pt"))
    delta_max_s = float(os.environ.get("OMNI_DELTA_MAX_S", 45))
    use_delta = delta_pt.is_file()

    def dur_of(r):
        return float(r.get("file_duration_seconds") or 0)
    if use_delta:
        long_rows = [r for r in order if dur_of(r) > delta_max_s]
        short_rows = [r for r in order if dur_of(r) <= delta_max_s]
        log(f"delta {delta_pt.name}: base weights for {len(long_rows)} clips over {delta_max_s:.0f}s, delta for {len(short_rows)}")
    else:
        long_rows, short_rows = order, []
        log("no delta file: base weights for all clips")
    state = {"done": 0, "audio": 0.0, "stopped": False}
    t_gen = time.time()

    def decode_rows(part_rows, tag):
        n_chunks = (len(part_rows) + chunk - 1) // chunk
        for ci in range(0, len(part_rows), chunk):
            if time.time() - T0 > budget:
                log("time budget hit: aborting (STRICT_V8, no partial output)")
                sys.exit(4)
            part = part_rows[ci:ci + chunk]
            wavs = list(pool.map(lambda r: load_audio(clip_dir / r["audio_filename"]), part))
            reqs, params = [], []
            for r, wav in zip(part, wavs):
                dur = len(wav) / SR
                state["audio"] += dur
                reqs.append({"prompt": prompt_text, "multi_modal_data": {"audio": (wav, SR)}})
                params.append(SamplingParams(
                    temperature=0.0,
                    max_tokens=max(MIN_NEW_TOKENS, int(dur * TOKENS_PER_SECOND) + TOKENS_MARGIN),
                ))
            outs = llm.generate(reqs, params, use_tqdm=False)
            for r, o in zip(part, outs):
                preds[r["audio_filename"]] = clean_prediction(o.outputs[0].text)
            state["done"] += len(part)
            write_csv(out_path, rows, preds)
            el = time.time() - t_gen
            # progress only; no statistics about the test data (competition rule)
            log(f"{tag} chunk {ci // chunk + 1}/{n_chunks} done, generation {el:.0f}s")
            if os.environ.get("OMNI_BENCH_LOG") == "1":
                log(f"[bench] {state['done']} clips, {state['audio'] / 3600:.2f} h audio, RTFx {state['audio'] / max(el, 1e-6):.1f}")

    decode_rows(long_rows, "base" if use_delta else "chunk")
    if use_delta and not state["stopped"]:
        try:
            merge_delta(llm, model_dir, delta_pt)
            decode_rows(short_rows, "delta")
        except Exception as e:  # v8: no fallback. Abort so the whole job fails without a submission.
            log(f"delta merge failed ({type(e).__name__}): aborting (no base-weight fallback)")
            Path(str(out_path) + ".delta_failed").write_text(f"{type(e).__name__}: {e}\n")
            sys.exit(3)

    write_csv(out_path, rows, preds)
    log("done")


if __name__ == "__main__":
    main()
