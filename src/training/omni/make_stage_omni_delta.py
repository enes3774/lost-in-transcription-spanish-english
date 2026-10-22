"""Build v7_stage/stage_omni.py: the v7 Omni stage plus an optional attention LoRA delta (models/omni_delta/delta.pt or
OMNI_DELTA). Clips longer than OMNI_DELTA_MAX_S (default 45 s) are decoded FIRST with the base weights (longest-first order
is kept), then the delta is merged into the running engine exactly as stage_refine.py does, and the rest is decoded."""
import py_compile
from pathlib import Path

src = Path("/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track/submission_v7_code/stage_omni.py").read_text(encoding="utf-8")
helper = '''
_DELTA_DEFAULT = Path(__file__).resolve().parent / "models" / "omni_delta" / "delta.pt"
if Path(os.environ.get("OMNI_DELTA") or _DELTA_DEFAULT).is_file():
    # the delta is pushed into the running engine with collective_rpc(<function>), which needs the in-process
    # engine core (same setting as stage_refine.py and the lab); the default multiprocess core cannot serialise it
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

ATTN_RE = re.compile(r"^thinker\\.model\\.layers\\.\\d+\\.self_attn\\.(q|k|v|o)_proj\\.weight$")


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


def write_csv(path, rows, preds):'''
assert src.count("\ndef write_csv(path, rows, preds):") == 1
src = src.replace("\ndef write_csv(path, rows, preds):", helper, 1)
start = src.index('    log("engine ready")\n') + len('    log("engine ready")\n')
end = src.rindex('    write_csv(out_path, rows, preds)\n    log("done")')
body = '''
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
                log("time budget hit: remaining clips left empty")
                state["stopped"] = True
                return
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
        except Exception as e:  # never lose the run: finish with base weights
            log(f"delta merge failed ({type(e).__name__}): continuing with base weights")
            decode_rows([r for r in short_rows if r["audio_filename"] not in preds], "base")

'''
src = src[:start] + body + src[end:]
out = Path("/kaggle/working/omni_rt/v7_stage/stage_omni.py")
out.write_text(src, encoding="utf-8")
py_compile.compile(str(out), doraise=True)
print("v7_stage/stage_omni.py written and compiles")
