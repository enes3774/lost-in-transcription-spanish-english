"""V9: v8 + Whisper on 0.9x/1.1x speed-perturbed audio (2 voters, w8 each) + Phi-4-multimodal Miami fine-tune (w8).
Validated (xeval tta_whisper_phi4): dev 0.0755->0.0751, glued long 0.0826->0.0808, Miami half 0.1846->0.1799.

Lost in Transcription (Spanish-English) - v7 entrypoint: five members -> guarded vote -> Omni refiner -> verified writing rules.

v7 vs v6: (1) every stage's stdout/stderr is written to WORK/logs/<stage>.log and only our own summary lines reach
the job log, because the platform keeps the first 500 lines only and vLLM output hid everything after the Voxtral
stage in v4-v6; (2) the underscore rule in spelling_rules.option2 (Miami-style Curious_George / T_V -> Curious George /
TV; the references never contain "_"); (3) refiner delta retrained on Miami targets that keep the cut-off-word
convention ("gro..."), which the references use and no member had ever produced.

Members (each runs as its own subprocess so GPU memory is fully released between them):
  omni       Qwen3-Omni-30B-A3B thinker, untrained, AWQ 4-bit, vLLM            (anchor + fallback)   whole clips
  voxtral    Voxtral-Small-24B FP8 + Miami LoRA, vLLM                                                  whole clips
  whisper    Whisper large-v3 + Miami LoRA (merged), HF pipeline fp16                                  split > 30 s
  conformer  Conformer-CTC, plain PyTorch port                                                         split > 60 s
  qwen3asr   Qwen3-ASR-1.7B + Miami LoRA (merged), beam 4 (greedy if late)                             split > 45 s

Long-clip splitting (seg_long.py). Measured on dev clips glued to ~220 s: Whisper drops ~20% of the words
(0.27 -> 0.11 WER once split), Qwen3-ASR truncates (0.53 -> 0.13), the conformer port crashes above 200 s, which
would drop that member for every clip. Omni and
Voxtral are unaffected up to 274 s and keep whole clips. Thresholds come from dev's natural 31-60 s clips: only
Whisper benefits below 60 s; the conformer is slightly worse when split there, so it is left alone until 60 s.

v8 STRICT_V8: there is no fallback anywhere. Every member must finish with a complete output, otherwise the job
exits with an error and leaves NO submission file, so the platform refunds the slot and the job log says what broke.
The submission file is written once, by the vote, at the very end.

v6 adds the refiner (stage_refine.py): the Omni thinker with an attention-only LoRA trained on Miami out-of-fold
hypotheses re-transcribes each clip given the audio and the five member transcripts; measured on dev (never used for
training or selection) 0.0829 -> 0.0805 with the rules on, and 0.0944 -> 0.0892 on clips glued to 60-274 s.
models/omni is now a hybrid checkpoint: the same AWQ int4 experts, attention in bf16 so the delta merges exactly.

Layout: main.py common.py vote.py seg_long.py spelling_rules.py stage_*.py conformer_port.py vendor/
        models/{omni,voxtral,whisper,conformer,qwen3asr,refiner}
"""
import os
import re
import contextlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import log, read_csv, read_manifest, write_csv  # noqa: E402
import seg_long  # noqa: E402

T0 = time.time()
CWD = Path.cwd()
DATA = Path(os.environ.get("OMNI_DATA_DIR", CWD / "data"))
OUT = Path(os.environ.get("OMNI_OUT_PATH", CWD / "submission" / "submission.csv"))
WORK = Path(os.environ.get("OMNI_WORK_DIR", CWD / "work_v7"))
LOGS = WORK / "logs"
MODELS = Path(os.environ.get("OMNI_MODELS_DIR", HERE / "models"))
BUDGET = float(os.environ.get("OMNI_TIME_BUDGET_S", 100 * 60))   # unzip already happened before this clock
SPLIT_OVER_S = {"whisper": 30.0, "qwen3asr": 45.0, "conformer": 60.0, "cohere": 30.0, "parakeet": 30.0, "phi4": 30.0}   # members not listed decode whole clips
MAX_PIECE_S, MIN_PIECE_S = 28.0, 12.0
MEMBERS = ("omni", "voxtral", "whisper", "conformer", "qwen3asr", "cohere", "parakeet", "phi4", "whisper_s090", "whisper_s110")   # cohere (v7) is optional: only if models/cohere exists



def abort(msg):
    """STRICT_V8: stop the whole job without a submission file (the platform then refunds the slot)."""
    log(f"ABORT: {msg} -> no submission written")
    if OUT.exists():
        OUT.unlink()
    sys.exit(1)


WORK.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    OUT.unlink()
rows = read_manifest(DATA)
log(f"v9 start: {len(rows)} clips")
_vz = HERE / "vendor_tf.zip"                      # v8: the parakeet stage's libraries ship as ONE nested zip (the runner's
if _vz.is_file() and not (HERE / "vendor_tf").is_dir():   # unzip listing of 3,482 loose files flooded the 500-line job log)
    import zipfile as _zipfile
    with _zipfile.ZipFile(_vz) as _z:
        _z.extractall(HERE / "vendor_tf")
    log(f"vendor_tf unpacked next to main.py: {sum(1 for _p in (HERE / 'vendor_tf').rglob('*') if _p.is_file())} files")
if os.environ.get("LOST_IN_TRANSCRIPTION_IS_SMOKE", "0") == "1" and os.environ.get("OMNI_SMOKE_FULL", "0") != "1":
    write_csv(OUT, rows, {})
    log("smoke run: empty transcripts written")
    sys.exit(0)

# STRICT_V8 pre-flight: everything the stages will need must exist before any model is loaded.
NEED = {"omni": ["config.json", "model.safetensors.index.json"], "omni_delta": ["delta.pt"],
        "voxtral": ["params.json", "tekken.json", "consolidated.safetensors", "lora/adapter_config.json", "lora/adapter_model.safetensors"],
        "whisper": ["config.json"], "conformer": ["conformer_miami.pt", "tokenizer.model"], "qwen3asr": ["config.json"],
        "cohere": ["config.json", "model.safetensors"], "parakeet": ["config.json", "generation_config.json", "model.safetensors"],
        "phi4": ["config.json", "model.safetensors.index.json", "modeling_phi4mm.py", "speech_conformer_encoder.py", "processing_phi4mm.py"],
        "phi4_delta": ["delta.safetensors"]}
_missing = [f"{m}/{f}" for m, fs in NEED.items() for f in fs if not (MODELS / m / f).is_file()]
if _missing:
    abort(f"model files missing: {_missing[:5]}")
if not (HERE / "vendor_tf" / "transformers" / "__init__.py").is_file():
    abort("vendor_tf (transformers for the parakeet stage) is not unpacked")
if not (HERE / "vendor_phi" / "backoff" / "__init__.py").is_file():
    abort("vendor_phi/backoff (needed by the Phi-4 remote code) is missing")
if not rows:
    abort("empty test_metadata.csv")
_no_audio = [r["audio_filename"] for r in rows if not (DATA / "clips" / r["audio_filename"]).is_file()]
if _no_audio:
    abort(f"{len(_no_audio)} audio files missing")
log("pre-flight ok: 8 models (+2 TTA passes) + omni delta + phi4 delta + vendor_tf + vendor_phi + all audio present")


def remaining():
    return BUDGET - (time.time() - T0)


OURS = re.compile(r"^\[\s*[\d.]+s\] ")   # our own log() lines inside a stage log


def run_stage(name, args, env=None, timeout=None):
    """Run a stage as a subprocess. Its full stdout/stderr goes to WORK/logs/<name>.log (vLLM alone prints
    hundreds of lines and the platform keeps only the first 500 of the job log); here we print one start line,
    one exit line, and the stage's own summary lines, so the job log stays readable to the end."""
    t = time.time()
    log(f"stage {name} start")
    LOGS.mkdir(parents=True, exist_ok=True)
    lf = LOGS / f"{name}.log"
    try:
        with open(lf, "w", encoding="utf-8", errors="replace") as f:
            p = subprocess.run([sys.executable] + args, env=dict(os.environ, **(env or {})), cwd=str(CWD),
                               timeout=timeout, stdout=f, stderr=subprocess.STDOUT)
        rc = p.returncode
        status = f"exit {rc}"
    except subprocess.TimeoutExpired:
        rc, status = None, "TIMEOUT"
    try:
        lines = open(lf, encoding="utf-8", errors="replace").read().splitlines()
        ours = [l for l in lines if OURS.match(l)]
        for l in ours[-10:]:
            log(f"  [{name}] {l[l.index(']') + 2:][:150]}")
        if rc not in (0, None):
            for l in [l for l in lines if re.search(r"Error|Traceback|Killed|out of memory", l)][-4:]:
                log(f"  [{name}] {l.strip()[:150]}")
        log(f"stage {name} {status} in {time.time() - t:.0f}s ({len(lines)} log lines in {lf.name})")
    except Exception as e:
        log(f"stage {name} {status} in {time.time() - t:.0f}s (log unreadable: {type(e).__name__})")
    if rc != 0:
        abort(f"stage {name} {status}")


def check_member(name):
    """STRICT_V8: a member counts only if its CSV has every clip and >= 90 % of them non-empty (vote.py's own bar)."""
    d = read_csv(csvs[name])
    n_missing = sum(1 for r in rows if r["audio_filename"] not in d)
    nonempty = sum(1 for r in rows if (d.get(r["audio_filename"]) or "").strip())
    if n_missing or nonempty < 0.9 * len(rows):
        abort(f"member {name} output incomplete ({n_missing} clips missing, {nonempty}/{len(rows)} non-empty)")
    log(f"member {name} ok: {nonempty}/{len(rows)} non-empty")


def data_for(name):
    """Whole-clip data dir, or a per-member copy in which clips over that member's limit are cut at pauses."""
    if name not in SPLIT_OVER_S:
        return DATA
    d = WORK / f"seg_{name}"
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / f"seg_{name}.log", "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        seg_long.split(DATA, d, SPLIT_OVER_S[name], MAX_PIECE_S, MIN_PIECE_S)   # its summary lines stay out of the job log
    return d


def finish(name, stage_data, pieces_csv, final_csv):
    """Join piece transcripts back into one row per original clip (no-op for whole-clip members)."""
    if not pieces_csv.exists():
        abort(f"member {name} wrote no output file")
    if stage_data == DATA:
        if pieces_csv != final_csv:
            shutil.copy(pieces_csv, final_csv)
        return
    with open(LOGS / f"seg_{name}.log", "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        seg_long.merge(stage_data, pieces_csv, final_csv)


csvs = {k: WORK / f"{k}.csv" for k in MEMBERS}
have = {k: (MODELS / k).exists() for k in MEMBERS}
log(f"models present: {[k for k, v in have.items() if v]}")

# 1. Omni (anchor)
if True:
    run_stage("omni", [str(HERE / "stage_omni.py")],
              env={"OMNI_MODEL_DIR": str(MODELS / "omni"), "OMNI_DATA_DIR": str(DATA), "OMNI_OUT_PATH": str(csvs["omni"]),
                   "OMNI_TIME_BUDGET_S": str(max(600, remaining() - 1800)), "OMNI_MOE_BACKEND": os.environ.get("OMNI_MOE_BACKEND", "auto")})
    if Path(str(csvs["omni"]) + ".delta_failed").exists():   # v8: no fallback -> fail the job, no submission
        log("omni delta merge failed: aborting without a submission")
        if OUT.exists():
            OUT.unlink()
        sys.exit(1)
    check_member("omni")

# 2. Voxtral (whole clips)
if remaining() <= 900:
    abort(f"only {remaining():.0f}s left before voxtral")
run_stage("voxtral", [str(HERE / "stage_voxtral.py"), str(MODELS / "voxtral"), str(DATA), str(csvs["voxtral"])],
          timeout=max(300, remaining() - 900))
check_member("voxtral")

# 3-5. The three members that need long clips cut up
PLAN = [("whisper", "stage_whisper.py", lambda: ["16"], 600),
        ("conformer", "stage_conformer.py", lambda: [], 300),
        ("qwen3asr", "stage_qwen3asr.py", lambda: ["4"], 600),              # beam 4, as validated (STRICT_V8: no greedy downgrade)
        ("cohere", "stage_cohere.py", lambda: ["16", "4"], 300),            # Cohere Transcribe 2B, Miami full fine-tune, 4 beams
        ("parakeet", "stage_parakeet.py", lambda: ["16"], 300),   # Parakeet Ultra 0.6B TDT, Miami fine-tune
        ("phi4", "stage_phi4.py", lambda: ["8"], 600)]            # v9: Phi-4-multimodal + Miami delta, greedy, batch 8
for name, script, extra, need in PLAN:
    if remaining() <= need:
        abort(f"only {remaining():.0f}s left before {name}")
    d = data_for(name)
    pieces = WORK / f"{name}_pieces.csv" if d != DATA else csvs[name]
    run_stage(name, [str(HERE / script), str(MODELS / name), str(d), str(pieces)] + extra(),
              timeout=max(need // 2, remaining() - 240))
    finish(name, d, pieces, csvs[name])
    check_member(name)

# 6. v9 TTA: Whisper again on 0.9x and 1.1x speed-perturbed audio (resampled copies, cut > 30 s like the original)
for factor, tag in ((0.9, "whisper_s090"), (1.1, "whisper_s110")):
    if remaining() <= 600:
        abort(f"only {remaining():.0f}s left before {tag}")
    td = WORK / f"tta_{tag}"
    run_stage(f"tta_{tag}", [str(HERE / "stage_tta.py"), str(DATA), str(td), str(factor)], timeout=900)
    if sum(1 for _ in (td / "clips").iterdir()) != len(rows):
        abort(f"tta audio for {tag} is incomplete")
    d = WORK / f"seg_{tag}"
    with open(LOGS / f"seg_{tag}.log", "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
        seg_long.split(td, d, SPLIT_OVER_S["whisper"], MAX_PIECE_S, MIN_PIECE_S)
    pieces = WORK / f"{tag}_pieces.csv"
    run_stage(tag, [str(HERE / "stage_whisper.py"), str(MODELS / "whisper"), str(d), str(pieces), "16"],
              timeout=max(300, remaining() - 240))
    finish(tag, d, pieces, csvs[tag])
    check_member(tag)

# 7. Vote (repetition guard on every member, then the verified writing rules on the final text)
run_stage("vote", [str(HERE / "vote.py"), str(DATA), str(OUT)] + [f"{k}={v}" for k, v in csvs.items()],
          env={"V6_VOTE_RAW": str(WORK / "vote_raw.csv"), "V8_STRICT": "1"})
if not OUT.is_file():
    abort("vote wrote no submission file")

# 7. STRICT_V8 final check: the vote must have written one row per clip, (almost) all non-empty
final = read_csv(OUT)
if sum(1 for r in rows if r["audio_filename"] in final) != len(rows):
    abort("vote output is missing clips")
if sum(1 for r in rows if (final.get(r["audio_filename"]) or "").strip()) < 0.98 * len(rows):
    abort("vote output has too many empty transcripts")
log(f"v9 done in {time.time() - T0:.0f}s: {sum(1 for r in rows if final.get(r['audio_filename'], '').strip())}/{len(rows)} non-empty transcripts")
