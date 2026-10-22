"""Build $T/submission_v9_code from the final (patched, submitted) submission_v8_code:
v8 + Whisper on 0.9x / 1.1x speed-perturbed audio (w8 each) + Phi-4-multimodal Miami fine-tune (w8), vote order
= the v8 seven, whisper_s090, whisper_s110, phi4 (exactly the validated xeval candidate tta_whisper_phi4).
Every replacement is asserted to hit exactly once. STRICT_V8 behaviour kept, and the vote now refuses member names
that are not in WEIGHTS (they used to be ignored silently)."""
import py_compile
import shutil
from pathlib import Path

T = Path("/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track")
SRC, DST = T / "submission_v8_code", T / "submission_v9_code"
NEW = Path("/kaggle/working/omni_rt/kfold/v9")
assert not DST.exists(), f"{DST} exists; remove it first"
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("__pycache__"))
for f in ("stage_phi4.py", "stage_tta.py"):
    shutil.copy(NEW / f, DST / f)
shutil.copytree("/home/jovyan/work/Enes/phi_extra/backoff", DST / "vendor_phi" / "backoff", ignore=shutil.ignore_patterns("__pycache__"))


def rep(path, old, new):
    s = path.read_text(encoding="utf-8")
    assert s.count(old) == 1, f"{path.name}: expected exactly one match for {old[:70]!r}, found {s.count(old)}"
    path.write_text(s.replace(old, new), encoding="utf-8")


m = DST / "main.py"
rep(m, '"""Lost in Transcription (Spanish-English) - v7 entrypoint',
    '"""V9: v8 + Whisper on 0.9x/1.1x speed-perturbed audio (2 voters, w8 each) + Phi-4-multimodal Miami fine-tune (w8).\n'
    'Validated (xeval tta_whisper_phi4): dev 0.0755->0.0751, glued long 0.0826->0.0808, Miami half 0.1846->0.1799.\n\n'
    'Lost in Transcription (Spanish-English) - v7 entrypoint')
rep(m, 'SPLIT_OVER_S = {"whisper": 30.0, "qwen3asr": 45.0, "conformer": 60.0, "cohere": 30.0, "parakeet": 30.0}',
    'SPLIT_OVER_S = {"whisper": 30.0, "qwen3asr": 45.0, "conformer": 60.0, "cohere": 30.0, "parakeet": 30.0, "phi4": 30.0}')
rep(m, 'MEMBERS = ("omni", "voxtral", "whisper", "conformer", "qwen3asr", "cohere", "parakeet")',
    'MEMBERS = ("omni", "voxtral", "whisper", "conformer", "qwen3asr", "cohere", "parakeet", "phi4", "whisper_s090", "whisper_s110")')
rep(m, 'log(f"v8 start: {len(rows)} clips")', 'log(f"v9 start: {len(rows)} clips")')
rep(m, '"cohere": ["config.json", "model.safetensors"], "parakeet": ["config.json", "generation_config.json", "model.safetensors"]}',
    '"cohere": ["config.json", "model.safetensors"], "parakeet": ["config.json", "generation_config.json", "model.safetensors"],\n'
    '        "phi4": ["config.json", "model.safetensors.index.json", "modeling_phi4mm.py", "speech_conformer_encoder.py", "processing_phi4mm.py"],\n'
    '        "phi4_delta": ["delta.safetensors"]}')
rep(m, '''if not rows:
    abort("empty test_metadata.csv")''', '''if not (HERE / "vendor_phi" / "backoff" / "__init__.py").is_file():
    abort("vendor_phi/backoff (needed by the Phi-4 remote code) is missing")
if not rows:
    abort("empty test_metadata.csv")''')
rep(m, 'log("pre-flight ok: 7 members + omni delta + vendor_tf + all audio present")',
    'log("pre-flight ok: 8 models (+2 TTA passes) + omni delta + phi4 delta + vendor_tf + vendor_phi + all audio present")')
rep(m, '''        ("parakeet", "stage_parakeet.py", lambda: ["16"], 300)]   # Parakeet Ultra 0.6B TDT, Miami fine-tune''',
    '''        ("parakeet", "stage_parakeet.py", lambda: ["16"], 300),   # Parakeet Ultra 0.6B TDT, Miami fine-tune
        ("phi4", "stage_phi4.py", lambda: ["8"], 600)]            # v9: Phi-4-multimodal + Miami delta, greedy, batch 8''')
rep(m, '''# 6. Vote (repetition guard on every member, then the verified writing rules on the final text)''',
    '''# 6. v9 TTA: Whisper again on 0.9x and 1.1x speed-perturbed audio (resampled copies, cut > 30 s like the original)
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

# 7. Vote (repetition guard on every member, then the verified writing rules on the final text)''')
rep(m, 'log(f"v8 done in {time.time() - T0:.0f}s:', 'log(f"v9 done in {time.time() - T0:.0f}s:')

v = DST / "vote.py"
rep(v, '"omni": 20.0, "whisper": 16.0, "cohere": 16.0}',
    '"omni": 20.0, "whisper": 16.0, "cohere": 16.0,\n'
    '           "whisper_s090": 8.0, "whisper_s110": 8.0, "phi4": 8.0}   # v9: TTA Whisper pair + Phi-4 Miami FT, weights fixed before validation')
rep(v, '''if not systems:
    log("no usable system output")
    sys.exit(1)''', '''if not systems:
    log("no usable system output")
    sys.exit(1)
if os.environ.get("V8_STRICT") == "1" and set(systems) != set(WEIGHTS):   # v9: unknown or missing names are an error
    log(f"STRICT: members {sorted(systems)} != weighted members {sorted(WEIGHTS)} -> no vote")
    sys.exit(1)''')
for f in DST.glob("*.py"):
    py_compile.compile(str(f), doraise=True)
print("submission_v9_code ok:", sorted(p.name for p in DST.iterdir()))
