"""Score extra-voter candidates against the v8 seven with the real v8 vote (vote_any.py): dev, long, Miami half total and
by length (the 5-30 s buckets are the dev-like ones). Weights were fixed BEFORE any result: a TTA pass or a sampled pass
gets half its member's weight (the pair together = the member), a second Cohere seed gets Cohere's weight (16).
Only candidates whose output files all exist are run. usage: xeval.py [candidate ...]  (default: all ready ones)"""
import csv
import os
import subprocess
import sys

sys.path.insert(0, "/kaggle/working/omni_rt")
import jiwer  # noqa: E402
from score import normalize_text  # noqa: E402

S = "/kaggle/working/omni_rt"; T = "/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track"
K = "/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/miami_kfold"; X = f"{T}/xexp"; PY = f"{S}/venv/bin/python"
W = {"omni": 20, "voxtral": 20, "whisper": 16, "cohere": 16, "parakeet": 16}
tta = lambda m: [(f"{m}_s{f}", f"{X}/{m}_s{f}", W[m] / 2) for f in ("090", "110")]
CANDS = {"base": []}
for m in W:
    CANDS[f"tta_{m}"] = tta(m)
CANDS["tta_small3"] = tta("cohere") + tta("parakeet") + tta("whisper")
CANDS["tta_all5"] = sum((tta(m) for m in W), [])
CANDS["cohere_seed2"] = [("cohere2", f"{X}/cohere_seed2", 16)]
CANDS["cohere_concat"] = [("cohere_c", f"{X}/cohere_concat_s3", 16)]
CANDS["cohere_seed3"] = [("cohere3", f"{X}/cohere_seed3", 16)]
CANDS["cohere_seed3_concat"] = [("cohere3", f"{X}/cohere_seed3", 16), ("cohere_c", f"{X}/cohere_concat_s3", 16)]
CANDS["tta_whisper_seed3"] = tta("whisper") + [("cohere3", f"{X}/cohere_seed3", 16)]
CANDS["omni_samp"] = [("omni_samp1", f"{X}/omni_samp1", 10), ("omni_samp2", f"{X}/omni_samp2", 10)]
# night of 2026-09-28: new 8th members at a fixed weight 8 (the conformer's), 16 as the secondary check
for m in ("canary_ft", "granite_nar", "phi4", "moss", "granite_nar_ft", "phi4_ft", "moss_ft"):
    CANDS[m] = [(m, f"{X}/{m}", 8)]
    CANDS[m + "_w16"] = [(m, f"{X}/{m}", 16)]
CANDS["tta_whisper_phi4"] = tta("whisper") + [("phi4_ft", f"{X}/phi4_ft", 8)]
CANDS["v9_moss"] = CANDS["tta_whisper_phi4"] + [("moss_ft", f"{X}/moss_ft", 8)]
CANDS["v9_concat16"] = CANDS["tta_whisper_phi4"] + [("cohere_c", f"{X}/cohere_concat_s3", 16)]
CANDS["v9_swapc"] = CANDS["tta_whisper_phi4"] + [("cohere_c", f"{X}/cohere_concat_s3", 16)]   # run with VOTE_EXCLUDE=cohere
CANDS["swapc"] = [("cohere_c", f"{X}/cohere_concat_s3", 16)]   # run with VOTE_EXCLUDE=cohere
CANDS["v9_seamless"] = CANDS["tta_whisper_phi4"] + [("seamless_ft", f"{X}/seamless_ft", 8)]
CANDS["v9_w2vbert"] = CANDS["tta_whisper_phi4"] + [("w2vbert_ft", f"{X}/w2vbert_ft", 8)]
CANDS["v9_phi4tta"] = CANDS["tta_whisper_phi4"] + [("phi4_ft_s090", f"{X}/phi4_ft_s090", 4), ("phi4_ft_s110", f"{X}/phi4_ft_s110", 4)]
CANDS["v9_vibevoice_last"] = CANDS["tta_whisper_phi4"] + [("vibevoice_last", f"{X}/vibevoice_last", 8)]
CANDS["v9_nemotron_last"] = CANDS["tta_whisper_phi4"] + [("nemotron_last", f"{X}/nemotron_last", 8)]
CANDS["v9_oasr7_ft"] = CANDS["tta_whisper_phi4"] + [("oasr7_ft", f"{X}/oasr7_ft", 8)]
CANDS["v9_oasr3_ft"] = CANDS["tta_whisper_phi4"] + [("oasr3_ft", f"{X}/oasr3_ft", 8)]
CANDS["v9_ttacohere"] = CANDS["tta_whisper_phi4"] + tta("cohere")
CANDS["v9_tta2_w8"] = CANDS["tta_whisper_phi4"] + [("whisper_s095", f"{X}/whisper_s095", 8), ("whisper_s105", f"{X}/whisper_s105", 8)]
CANDS["v9_tta2_w4"] = CANDS["tta_whisper_phi4"] + [("whisper_s095", f"{X}/whisper_s095", 4), ("whisper_s105", f"{X}/whisper_s105", 4)]
CANDS["v9_wfrag"] = CANDS["tta_whisper_phi4"] + [("wsib_frag_s2", f"{X}/wsib_frag_s2", 8)]
CANDS["v9_ws1"] = CANDS["tta_whisper_phi4"] + [("wsib_s1", f"{X}/wsib_s1", 8)]
CANDS["v9_wfull_add"] = CANDS["tta_whisper_phi4"] + [("wfull", f"{X}/wfull", 8)]
CANDS["v9_wfull_swap"] = CANDS["tta_whisper_phi4"] + [("wfull", f"{X}/wfull", 16)]   # run with VOTE_EXCLUDE=whisper
CANDS["v9_hfx_small"] = CANDS["tta_whisper_phi4"] + [("hfx_small", f"{X}/hfx_small_spanglish", 8)]
CANDS["v9_hfx_enes"] = CANDS["tta_whisper_phi4"] + [("hfx_enes", f"{X}/hfx_lora_en_es_merged", 8)]
CANDS["v9_hfx_esen"] = CANDS["tta_whisper_phi4"] + [("hfx_esen", f"{X}/hfx_lora_es_en_merged", 8)]
CANDS["v9_voxmini"] = CANDS["tta_whisper_phi4"] + [("voxmini_ft", f"{X}/voxmini_ft", 8)]
CANDS["v9_oasr7v2_ft"] = CANDS["tta_whisper_phi4"] + [("oasr7v2_ft", f"{X}/oasr7v2_ft", 8)]
CANDS["v9_qtta"] = CANDS["tta_whisper_phi4"] + [("qwen3asr_s090", f"{X}/qwen3asr_s090", 5), ("qwen3asr_s110", f"{X}/qwen3asr_s110", 5)]
CANDS["v9_wturbo"] = CANDS["tta_whisper_phi4"] + [("wturbo", f"{X}/wturbo", 8)]
CANDS["v9_phi4s2"] = CANDS["tta_whisper_phi4"] + [("phi4_ft_seed2", f"{X}/phi4_ft_seed2", 8)]
CANDS["v9_oasr7v2lr_ft"] = CANDS["tta_whisper_phi4"] + [("oasr7v2lr_ft", f"{X}/oasr7v2lr_ft", 8)]
CANDS["wswap_full"] = [("wfull", f"{X}/wfull", 16), ("wfull_s090", f"{X}/wfull_s090", 8), ("wfull_s110", f"{X}/wfull_s110", 8), ("phi4_ft", f"{X}/phi4_ft", 8)]   # run with VOTE_EXCLUDE=whisper
REFS = {"dev": f"{S}/dev/refs.csv", "T": f"{K}/miami_Ttest/refs_frag2.csv", "long": f"{S}/dev_long/refs.csv"}
load = lambda p: {r["audio_filename"]: r["transcript"] or "" for r in csv.DictReader(open(p, encoding="utf-8"))}
dur = {r["audio_filename"]: float(r["file_duration_seconds"]) for r in csv.DictReader(open(f"{K}/miami_Ttest/test_metadata.csv"))}
R = {s: load(p) for s, p in REFS.items()}


def wer(s, H, names=None):
    names = [n for n in (names or R[s]) if normalize_text(R[s][n]).strip()]
    return jiwer.wer([normalize_text(R[s][n]) for n in names], [normalize_text(H.get(n, "")) for n in names])


bucket = lambda lo, hi: [n for n in R["T"] if lo <= dur.get(n, 0) < hi]
want = sys.argv[1:] or list(CANDS)
print(f"{'candidate':14s} {'dev':>7s} {'long':>7s} {'T all':>7s} {'T 0-5':>7s} {'T 5-10':>7s} {'T 10-30':>7s}")
for c in want:
    extra = CANDS[c]
    if not all(os.path.exists(f"{d}/{s}.csv") for _, d, _ in extra for s in ("dev", "T", "long")):
        print(f"{c:14s} (outputs not ready)")
        continue
    row = []
    for s in ("dev", "long", "T"):
        out = f"{X}/votes/{c}_{s}.csv"
        os.makedirs(os.path.dirname(out), exist_ok=True)
        r = subprocess.run([PY, f"{S}/kfold/vote_any.py", s, out] + [f"{n}={d}/{s}.csv:{w}" for n, d, w in extra], capture_output=True, text=True)
        if r.returncode:
            print(c, s, "VOTE FAILED", r.stderr[-300:]); break
        row.append(load(out))
    else:
        d, l, t = row
        print(f"{c:14s} {wer('dev', d):7.4f} {wer('long', l):7.4f} {wer('T', t):7.4f} {wer('T', t, bucket(0, 5)):7.4f} "
              f"{wer('T', t, bucket(5, 10)):7.4f} {wer('T', t, bucket(10, 30)):7.4f}", flush=True)
