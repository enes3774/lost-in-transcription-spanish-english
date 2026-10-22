"""Score every k-fold hypothesis file on its held-out Miami group, plus the 4-member vote per group."""
import csv
import re
import sys

sys.path.insert(0, "/kaggle/working/omni_rt")
sys.path.insert(0, "/kaggle/working/omni_rt/vendor")
import jiwer  # noqa: E402
from asr_ensemble import ensemble_text_lists  # noqa: E402
from score import normalize_text  # noqa: E402

K = "/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/miami_kfold/"
H = "/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track/kfold/hyps/"
fix = lambda t: re.sub(r"\bo_k\b", "okay", t or "")
FILES = {"omni": "omni_base_on{g}.csv", "voxtral": "voxtral_train{o}_on{g}.csv",
         "qwen3asr": "qwen3asr_train{o}_on{g}.csv", "conformer": "conformer_train{o}_on{g}.csv"}
W = {"qwen3asr": 10, "voxtral": 20, "conformer": 8, "omni": 20}


def load(p):
    return {r["audio_filename"]: fix(r["transcript"]) for r in csv.DictReader(open(p, encoding="utf-8"))}


print(f"{'model':10s} {'on A (fold B)':>22s} {'on B (fold A)':>22s} {'on T (fold A)':>22s}")
res = {}
for g, o in (("A", "B"), ("B", "A"), ("T", "A")):
    refs = load(K + f"miami_{g}/refs.csv")
    names = [n for n in sorted(refs) if normalize_text(refs[n]).strip()]
    R = [normalize_text(refs[n]) for n in names]
    hyps = {m: load(H + p.format(g=g, o=o)) for m, p in FILES.items()}
    for m, h in hyps.items():
        res[(m, g)] = (jiwer.wer(R, [normalize_text(h.get(n, "")) for n in names]),
                       sum(1 for n in names if not h.get(n, "").strip()))
    order = list(W)
    vote = ensemble_text_lists([[hyps[m].get(n, "") for m in order] for n in names], normalize=False, char_level=True,
                               weights=[W[m] for m in order], ensemble_type="median_extended", language=None,
                               max_workers=6, verbose=False)
    res[("VOTE", g)] = (jiwer.wer(R, [normalize_text(v) for v in vote]), 0)
    res[("N", g)] = (len(names), sum(len(r.split()) for r in R))
for m in list(FILES) + ["VOTE"]:
    print(f"{m:10s} " + " ".join(f"{res[(m, g)][0]:>13.4f} ({res[(m, g)][1]:4d} empty)" for g in "ABT"))
print("clips / ref words: " + " | ".join(f"{g}: {res[('N', g)][0]} / {res[('N', g)][1]}" for g in "ABT"))
