"""v8 vote with ANY member list: the real submission_v8_code/vote.py functions (collapse, chunked vote for clips > 30 s,
median_extended word-level, anchor fallback, gap-fill, option2), same flow as its __main__, but members and weights come
from the command line so extra voters (TTA passes, a second seed, sampled passes) can be added.
usage: vote_any.py <set dev|T|long> <out_csv> [name=path:weight ...]   (no extra args = the v8 seven)
The v8 seven are always first, in vote.py's WEIGHTS order; extras follow in the order given."""
import csv
import os
import re
import sys
from pathlib import Path

V8 = "/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track/submission_v8_code"
src = open(f"{V8}/vote.py", encoding="utf-8").read()
src = src[:src.index('if __name__ == "__main__" and len(sys.argv) == 2')].replace("Path(__file__).resolve().parent", f"Path('{V8}')")
g = {"__name__": "v8vote"}
exec(compile(src, "vote.py", "exec"), g)

S = "/kaggle/working/omni_rt"; T = "/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track"
K = "/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/miami_kfold"; H = f"{T}/kfold/hyps"
OD = f"{S}/out/stage_delta"; WD = f"{T}/build_v7_val/work"; WL = f"{T}/build_v7_val_long/work"; E = f"{T}/cohere_beam/b4"
DATA = {"dev": f"{S}/dev", "T": f"{K}/miami_Ttest", "long": f"{S}/dev_long"}
BASE = {"dev": dict(parakeet=f"{WD}/parakeet.csv", qwen3asr=f"{WD}/qwen3asr.csv", voxtral=f"{WD}/voxtral.csv", conformer=f"{WD}/conformer.csv", omni=f"{OD}/dev.csv", whisper=f"{WD}/whisper.csv", cohere=f"{E}/dev.csv"),
        "T": dict(parakeet=f"{H}/parakeet_lr5e5s1000_onT.csv", qwen3asr=f"{H}/qwen3asr_trainAB_onT.csv", voxtral=f"{H}/voxtral_trainAB_onT.csv", conformer=f"{H}/conformer_trainAB_onT.csv", omni=f"{OD}/T.csv", whisper=f"{H}/whisperS2_full_onT.csv", cohere=f"{E}/T.csv"),
        "long": dict(parakeet=f"{WL}/parakeet.csv", qwen3asr=f"{WL}/qwen3asr.csv", voxtral=f"{WL}/voxtral.csv", conformer=f"{WL}/conformer.csv", omni=f"{OD}/long.csv", whisper=f"{WL}/whisper.csv", cohere=f"{E}/long.csv")}

sname, out_csv = sys.argv[1], Path(sys.argv[2])
rows = g["read_manifest"](DATA[sname])
for kv in filter(None, os.environ.get("VOTE_W", "").split(",")):   # research override, e.g. VOTE_W=omni:16
    k, v = kv.split(":"); assert k in g["WEIGHTS"], k; g["WEIGHTS"][k] = float(v)
members = [(n, BASE[sname][n], g["WEIGHTS"][n]) for n in g["WEIGHTS"] if n not in os.environ.get("VOTE_EXCLUDE", "").split(",")]
for a in sys.argv[3:]:
    name, rest = a.split("=", 1)
    path, w = rest.rsplit(":", 1)
    members.append((name, path, float(w)))
systems = {}
for name, path, w in members:
    d = g["read_csv"](path)
    nonempty = sum(1 for r in rows if d.get(r["audio_filename"], "").strip())
    if nonempty < 0.9 * len(rows):
        sys.exit(f"member {name} has only {nonempty}/{len(rows)} transcripts ({path})")
    systems[name] = d
names = [m[0] for m in members]
weights = [m[2] for m in members]
collapse = g["collapse"]
lists = [[collapse(re.sub(r"\bo_k\b", "okay", systems[n].get(r["audio_filename"], ""))) for n in names] for r in rows]
idx = [i for i, l in enumerate(lists) if any(x.strip() for x in l)]
jobs, owner = [], []
for i in idx:
    long_clip = float(rows[i].get("file_duration_seconds") or 0) > g["CHUNK_OVER_S"]
    for p in (g["chunk_texts"](lists[i], weights) if long_clip else [lists[i]]):
        jobs.append(p); owner.append(i)
out = g["ensemble_text_lists"](jobs, normalize=False, char_level=False, weights=weights, ensemble_type="median_extended",
                               language=None, max_workers=6, verbose=False)
parts = {}
for i, o in zip(owner, out):
    parts.setdefault(i, []).append(" ".join(o.split()))
final = {rows[i]["audio_filename"]: " ".join(" ".join(v).split()) for i, v in parts.items()}
anchor = systems["omni"]
for r in rows:
    if not final.get(r["audio_filename"], "").strip():
        final[r["audio_filename"]] = collapse(anchor.get(r["audio_filename"], ""))
for i in idx:
    n = rows[i]["audio_filename"]
    final[n], _ = g["gap_fill"](final.get(n, ""), lists[i], weights)
g["write_csv"](out_csv, rows, {k: g["POSTPROC"](v) for k, v in final.items()})
print(f"{sname}: {len(names)} members {list(zip(names, weights))} -> {out_csv}")
