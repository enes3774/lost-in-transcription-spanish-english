"""Check one packaged v8 run: every v8 ingredient must be visibly ACTIVE in the logs/outputs, nothing silently dropped.
usage: verify_v8_run.py <val_dir> <data_dir> <expect_chunked_clips>   -> prints PASS/FAIL per item, exit 1 on any FAIL."""
import csv
import re
import sys
from pathlib import Path

V, DATA, EXP_CHUNK = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
job = (V / "run.log").read_text(encoding="utf-8", errors="replace")
logs = {p.stem: p.read_text(encoding="utf-8", errors="replace") for p in (V / "work" / "logs").glob("*.log")}
rows = list(csv.DictReader(open(DATA / "test_metadata.csv", encoding="utf-8")))
n = len(rows)
n_over45 = sum(1 for r in rows if float(r["file_duration_seconds"]) > 45)
n_over30 = sum(1 for r in rows if float(r["file_duration_seconds"]) > 30)
fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        fails.append(name)


omni = logs.get("omni", "")
m = re.search(r"base weights for (\d+) clips over 45s, delta for (\d+)", omni)
check("1 Omni delta: split base/delta by 45 s", bool(m) and int(m.group(1)) == n_over45 and int(m.group(2)) == n - n_over45,
      m.group(0) if m else "no split line")
m = re.search(r"omni delta merged into (\d+) attention tensors", omni)
check("1 Omni delta: merged into the engine", bool(m) and int(m.group(1)) == 192, m.group(0) if m else "no merge line")
check("1 Omni delta: clips decoded with the delta", ("delta chunk" in omni) == (n - n_over45 > 0))
sub = {r["audio_filename"]: r["transcript"] for r in csv.DictReader(open(V / "submission.csv", encoding="utf-8"))}
raw = {r["audio_filename"]: r["transcript"] for r in csv.DictReader(open(V / "work" / "vote_raw.csv", encoding="utf-8"))}
n_super_raw = sum(len(re.findall(r"(?<![\w'])super(?![\w'])", t, re.I)) for t in raw.values())
n_bare_final = sum(len(re.findall(r"(?<![\w'])super(?![\w'])", t, re.I)) for t in sub.values())
n_super_final = sum(len(re.findall(r"(?<![\w'])súper(?![\w'])", t, re.I)) for t in sub.values())
check("2 súper rule applied to the final text", n_bare_final == 0 and n_super_final >= n_super_raw,
      f"vote had {n_super_raw} 'super', final has {n_super_final} 'súper' and {n_bare_final} bare 'super'")
vote = logs.get("vote", "")
m = re.search(r"gap-fill \(>= 60% of the weight\): (\d+) words inserted", vote)
check("3 gap-fill ran (threshold 60 %)", bool(m), m.group(0) if m else "no gap-fill line")
coh = logs.get("cohere", "")
check("4 Cohere decoded with 4 beams", "cohere beams 4" in coh, "stage log: " + ("'cohere beams 4'" if "cohere beams 4" in coh else "missing"))
m = re.search(r"chunked vote: (\d+) clips over 30 s cut into chunks \((\d+) vote jobs\)", vote)
check("5 chunked vote active on clips > 30 s", bool(m) and int(m.group(1)) == EXP_CHUNK, (m.group(0) if m else "no line") + f" (expected {EXP_CHUNK}, clips > 30 s: {n_over30})")
check("5 vote is word-level, all 10 voters used", "vote over ['parakeet', 'qwen3asr', 'voxtral', 'conformer', 'omni', 'whisper', 'cohere', 'whisper_s090', 'whisper_s110', 'phi4']" in vote)
check("6 strict: pre-flight passed", "pre-flight ok" in job)
members_ok = re.findall(r"member (\w+) ok: (\d+)/(\d+) non-empty", job)
check("6 strict: every member complete", sorted(x[0] for x in members_ok) == sorted(["omni", "voxtral", "whisper", "conformer", "qwen3asr", "cohere", "parakeet", "phi4", "whisper_s090", "whisper_s110"]),
      ", ".join(f"{a} {b}/{c}" for a, b, c in members_ok))
check("6 strict: no ABORT, no dropped member", "ABORT" not in job and "dropped" not in vote)
check("6 strict: no submission-time fallback text in any log", not re.search(r"falling back|greedy for this batch|clips one by one|left empty|restoring", job + "".join(logs.values())))
check("7 vendor_tf.zip unpacked by main.py", "vendor_tf unpacked next to main.py" in job)
check("7 parakeet used the vendored transformers", "(vendored)" in logs.get("parakeet", ""))
nl = job.count("\n")
check("7 job log short enough (platform keeps 500)", nl < 400, f"{nl} lines")
check("final: one row per clip, all non-empty", len(sub) == n and all((sub.get(r["audio_filename"]) or "").strip() for r in rows),
      f"{sum(1 for r in rows if (sub.get(r['audio_filename']) or '').strip())}/{n}")
check("final: 'v9 done' reached", "v9 done" in job)
ph = logs.get("phi4", "")
check("9 Phi-4 Miami delta loaded", "Miami delta: 1137 tensors" in ph, "stage log: " + ("delta line present" if "Miami delta" in ph else "missing"))
for f, tag in (("0.9", "whisper_s090"), ("1.1", "whisper_s110")):
    tl = logs.get(f"tta_{tag}", "")
    check(f"9 TTA x{f} audio built for every clip", f"tta x{f}: {n} clips" in tl, tl.strip().splitlines()[-1][:80] if tl.strip() else "no log")
leak = [l for l in job.splitlines() if any(t and len(t) > 25 and t[:25] in l for t in list(sub.values())[:50])]
check("privacy: no transcript text in the job log", not leak, f"{len(leak)} suspicious lines")
print("VERIFY_" + ("OK" if not fails else "FAILED " + "; ".join(fails)))
sys.exit(1 if fails else 0)
