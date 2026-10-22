"""Create submission_v8_code from submission_v7_code:
  1. stage_omni.py  -> delta-aware stage (base weights over OMNI_DELTA_MAX_S, A+B LoRA delta merged for the rest)
  2. spelling_rules.py -> option2 also writes 'super' as 'súper' (organizer refs 5:1 / 15:3)
  3. vote.py -> majority gap-fill post-pass (phrase a >=60 %-weight majority inserts in the same gap is inserted)
  4. main.py -> vendor_tf.zip unpacked before the stages; seg_long prints routed to a stage log; v8 labels
Runs the rules self-test and a gap-fill unit test at the end."""
import py_compile
import shutil
import subprocess
from pathlib import Path

T = Path("/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track")
SRC, DST = T / "submission_v7_code", T / "submission_v8_code"
if DST.exists():
    shutil.rmtree(DST)
shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.bak", "*.char_bak"))

# 1. delta-aware Omni stage (built and validated from v7_stage)
shutil.copy("/kaggle/working/omni_rt/v7_stage/stage_omni.py", DST / "stage_omni.py")

# 2. súper
p = DST / "spelling_rules.py"; s = p.read_text(encoding="utf-8")
old = '''def option2(text):
    """Everything: five spelling rules + casing guard + filler spelling + underscores. Apply once, to the FINAL transcript."""
    return fix_underscores(normalize_fillers(protect_i(normalize_spelling(text))))
'''
new = '''_SUPER = re.compile(r"(?<![\\w'’])super(?![\\w'’])", re.IGNORECASE)


def fix_super(text):
    """v8: the organizer references write the Spanish intensifier with its accent (dev súper 5 : super 1, long set
    15 : 3) while every member emits 'super'. Dev vote 0.0773 -> 0.0765, long set 0.0891 -> 0.0884, Miami half 0."""
    if not text or "super" not in text.lower():
        return text
    return _SUPER.sub(lambda m: "Súper" if m.group(0)[0].isupper() else "súper", text)


def option2(text):
    """Everything: five spelling rules + casing guard + filler spelling + underscores + súper. Apply once, to the FINAL transcript."""
    return fix_super(fix_underscores(normalize_fillers(protect_i(normalize_spelling(text)))))
'''
assert old in s; s = s.replace(old, new, 1)
s = s.replace('''    assert option2("well. I'm gonna go ok. ah Curious_George") == "well I'm going to go okay. uh Curious George"
''', '''    assert option2("well. I'm gonna go ok. ah Curious_George") == "well I'm going to go okay. uh Curious George"
    assert fix_super("está super dark, Super bien, superman supermarket") == "está súper dark, Súper bien, superman supermarket"
''', 1)
p.write_text(s, encoding="utf-8")

# 3. gap-fill in vote.py
p = DST / "vote.py"; s = p.read_text(encoding="utf-8")
helper = '''

# ---------------------------------------------------------------------------------------------------
# v8: majority gap-fill. The word-level median vote drops words that a weighted majority of members has at
# the same position (Miami half 0.005 WER, long set 0.007). Align every member to the vote; where members
# holding >= V8_GAPFILL of the weight insert the SAME phrase in the SAME gap, insert it (heaviest member's
# spelling). Measured (real vote.py): Miami half 0.1902 -> 0.1899, dev 0, long 0.0891 -> 0.0885. Pure Python.
# ---------------------------------------------------------------------------------------------------
_PUNCT = '.,?!¿¡;:"“”«»'
GAPFILL = float(os.environ.get("V8_GAPFILL", "0.6"))


def _key(tok):
    return tok.strip(_PUNCT).lower() or "_"


def _inserts(ref, hyp):
    """Levenshtein-align hyp to ref (lists of keys); return {gap index in ref: [hyp indices inserted there]}."""
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i
    for j in range(1, m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        ri, Di, Dp = ref[i - 1], D[i], D[i - 1]
        for j in range(1, m + 1):
            Di[j] = min(Dp[j - 1] + (0 if ri == hyp[j - 1] else 1), Dp[j] + 1, Di[j - 1] + 1)
    i, j, ins = n, m, {}
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            i -= 1; j -= 1
        elif j > 0 and D[i][j] == D[i][j - 1] + 1:
            ins.setdefault(i, []).append(j - 1); j -= 1
        else:
            i -= 1
    return {g: sorted(v) for g, v in ins.items()}


def gap_fill(vote_text, member_texts, member_weights, thr=GAPFILL, max_phrase=4):
    vt = (vote_text or "").split()
    if not vt or thr <= 0:
        return vote_text, 0
    vk = [_key(t) for t in vt]
    total = float(sum(member_weights)) or 1.0
    gaps = {}
    for text, w in zip(member_texts, member_weights):
        mt = (text or "").split()
        if not mt:
            continue
        mk = [_key(t) for t in mt]
        for g, idxs in _inserts(vk, mk).items():
            phrase = tuple(mk[j] for j in idxs)
            ent = gaps.setdefault(g, {}).setdefault(phrase, [0.0, -1.0, None])
            ent[0] += w
            if w > ent[1]:
                ent[1], ent[2] = w, [mt[j] for j in idxs]
    out, added = [], 0
    for i in range(len(vt) + 1):
        if i in gaps:
            phrase, (wsum, _, toks) = max(gaps[i].items(), key=lambda kv: kv[1][0])
            if wsum / total >= thr and len(phrase) <= max_phrase:
                out.extend(t.strip(_PUNCT) if k < len(toks) - 1 else t for k, t in enumerate(toks))
                added += len(toks)
        if i < len(vt):
            out.append(vt[i])
    return " ".join(out), added


if __name__ == "__main__" and len(sys.argv) == 2 and sys.argv[1] == "--selftest":
    v, k = gap_fill("we went to the store", ["we went to the big store", "we went to the big store", "we went to the store"], [20, 16, 8])
    assert v == "we went to the big store" and k == 1, (v, k)
    v, k = gap_fill("we went to the store", ["we went to the big store", "we went to the store", "we went to the store"], [20, 16, 8])
    assert v == "we went to the store" and k == 0, (v, k)
    v, k = gap_fill("okay let's go", ["okay. let's go", "okay, let's go now", "okay let's go now."], [10, 10, 10])
    assert v.rstrip(".") == "okay let's go now" and k == 1, (v, k)
    v, k = gap_fill("", ["a b", "a b"], [1, 1]); assert v == "" and k == 0
    print("gap_fill self-test passed")
    sys.exit(0)

'''
anchor = "\n\ndata_dir, out_csv = Path(sys.argv[1]), Path(sys.argv[2])\n"
assert anchor in s; s = s.replace(anchor, helper + "data_dir, out_csv = Path(sys.argv[1]), Path(sys.argv[2])\n", 1)
old = '''    anchor = systems.get("omni", systems[names[0]])
    for r in rows:                                                            # a clip the vote left empty falls back to the anchor
        if not final.get(r["audio_filename"], "").strip():
            final[r["audio_filename"]] = collapse(anchor.get(r["audio_filename"], ""))
'''
new = old + '''    if GAPFILL > 0:                                                           # v8: majority gap-fill
        added = 0
        for i in idx:
            n = rows[i]["audio_filename"]
            final[n], k = gap_fill(final.get(n, ""), lists[i], weights)
            added += k
        log(f"gap-fill (>= {GAPFILL:.0%} of the weight): {added} words inserted")
'''
assert old in s; s = s.replace(old, new, 1)
s = s.replace('"""Weighted char-level median vote', '"""Weighted word-level median vote (v8: + majority gap-fill)', 1)
p.write_text(s, encoding="utf-8")

# 4. main.py: nested vendor zip, quiet splitting, v8 labels
p = DST / "main.py"; s = p.read_text(encoding="utf-8")
old = 'log(f"v7 start: {len(rows)} clips")\n'
new = '''log(f"v8 start: {len(rows)} clips")
_vz = HERE / "vendor_tf.zip"                      # v8: the parakeet stage's libraries ship as ONE nested zip (the runner's
if _vz.is_file() and not (HERE / "vendor_tf").is_dir():   # unzip listing of 3,482 loose files flooded the 500-line job log)
    try:
        import zipfile as _zipfile
        with _zipfile.ZipFile(_vz) as _z:
            _z.extractall(HERE / "vendor_tf")
        log(f"vendor_tf unpacked next to main.py: {sum(1 for _p in (HERE / 'vendor_tf').rglob('*') if _p.is_file())} files")
    except Exception as _e:
        log(f"vendor_tf unpack failed ({type(_e).__name__}); parakeet stage will use the runner's transformers")
'''
assert old in s; s = s.replace(old, new, 1)
old = '''    d = WORK / f"seg_{name}"
    try:
        seg_long.split(DATA, d, SPLIT_OVER_S[name], MAX_PIECE_S, MIN_PIECE_S)
        return d
'''
new = '''    d = WORK / f"seg_{name}"
    try:
        LOGS.mkdir(parents=True, exist_ok=True)
        with open(LOGS / f"seg_{name}.log", "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            seg_long.split(DATA, d, SPLIT_OVER_S[name], MAX_PIECE_S, MIN_PIECE_S)   # its summary lines stay out of the job log
        return d
'''
assert old in s; s = s.replace(old, new, 1)
old = '''    try:
        seg_long.merge(stage_data, pieces_csv, final_csv)
'''
new = '''    try:
        with open(LOGS / f"seg_{name}.log", "a", encoding="utf-8") as f, contextlib.redirect_stdout(f):
            seg_long.merge(stage_data, pieces_csv, final_csv)
'''
assert old in s; s = s.replace(old, new, 1)
assert "import shutil" in s
s = s.replace("import shutil\n", "import contextlib\nimport shutil\n", 1)
s = s.replace('log(f"v7 done in', 'log(f"v8 done in', 1)
assert "v8 done in" in s
p.write_text(s, encoding="utf-8")

for f in DST.glob("*.py"):
    py_compile.compile(str(f), doraise=True)
print("compiled:", sorted(f.name for f in DST.glob("*.py")))
PY = "/kaggle/working/omni_rt/venv/bin/python"
print(subprocess.run([PY, str(DST / "spelling_rules.py")], capture_output=True, text=True).stdout.strip() or "RULES SELF-TEST FAILED")
r = subprocess.run([PY, str(DST / "vote.py"), "--selftest"], capture_output=True, text=True)
print(r.stdout.strip() or ("VOTE SELF-TEST FAILED\n" + r.stderr[-800:]))
