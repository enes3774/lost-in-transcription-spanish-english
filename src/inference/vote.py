"""Weighted word-level median vote (v8: + majority gap-fill) over the available stage outputs (vendored asr_ensemble), v5/v6.
usage: vote.py <data_dir> <out_csv> name=csv ...

v5 changes vs v4:
  * Whisper joins the vote at weight 16 (order below is the measured one; order matters slightly).
  * every member passes through a repetition guard first (runaway loops like "no no no" x40).
  * the FINAL text gets the test-verified writing rules (V5_POSTPROC = option2 | spelling | none; default option2).
"""
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
sys.path.insert(0, str(HERE))
from common import log, read_csv, read_manifest, write_csv  # noqa: E402
from asr_ensemble import ensemble_text_lists  # noqa: E402
from spelling_rules import normalize_spelling, option2  # noqa: E402

WEIGHTS = {"parakeet": 16.0, "qwen3asr": 10.0, "voxtral": 20.0, "conformer": 8.0, "omni": 20.0, "whisper": 16.0, "cohere": 16.0,
           "whisper_s090": 8.0, "whisper_s110": 8.0, "phi4": 8.0}   # v9: TTA Whisper pair + Phi-4 Miami FT, weights fixed before validation   # v8: word-level merging + parakeet@16; dev 0.0805->0.0773, Miami half 0.1776->0.1712 (real vote.py)
POSTPROC = {"option2": option2, "spelling": normalize_spelling, "none": lambda t: t}[os.environ.get("V5_POSTPROC", "option2")]


def collapse(text, max_uni=4, max_ngram=2):
    """Cap runaway repetition: a single word at 4 in a row, a 2-6 word phrase at 2 in a row."""
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


# ---------------------------------------------------------------------------------------------------
# v8: chunked vote for long clips (CHUNK_VOTE_PATCH). median_extended only builds candidates on the edit path
# between TWO members, so on a 300-word clip it cannot take member X's words in one sentence and member Y's in
# another. Cut every member at anchors (a pivot word, heaviest member, that every non-empty member has at the
# aligned position), >= 30 pivot words per chunk (~ one dev clip), vote each chunk, join. Lossless cut. Only for
# clips over 30 s. Measured (real harness): long set 0.0874 -> 0.0824, dev 0.0759 -> 0.0755, Miami half unchanged.
# ---------------------------------------------------------------------------------------------------
CHUNK_WORDS = int(os.environ.get("V8_CHUNK_WORDS", "30"))
CHUNK_OVER_S = 30.0


def _matches(ref, hyp):
    """word Levenshtein on keys; {ref index: hyp index} for exact matches on the optimal path."""
    n, m = len(ref), len(hyp)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        ri, Di, Dp = ref[i - 1], D[i], D[i - 1]
        for j in range(1, m + 1):
            Di[j] = min(Dp[j - 1] + (0 if ri == hyp[j - 1] else 1), Dp[j] + 1, Di[j - 1] + 1)
    i, j, out = n, m, {}
    while i > 0 and j > 0:
        if D[i][j] == D[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] == hyp[j - 1]:
                out[i - 1] = j - 1
            i -= 1; j -= 1
        elif D[i][j] == D[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def chunk_texts(texts, weights, ch=CHUNK_WORDS):
    """-> list of chunks (each a list of member texts in the same order); [texts] when there is no safe cut."""
    toks = [(t or "").split() for t in texts]
    live = [k for k, t in enumerate(toks) if t]
    if len(live) < 2 or ch <= 0:
        return [texts]
    piv = max(live, key=lambda k: (weights[k], -k))
    pk = [_key(w) for w in toks[piv]]
    maps = {k: _matches(pk, [_key(w) for w in toks[k]]) for k in live if k != piv}
    cuts, start = [], 0
    for i in range(len(pk)):
        if i - start >= ch and len(pk) - i >= ch // 2 and all(i in maps[k] for k in maps):
            cuts.append(i); start = i
    if not cuts:
        return [texts]
    bounds, chunks = [0] + cuts + [None], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        row = []
        for k in range(len(texts)):
            if k not in live:
                row.append(""); continue
            ia = 0 if a == 0 else (a if k == piv else maps[k][a])
            ib = len(toks[k]) if b is None else (b if k == piv else maps[k][b])
            row.append(" ".join(toks[k][ia:ib]))
        chunks.append(row)
    if any(" ".join(" ".join(c[k] for c in chunks).split()) != " ".join(toks[k]) for k in range(len(texts))):
        return [texts]                                                          # never drop or duplicate a word
    return chunks


if __name__ == "__main__" and len(sys.argv) == 2 and sys.argv[1] == "--selftest":
    v, k = gap_fill("we went to the store", ["we went to the big store", "we went to the big store", "we went to the store"], [20, 16, 8])
    assert v == "we went to the big store" and k == 1, (v, k)
    v, k = gap_fill("we went to the store", ["we went to the big store", "we went to the store", "we went to the store"], [20, 16, 8])
    assert v == "we went to the store" and k == 0, (v, k)
    v, k = gap_fill("okay let's go", ["okay. let's go", "okay, let's go now", "okay let's go now."], [10, 10, 10])
    assert v.rstrip(".") == "okay let's go now" and k == 1, (v, k)
    v, k = gap_fill("", ["a b", "a b"], [1, 1]); assert v == "" and k == 0
    c = chunk_texts([" ".join(f"w{i}" for i in range(70)), " ".join(f"w{i}" for i in range(70) if i != 40)], [20, 16], ch=30)
    assert len(c) == 2 and " ".join(x[0] for x in c).split() == [f"w{i}" for i in range(70)], c
    assert chunk_texts(["a b c", ""], [20, 16]) == [["a b c", ""]]
    print("gap_fill self-test passed")
    sys.exit(0)

data_dir, out_csv = Path(sys.argv[1]), Path(sys.argv[2])
rows = read_manifest(data_dir)
systems = {}
for arg in sys.argv[3:]:
    name, path = arg.split("=", 1)
    if not os.path.exists(path):
        log(f"{name}: no output file -> dropped")
        if os.environ.get("V8_STRICT") == "1":                      # STRICT_V8: never vote with a member missing
            sys.exit(1)
        continue
    d = read_csv(path)
    nonempty = sum(1 for r in rows if d.get(r["audio_filename"], "").strip())
    if nonempty >= 0.9 * len(rows):
        systems[name] = d
    log(f"{name}: {nonempty}/{len(rows)} transcripts -> {'used' if name in systems else 'dropped'}")
    if os.environ.get("V8_STRICT") == "1" and name not in systems:   # STRICT_V8
        log(f"STRICT: member {name} unusable -> no vote")
        sys.exit(1)
if not systems:
    log("no usable system output")
    sys.exit(1)
if os.environ.get("V8_STRICT") == "1" and set(systems) != set(WEIGHTS):   # v9: unknown or missing names are an error
    log(f"STRICT: members {sorted(systems)} != weighted members {sorted(WEIGHTS)} -> no vote")
    sys.exit(1)
names = [n for n in WEIGHTS if n in systems]
weights = [WEIGHTS[n] for n in names]
t = time.time()
if len(names) == 1:
    final = {k: collapse(v) for k, v in systems[names[0]].items()}
else:
    lists = [[collapse(re.sub(r"\bo_k\b", "okay", systems[n].get(r["audio_filename"], ""))) for n in names] for r in rows]
    idx = [i for i, l in enumerate(lists) if any(x.strip() for x in l)]      # clips where every member is empty stay empty
    jobs, owner, nchunked = [], [], 0                                         # v8: long clips are voted chunk by chunk
    for i in idx:
        try:
            long_clip = float(rows[i].get("file_duration_seconds") or 0) > CHUNK_OVER_S
        except ValueError:
            long_clip = False
        parts = chunk_texts(lists[i], weights) if long_clip else [lists[i]]
        nchunked += len(parts) > 1
        for p in parts:
            jobs.append(p); owner.append(i)
    log(f"chunked vote: {nchunked} clips over {CHUNK_OVER_S:.0f} s cut into chunks ({len(jobs)} vote jobs)")
    out = ensemble_text_lists(jobs, normalize=False, char_level=False, weights=weights,
                              ensemble_type="median_extended", language=None,
                              max_workers=max(2, min(16, (os.cpu_count() or 4) - 2)), verbose=False)
    parts_out = {}
    for i, o in zip(owner, out):
        parts_out.setdefault(i, []).append(" ".join(o.split()))
    final = {rows[i]["audio_filename"]: " ".join(" ".join(v).split()) for i, v in parts_out.items()}
    anchor = systems.get("omni", systems[names[0]])
    for r in rows:                                                            # a clip the vote left empty falls back to the anchor
        if not final.get(r["audio_filename"], "").strip():
            final[r["audio_filename"]] = collapse(anchor.get(r["audio_filename"], ""))
    if GAPFILL > 0:                                                           # v8: majority gap-fill
        added = 0
        for i in idx:
            n = rows[i]["audio_filename"]
            final[n], k = gap_fill(final.get(n, ""), lists[i], weights)
            added += k
        log(f"gap-fill (>= {GAPFILL:.0%} of the weight): {added} words inserted")
if os.environ.get("V6_VOTE_RAW"):                                            # v6: the refiner wants the vote before the rules
    write_csv(Path(os.environ["V6_VOTE_RAW"]), rows, final)
final = {k: POSTPROC(v) for k, v in final.items()}
write_csv(out_csv, rows, final)
log(f"vote over {names} done in {time.time() - t:.0f}s, postproc={os.environ.get('V5_POSTPROC', 'option2')}")
