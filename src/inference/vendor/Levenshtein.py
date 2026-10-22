"""Pure-Python stand-in for the `Levenshtein` C extension (only what asr_ensemble uses):
distance() via Myers' bit-parallel algorithm, editops() via a DP with traceback."""


def distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    m = len(a)
    if m == 0:
        return len(b)
    mask = (1 << m) - 1
    hib = 1 << (m - 1)
    peq = {}
    for i, ch in enumerate(a):
        peq[ch] = peq.get(ch, 0) | (1 << i)
    pv, mv, score = mask, 0, m
    for ch in b:
        eq = peq.get(ch, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = (mv | ~(xh | pv)) & mask
        mh = pv & xh
        if ph & hib:
            score += 1
        elif mh & hib:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = (mh | ~(xv | ph)) & mask
        mv = ph & xv
    return score


def editops(s1, s2):
    """Ops turning s1 into s2 as (op, src_pos, dst_pos), op in replace/insert/delete."""
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        row, prev, c1 = dp[i], dp[i - 1], s1[i - 1]
        for j in range(1, m + 1):
            cost = 0 if c1 == s2[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if s1[i - 1] == s2[j - 1] else 1):
            if s1[i - 1] != s2[j - 1]:
                ops.append(("replace", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("delete", i - 1, j))
            i -= 1
        else:
            ops.append(("insert", i, j - 1))
            j -= 1
    ops.reverse()
    return ops
