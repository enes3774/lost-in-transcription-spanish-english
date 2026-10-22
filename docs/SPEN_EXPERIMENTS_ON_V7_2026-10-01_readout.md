# What the near-miss candidates actually change (2026-10-01, `kfold/read_diffs.py`, report `$T/xexp/read_diffs.txt`)

Per candidate, v9 vs v9 + candidate, clips changed / better / worse and net word errors:

| candidate | dev | long | Miami half |
|---|---|---|---|
| Whisper seed 1 | 12 clips, 5/6, +1 | 31, 13/8, -1 | 123, 40/40, -8 (sub +15, del -11, ins -12) |
| Phi-4 seed 2 | 5, 1/3, +5 | 26, 8/11, +8 | 108, 42/32, -19 (ins -14) |
| Whisper turbo | 14, 5/8, +8 | 28, 13/10, -2 | 102, 38/32, -12 (del -12) |
| Voxtral Mini | 12, 6/4, +1 | 33, 15/12, +2 | 118, 32/48, +27 |
| VibeVoice retrain | 15, 8/5, 0 | 34, 10/20, +22 | 118, 50/39, -21 (ins -20) |
| Cohere speed pair | 25, 11/8, +2 | 46, 16/19, +6 | 192, 78/55, -28 (del -20) |
| Qwen3-ASR speed pair | 13, 7/4, -2 | 32, 12/12, -1 | 127, 33/51, +22 |
| Whisper full FT (added) | 18, 9/8, +3 | 31, 11/9, +4 | 117, 40/40, -2 |

Reading the clips:
1. **Dev changes are one-word coin flips.** Each candidate touches 5-25 of 155 dev clips, almost all by one word in
   either direction: they're/we're, get/got, salgo/salvo, miates/niates, "chile en nogada" vs "chile nogada", a filler
   (eh/um) added or dropped, Nogada/nogada casing. Nothing systematic is fixed; the net dev number is the sum of these flips.
   The one real dev win seen (Phi-4 seed 2: "Nahuatl" instead of "what") is outweighed by a 4-word regression on clear
   speech in the same run ("es like the only thing" -> "es que me dicen").
2. **The Miami "gains" are mostly fewer inserted words on very short, garbled clips.** The biggest wins are 1-3 s clips
   where v9 itself is wrong about everything ("yes kids" -> v9 "look at look at me"; "como qué" -> v9 "sí se puede";
   "anyways there will be more ok" -> v9 "yes yes let's not let's not"), and the error-type deltas are dominated by
   ins/del drops. The SAME clips show up as wins for different candidates (yes kids, como qué, the Patty/María clip):
   these are unstable clips where any extra voter tips the vote, not something the candidate hears better. Dev-like data
   has few such clips.
3. **Losses on long are real substitutions** (VibeVoice +22: solo/sólo, su/sus, ahí/hay, Siki/siki), i.e. spelling and
   function-word choices that v9 had right.
Conclusion: none of the near-misses has a gain that can be explained as better recognition of dev-like speech; the
rejections stand. v9 stays.
