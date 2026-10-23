# Lost in Transcription SP-EN: Solution Write-up

Written 2026-10-13, before the private leaderboard was published. Final result: 2nd place, private WER 0.0998 (public 0.0993). Online version: https://claude.ai/code/artifact/1af409f1-07be-4e75-ac0e-6a3c5c8ba709

## Result

Second place on the Mozilla Data Collective "Lost in Transcription: Spanish-English" track with a public WER of 0.0993 (leader 0.0990), reached with a ten-voter ensemble of speech models fine-tuned on the Bangor Miami code-switching corpus and combined by a word-level weighted vote.

The approach in one paragraph: eight different open ASR and audio-LLM models (Qwen3-Omni, Voxtral Small, Whisper large-v3, Cohere Transcribe, Parakeet, Qwen3-ASR, a Conformer-CTC and Phi-4-multimodal) were each fine-tuned on the same two Miami folds, decoded in a strict, no-fallback pipeline, and merged word by word with a weighted median vote (ROVER). Two extra Whisper passes on speed-perturbed audio act as cheap test-time augmentation. A small set of writing rules then maps the Miami transcription style to the organizers' style. Every change was accepted only if it was non-negative on three validation sets (the official dev set, a glued long-clip set and an untouched half of Miami), and no weight or threshold was tuned on the test set.

## Task and metric

The track asks for verbatim transcripts of Spanish-English code-switched voice notes, scored by word error rate after the organizers' normalizer, inside a code-execution sandbox.

| item | value |
| --- | --- |
| Test set | hidden; seen only by the platform |
| Dev set (public) | 155 clips, 36 min, 2 speakers, 4,994 reference words |
| Metric | corpus WER = (S + D + I) / N after the official normalizer |
| What the normalizer keeps | accents, apostrophes, mid-sentence capitals and `...`; it lower-cases the first letter after `. ! ?` and drops commas, periods and `¿ ¡ ; :` |
| Runner | one A100 80 GB, 120-minute job limit, no internet, fixed Python environment (transformers 4.57, vLLM 0.23, no flash-attn) |
| Submission budget | 3 scored submissions per rolling 7 days |
| Package | one zip with `main.py` and all weights; a 63 GB archive ran, a 76 GB one hung and was cancelled |

Two properties of the metric drove the design. The normalizer keeps accents and mid-sentence case, so spelling conventions (`súper`, `quién`, `okay`) count as much as hearing. And it is corpus WER, so long clips carry more weight than their count suggests: a few clips over 30 s hold a large share of the words.

## Datasets

Every model was fine-tuned on one corpus, the Bangor Miami code-switching conversations, split by conversation so that no validation speaker was ever trained on. The competition dev set was used only to measure, never to train.

| dataset | what it is | how we used it |
| --- | --- | --- |
| Bangor Miami (TalkBank BilingBank) | 56 face-to-face Spanish-English conversations recorded in Miami, about 35 h of audio, CHAT transcripts | the only training data. Parsed to 22,290 utterances (24.1 h); CHAT codes stripped, `o_k` mapped to `okay`, fragment and underscore conventions normalised |
| Miami folds A + B | two speaker-disjoint folds by conversation, about 19,600 clips | fine-tuning data for all eight members (10,933 clips after a 2 s minimum and quality filter for the audio-LLMs) |
| Miami fold T | the remaining conversations, held out entirely | 600 clips (`Tsel600`) for checkpoint selection; the untouched 1,447-clip half as the third validation set |
| Miami windows | consecutive same-speaker utterances joined (gap ≤ 5 s) into windows of up to 24 s: 6,951 windows, 21.6 h, median 10 s | training unit for the Whisper family; raw 2.3 s utterances taught models to stop early |
| Competition dev set | 155 clips, 36 min, 2 speakers, 4,994 words | measurement only. Fine-tuning on it could not be checked on any held-out set that contains the dev speakers, so it was rejected as an overfitting risk; it was never used for training |
| Glued long set | 58 clips of 60–220 s built by gluing dev clips together | validation of long-clip handling (the long tail of the test set) |
| Speed-perturbed test audio | the test clips resampled by 0.9× and 1.1× at run time | test-time augmentation for Whisper (two extra voters) |
| CIEMPIESS | 28k Mexican Spanish conversation clips | tried as a 30 % mix on fold A: no change (vote 0.2110 → 0.2108 on Miami, dev 0.1065 → 0.1057), so not used |

Why Miami transfers although it is a different domain (2008–2011 conversations vs 2026 voice notes): zero-shot models translate one language into the other or pick one language per clip, and Miami fine-tuning removes exactly that. Voxtral went from 0.21 to 0.11 dev WER with a LoRA on 6,000 Miami clips, and its share of translation errors from 27 % to 6 %.

What we deliberately did not use: the dev set for training, the test set in any form, more than a third of Miami for selection, and public Spanglish Whisper checkpoints (three were tested as extra voters and all hurt).

## The ten ensemble members

Eight different model families, all fine-tuned on Miami folds A + B, plus two speed-perturbed Whisper passes; no single member is below 0.090 dev WER, the vote reaches 0.075.

| voter | base model | fine-tune | decoding | clips cut above | weight | alone, dev WER |
| --- | --- | --- | --- | --- | --- | --- |
| Omni | Qwen3-Omni-30B-A3B thinker, AWQ int4 experts with bf16 attention | LoRA r16 on attention q/k/v/o, lr 1e-4, 1 epoch; merged into the running vLLM engine at start-up, base weights kept for clips over 45 s | vLLM, greedy, whole clips | never | 20 | 0.0949 |
| Voxtral | Voxtral-Small-24B, FP8 (RedHatAI) | LoRA r16 on attention, served as a vLLM LoRA adapter | vLLM, greedy, whole clips | never | 20 | 0.0951 |
| Whisper | Whisper large-v3 | LoRA r32 on attention and MLP, 2 epochs on 24 s windows, merged in fp32 | HF pipeline, greedy, batch 16 | 30 s | 16 | 0.0907 |
| Whisper 0.9× | same weights | none | same, on audio slowed by 10 % | 30 s | 8 | 0.0901 |
| Whisper 1.1× | same weights | none | same, on audio sped up by 10 % | 30 s | 8 | 0.0907 |
| Cohere | Cohere Transcribe (2B conformer) | full fine-tune, lr 1e-5 bf16, about 40 min | 4 beams, collapse guard | 30 s | 16 | 0.0919 |
| Parakeet | Parakeet Ultra 0.6B TDT | full fine-tune, lr 5e-5, step 1000 | greedy, batch 16 | 30 s | 16 | 0.1039 |
| Qwen3-ASR | Qwen3-ASR-1.7B | LoRA on the thinker, merged | 4 beams | 45 s | 10 | 0.1059 |
| Conformer | NeMo Conformer-CTC large, code-switched es-en | full fine-tune, 6 epochs, ported to plain PyTorch | greedy CTC | 60 s | 8 | 0.2121 |
| Phi-4 | Phi-4-multimodal-instruct | speech LoRA r320 + audio encoder + speech projection (915M parameters, fp32 masters, lr 4e-5, step 1500), shipped as a merged delta | greedy, batch 8 | 30 s | 8 | 0.1334 |

How the set was built: a member earned its slot only by lowering the vote on all three validation sets at a weight fixed before the measurement (8 for a new member, half the parent's weight for each speed copy). Parakeet is the instructive case: it was rejected eight times under the old character-level vote and accepted as soon as the vote became word-level, which is where most of v7's gain came from. The Conformer contributes nothing measurable any more and stayed only because removing it was never positive either.

Why the weak voters still matter: a voter much weaker than the core (Nemotron, VibeVoice, omniASR, Granite, MOSS, canary, Seamless) added noise, but Phi-4 at 0.133 helped because its errors are independent of the others; it is the only member that is not an encoder-decoder ASR system or a Qwen-family audio-LLM.

The Omni fine-tune was the largest single gain of the competition (Miami half 0.1902 → 0.1850, dev 0.0773 → 0.0763). It needed fp32 master weights: a pure-bf16 AdamW run at lr 1e-5 left 90 % of the weights unchanged.

## Combination: the word-level vote

The combiner turned out to be the biggest lever of the whole competition: switching the ROVER vote from character level to word level and adding Parakeet moved dev from 0.0805 to 0.0775 and the untouched Miami half from 0.1776 to 0.1712, more than any single model change.

*(diagram in the online version: v9c pipeline, 10 voters, 4 combination steps; see README.md "Combination".)*

Each member runs as its own subprocess, so GPU memory is released between models; the vote runs once at the end and is the only step that writes the output file.

1. **Repetition guard** on every member: one word at most 4 times in a row, a 2–6-word phrase at most twice. Audio-LLMs occasionally loop ("no no no" × 40); the references keep real stutters, so the cap is loose.
2. **Chunking for long clips.** The vote library (`asr_ensemble`, `median_extended`) builds candidates on the edit path between two members, so on a 300-word clip it cannot take one member's words in one sentence and another's in the next. Clips over 30 s are cut at anchor words (a pivot word of the heaviest member that every member has at the aligned position), at least 30 words apart, voted chunk by chunk and joined. Glued long set 0.0874 → 0.0825, dev unchanged, and the vote on long clips went from 34 min to 15 s.
3. **Weighted word-level median vote.** Weights 20 / 20 / 16 / 16 / 16 / 10 / 8 / 8 / 8 / 8 (see the member table). Word level matters for code-switching: character merging can fuse a Spanish and an English word into a word that is neither.
4. **Majority gap-fill.** After the vote, every member is aligned to it; where members holding at least 60 % of the weight insert the same phrase in the same gap, it is inserted. Never negative; worth about 0.0006 on long clips.

Why we stopped working on the combiner: a decomposition on the Miami half showed that of the vote's misses, 0.069 WER were words no member had, 0.052 were words only a minority had, and only 0.005 were words a majority had. A text-only combiner can recover at most that last 0.005. Plain `median`, minimum-Bayes-risk selection, an LLM rewrite of the member texts and an Omni-based refiner (v6, submitted) were all measured and were not better.

## Writing rules on the final transcript

Nine small rules applied once, to the voted text, map Miami's transcription conventions to the organizers'; their effect was measured on dev when each rule was added and recorded in the docstrings of `src/inference/spelling_rules.py`: five spelling rules, v5 five-way vote 0.0993 → 0.0897; casing guard 0.0897 → 0.0853; filler rule a further −0.0008 (the first seven together about −0.015); underscores neutral in the vote; `súper` 0.0773 → 0.0765 on the v8 vote; `quién` 0.0751 → 0.0747 on the v9 vote. The v4 → v5 leaderboard step (0.1190 → 0.1033) bundled the rules with member changes, so it is not an isolated measurement of them.

| rule | why | measured effect |
| --- | --- | --- |
| `gonna` → `going to`, `wanna` → `want to`, `gotta` → `got to`, `kinda` → `kind of`, `ok` → `okay` | Miami writes `gonna` 298 : 66, the dev references `going to` 18 : 2; the fine-tuned members learned Miami's spelling | board −0.0096 (dev predicted −0.0096) |
| no `. ! ?` directly before `I`, `I'm`, `I'll`, `I've`, `I'd` | the scorer lower-cases the first letter after sentence punctuation; our members punctuate heavily, the annotators rarely do, so `well. I think` was scored as `i think` | board −0.0062 together with the next rule (dev predicted −0.0058) |
| `ah`, `er` → `uh` | the references never write `ah` or `er` and use `uh` | included above |
| underscores: `Curious_George` → `Curious George`, `T_V` → `TV` | Miami joins names with underscores; the references contain no underscore at all | dev −0.0010 |
| `super` → `súper` | the references write the Spanish intensifier with its accent 5 : 1 (dev) and 15 : 3 (long); no member ever writes it | dev −0.0008, long −0.0007 |
| `quien` → `quién` (added in the final submission) | the references never write it unaccented (dev 8 : 0, Miami 4 : 0) | dev −0.0004, long −0.0004, Miami 0 |

Rules that looked plausible and were rejected by measurement: deleting hesitations (the annotators keep about half of `um`/`uh`/`eh`, so every deletion is a coin flip), de-duplicating stutters (the references keep 293 of them on the Miami half), expanding contractions, writing digits, forcing one casing per name, and predicting a capital after a question start (the references put `¿` on only one question in three).

## Long clips

Three members break on clips over 30 s, so every clip over a per-member limit is cut at pauses before decoding and the piece texts are joined back in order.

- **The cut.** `seg_long.py` (numpy only) looks inside a 12–28 s window for the lowest-energy 0.3 s and cuts there, so pieces end at pauses, not mid-word. The unchanged member script decodes the pieces; `mapping.json` joins them.
- **Per-member limits** come from a stress test on the glued long set, not from one global threshold: Whisper and the three small ASR models at 30 s, Qwen3-ASR at 45 s, the Conformer at 60 s; Omni and Voxtral are never cut (clean to 274 s), and the Omni fine-tune switches to base weights above 45 s because the LoRA only saw short clips and loops on long ones.
- **Measured on the 58 glued clips (median 93 s), member WER whole → split:** Qwen3-ASR 0.354 → 0.128, Whisper 0.250 → 0.114, Cohere 0.118 → 0.104, Conformer crash → 0.225 (its position table ends at 200 s, which would have dropped the whole member).
- **At ensemble level** the gain is smaller than the member gains suggest, because Omni and Voxtral never needed cutting: 0.0989 → 0.0911 on the glued set, about 0.001–0.002 on the board. The chunked vote (previous section) added the rest: 0.0874 → 0.0825.

A last check before the final submission: scoring the glued set case-insensitively, v9 loses only 0.0005 on long audio against short; the remaining official gap is a casing artefact of gluing (every glued clip has only one sentence-initial lower-casing), so nothing was left to recover there.

## Validation protocol

A change shipped only if it was non-negative on three sets at once, with every weight and threshold fixed before the measurement; this rule is what kept dev gains from turning into board losses after v4.

| set | size | what it guards against |
| --- | --- | --- |
| competition dev | 155 clips, 2 speakers | the organizers' audio and writing style |
| glued long set | 58 clips, 60–220 s, built from dev | long-clip regressions (the long tail) |
| Miami half (fold T) | 1,447 clips, speaker-disjoint from every fine-tune | overfitting to the two dev speakers; judged mainly on its 10–30 s bucket, the dev-like one |

What we learned about the sets:

- **Noise floor.** Re-decoding one member moves the vote by about 0.0005; vLLM runs are not bit-reproducible (about 0.001 on dev). Differences under 0.001 on dev were treated as ties, which is why per-clip better/worse counts were read alongside the WER.
- **Transfer.** Dev gains reached the board at roughly a quarter to a third of their size (v5 → v7: dev −0.0064, board −0.0017). The dev-to-board ratio of 1.22 held only for changes with no tuned settings.
- **Dev fine-tuning was not shipped.** A speaker-disjoint experiment gave only −0.0026 on dev, and the result could not be checked on any held-out set that contains the dev speakers, so it was rejected as an overfitting risk. The dev set stayed a measurement set.
- **Selecting on Miami does not transfer either.** Every Miami-leaning change made after v8 (full fine-tunes, Miami-style Whisper, Voxtral Mini, omniASR) gained on short Miami clips and lost on dev; the board tracks dev, not Miami.

The last submission is the honest end of this protocol: v9c beat v8 by 0.0004–0.0008 on dev and 0.004 on Miami, and scored the same 0.0993 on the board. A predicted gain of about 0.0002 is within the board's rounding.

## What did not work

Most of the three weeks went into things that were measured and rejected; the list is the useful part for anyone repeating this, because each item looked reasonable and cost real GPU time.

| idea | result | why it died |
| --- | --- | --- |
| **Extra model families as voters** (Nemotron 3.5 ASR, VibeVoice, Granite-speech NAR, MOSS, canary, Gemma-4 audio, SeamlessM4T-v2, w2v-BERT CTC, omniASR-LLM 3B/7B v1 and v2, Voxtral Mini 3B, Whisper turbo, three public Spanglish Whispers), each Miami fine-tuned and tested at a fixed weight | every one neutral or worse on at least one set | the vote is saturated; a voter much weaker than the core adds noise, and all share the same Miami training data so their errors are correlated |
| **Second copies of a member** (Cohere seed 2 and 3, Whisper seed and label-format siblings, Phi-4 seed 2, Qwen3-ASR speed pair, a second Whisper speed pair) | Miami-only gains, dev flat or worse | more of the same votes; seed-to-seed spread (0.005 on dev) is larger than the effect |
| **Omni refiner** (v6): the audio-LLM re-transcribes each clip given the audio and all member texts | dev 0.0843 → 0.0825, board 0.1033 → 0.1043 | did not transfer; it also truncated long transcripts |
| **Dev-set fine-tuning**, member and vote level | dev −0.0026 out-of-fold; board worse | two speakers, one annotator: what it learns is style the rules already deliver |
| **Smarter combiners**: plain `median`, MBR selection, CTC beam + Miami trigram, LLM rewrite, per-clip oracle selection | all worse than the word-level vote | string selection loses to word splicing; the oracle gap is in words no member or only a minority had |
| **Full fine-tunes instead of LoRA** (Qwen3-ASR, Whisper large-v3, Voxtral v2 with fp32 masters) | members better on Miami, vote worse on dev and long | they learn Miami's style (more words, more cross-talk agreement) and lose the organizer-style audio |
| **More Miami** (2-epoch Omni, checkpoint soups, concat-mix long-clip training, retrain on all of Miami) | no gain, or runaway loops on long clips | the fine-tunes were already at the plateau; concat training taught Omni to keep talking |
| **Other data** (CIEMPIESS mix, synthetic TTS for unseen words) | flat | unseen words are 3–5 % of the references and mostly names that cannot be known in advance |
| **Weight and threshold tuning** (Omni 16–32, Parakeet 12–16, gap-fill 0.5, TTA at other weights) | within noise or worse | and any tuned setting breaks the transfer rule |
| **Text rules beyond the nine** (drop fillers, de-duplicate stutters, expand contractions, digits, one casing per name, capital after `¿`) | coin flips | the annotators are inconsistent on exactly these |

The pattern across all of it: after v8 the ensemble's errors were 0.030 words no member had, 0.026 words a minority had and 0.001 the vote's own mistakes (dev). Nothing text-side or vote-side can reach the first two, and every acoustic addition we could make shared the Miami training set with the voters already there.

## Engineering

The package is a 60.6 GB zip (74.6 GB unpacked) that runs in about 35 of the runner's 120 minutes, with no fallback path anywhere: if anything fails, the job writes no file and the platform refunds the slot.

- **Strict, no-fallback `main.py`.** A pre-flight checks every model file and every audio file before any model loads. Each member must return at least 90 % non-empty clips, every stage has a timeout, the final file must have one row per clip. Any miss calls `abort()`, which deletes any partial output and exits non-zero. Failure injection (missing file, corrupt delta, garbage weights) was tested end to end before the first strict submission.
- **One subprocess per member**, so vLLM and the HF models never share GPU memory, and each stage's full output goes to its own log file; the job log receives only our own summary lines. The platform keeps the first 500 lines of the job log, and vLLM alone prints hundreds, which had hidden every stage after Voxtral in v4–v6.
- **Vendored dependencies.** The runtime has transformers 4.57, vLLM 0.23 and no flash-attn. Parakeet needs a newer transformers, shipped as a nested `vendor_tf.zip` unpacked by `main.py` (3,482 loose files had flooded the log listing); the vote library and Levenshtein are pure Python; the Conformer was ported from NeMo to plain PyTorch (132 of 155 clips bit-identical).
- **Quantization to fit the size limit.** Omni runs as a hybrid checkpoint, AWQ int4 experts with bf16 attention so the LoRA delta merges exactly; Voxtral as FP8 with the LoRA served as a vLLM adapter. The fine-tunes ship as deltas over public base weights.
- **Platform size limit.** 63.4 GB stored archives ran; a 76.4 GB one hung for 48 minutes and was cancelled, as a 117 GB one had on day two. The final package merged the Phi-4 delta into its base shards, zero-filled Phi-4's unused image tower, and was deflated instead of stored: 60.6 GB, built in 25 minutes by five parallel zip processes stitched without recompression.
- **Upload.** 64 MB blocks to Azure blob storage with 8 workers at 14–17 MB/s, about 70–85 minutes per submission; the finalize request must go out while the browser session is still valid, which cost one re-upload on the last day.
- **Runtime.** v8: 21 minutes for the full test set. v9c: Omni 3 min, Voxtral 5, Whisper 3 × 2 min, Phi-4 3 min, the rest under a minute each, vote 30 s.

## Submission history

*(chart in the online version: public WER v4 0.1190, v5 0.1033, v6 0.1043, v7 0.1016, v8 0.0993, v9c 0.0993.)*

The two steps that mattered were v5 (fair Miami-trained members plus the writing rules) and v7–v8 (word-level vote, Parakeet and Cohere, the Omni fine-tune); v6's refiner and v9c's Phi-4 plus speed copies did not move the board.

| version | date | what changed |
| --- | --- | --- |
| v4 | Sep 15 | 5 members (Omni, Voxtral, Qwen3-ASR, Conformer, Whisper), members selected on dev, character-level vote; 60.6 GB after quantizing Omni and Voxtral to fit (the 117 GB bf16 package had hung) |
| v5 | Sep 20 | fair members (no checkpoint chosen on dev), the five spelling rules, casing guard, filler rule, pause-splitting of long clips |
| v6 | Sep 23 | v5 plus the Omni refiner stage; scored worse, not kept |
| v7 | Sep 26 | Cohere Transcribe as 6th voter, underscore rule, **word-level** vote, Parakeet as 7th voter; per-stage logs |
| v8 | Sep 28 | Omni Miami LoRA delta, `súper` rule, gap-fill, Cohere beam 4, chunked vote, strict no-fallback mode |
| v9c | Oct 1 | Phi-4 Miami fine-tune and the Whisper speed pair as voters 8–10, `quién` rule; deflated 60.6 GB package after the 76 GB stored one hung |



## Lessons

1. **Treat the combiner as a variable.** The vote's character-level setting was hard-coded and never questioned for two weeks; switching it to word level was worth more than any model. Parakeet was rejected eight times under the wrong setting.
2. **One in-domain corpus, used carefully, beats more data.** Miami fixed the translation and language-choice errors every zero-shot model made. Extra corpora, synthetic speech and bigger fine-tunes added nothing once that was done.
3. **Three validation sets and fixed weights.** Dev alone (two speakers) would have shipped the refiner, which then lost on the board (v6), and the dev fine-tune, which no held-out set could check. Deciding the weight before looking at the result kept the comparisons honest.
4. **Read the transcripts.** The error read-out showed that 43 % of the remaining dev errors were annotator conventions (hesitations, cut-off words, capitals) that no model can learn and no rule can fix; it stopped us chasing them.
5. **Measure the noise floor first.** Re-decoding one member moves the vote by 0.0005, and the board shows four decimals; most of the last week's candidates were inside that band.
6. **Make failure cheap.** Strict mode with no fallback means a broken run costs nothing (the slot is refunded) and a scored run is known to be the real pipeline.
7. **Know the platform limits before the last day.** The archive size ceiling, the 500-line log cap, the one-active-job rule and the session timeout each cost at least one upload. The parallel deflate build and the log-line counting trick came out of that pressure and would have been worth having from the start.

To reproduce: the package is `main.py` plus the stage scripts, `vote.py` with the vendored `asr_ensemble`, `seg_long.py`, `spelling_rules.py`, the public base models, and the Miami fine-tune deltas; the experiment log `SPEN_EXPERIMENTS_ON_V7_2026-09-27.md` holds every measurement behind the tables above.
