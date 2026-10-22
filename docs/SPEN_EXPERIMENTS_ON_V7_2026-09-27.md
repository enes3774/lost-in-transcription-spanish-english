# Spanish-English: experiments on top of the latest submission (v7)

Baseline = **v7** (id-2745, board 0.1016, rank #4): 7 members (Omni hybrid, Voxtral, Whisper, Conformer, Qwen3-ASR, Cohere, Parakeet),
word-level `median_extended` vote, option2 writing rules, pause-splitting above 30 s (Parakeet/Whisper/Cohere), 45 s (Qwen3-ASR), 60 s (Conformer).

All numbers below come from the **real** `vote.py` harness (`$T/real_vote/`) and `score.py`, on three sets:

| set | what it is | v7 vote WER |
|---|---|---|
| dev | 155 organizer clips, 2 speakers | 0.0773 |
| Miami half | 1,447 untouched Miami clips (speaker-disjoint from every fine-tune) | 0.1902 |
| long | 58 glued 120 s clips built from dev | 0.0891 |

Rule for shipping: a change goes in only if it is non-negative on **all three**, is a single separate change with its own logic,
and needs no setting tuned on our sets.

---

## 1. Ship candidates (each measured alone and together)

| change | dev | Miami half | long | status |
|---|---|---|---|---|
| **Omni A+B LoRA delta**, base weights for clips > 45 s | 0.0773 → 0.0763 | 0.1902 → 0.1850 | 0.0891 → 0.0894 (noise band) | **in** |
| **súper** rule (`super` → `súper`, organizer refs 5:1 dev, 15:3 long, members never write it) | −0.0008 | 0 | −0.0007 | **in** |
| **majority gap-fill** after the vote (insert a phrase that ≥ 60 % of the weight puts in the same gap) | 0 words inserted | 0 (5 at 0.5 → −0.0002) | 15 words, −0.0006 | **in** (never negative) |
| **v8 = all three together** (v8 `vote.py`, delta-stage Omni outputs) | **0.0755** | **0.1846** | **0.0880** | candidate for the experimental slot |

Plus runtime fixes with no score effect: vendored libraries as one nested `vendor_tf.zip` unpacked by `main.py` (the loose
3,482-file listing flooded the 500-line job log), `seg_long` summaries routed to a stage log, refiner kept out.

### Omni delta details
- Trained on the A+B halves only (`kfold/miamiAB_names.txt`, 10,933 clips after the trainer's 2 s / quality filters, zero overlap
  with the Miami half), LoRA r16 on attention q/k/v/o, lr 1e-4, 1 epoch, accumulation 8, 1,367 steps, 2.6 h on GPU 2.
- Member WER: Miami half 0.245 → **0.187** (best member now), dev 0.116 → 0.112, **120 s clips 0.107 → 0.117 (worse)**: Omni never
  splits long audio and the fine-tune saw only short clips. Splitting long clips hurts Omni itself (base 0.107 → 0.127, delta
  0.117 → 0.121), so the fix is per clip: decode clips over 45 s first with base weights, then merge the delta into the running
  vLLM engine (the refiner's `collective_rpc` mechanism, needs the in-process engine core) and decode the rest. Merge failure
  falls back to base weights. Validated on the exact hybrid checkpoint v7 ships: 1,662 clips in 303 s, merge in 6 s.
- Checkpoint choice: step 1000 looped on one dev clip (306 words for 29); the final delta is the one to ship.
- Contribution on the Miami half vs the other members (remove-one): Cohere 0.0046, Parakeet 0.0039, **Omni delta 0.0053**.

---

## 2. Closed as dead (measured, do not retry)

| experiment | result | why dead |
|---|---|---|
| plain `median` instead of `median_extended` (7 members) | dev 0.0791, Miami 0.1911, long 0.0937 | worse on all three (the earlier 6-member library number did not survive the real harness) |
| minimum-Bayes-risk selection (pick the member transcript closest to all others) | Miami 0.1720 vs 0.1631 vote | string selection loses to word splicing; with the vote as a candidate it just converges to the vote |
| stutter dedupe (drop immediate word repeats) | worse everywhere | the references keep repetitions (293 on the Miami half, 45 on dev) |
| filler rules (uh/um/eh) | no rule possible | refs keep them (dev: uh 15, um 9, eh 13); we over-produce by ~40 %; deleting would remove ~35 right to fix 21 wrong |
| CTC prefix beam + Miami trigram for the conformer (`kfold/ctc_beam.py`) | member 0.363 → 0.343 (Miami), 0.220 → 0.207 (dev); **vote** dev 0.0785, Miami 0.1901, long 0.0895 | helps the member, not the ensemble (same as on the other track) |
| Parakeet retrained on clips up to 40 s (bs 2) | member dev 0.111 vs 0.104, Miami 0.224 vs 0.204; vote dev 0.0769, Miami 0.1930, long 0.0897 | worse on all three; the 20 s cap was not the problem |
| Whisper beam 5 | member 0.2497 vs 0.2496; vote 0.1901 vs 0.1902 | identical |
| retrain every member on all of Miami (A+B+T) | A (9.2k) → A+B (19.6k clips) moved the vote 0.1695 → 0.1648 on the Miami half; adding T (2.6k) ≈ 0.001 there, ≈ 0.0003 board | not worth losing the only clean validation set |
| Nemotron 3.5 ASR 0.6B (zero-shot dev) | auto 0.258, es-US 0.291, en-US 0.480 | twice Parakeet's zero-shot (0.133); per-utterance language prompt fights code-switching |
| LLM-style rewrite from the member texts (blind test, 10 clips) | mine 51 errors, vote 50, oracle 44 | reproduces the vote; the residual is cross-talk, pseudonymised names and reference quirks |
| CIEMPIESS mix (28k Mexican Spanish conversation clips, 30 % of each batch; fold-A Voxtral / Qwen3-ASR / Conformer, 2026-09-17) | vote Miami 0.2110 → 0.2108, dev 0.1065 → 0.1057; Voxtral member worse on Miami (0.248 → 0.259) | flat; out-of-domain Spanish adds nothing the members lack |
| more data / synthetic TTS for unseen words (`kfold/oov_check.py`) | see section 3: unseen words are 3–5 % of the refs and the members already write 63–86 % of them | the fixable part is names we cannot know in advance; ≤ 0.004 dev even if every one were fixed |
| label-format rules: "...", "&", mid-sentence capitals (`kfold/label_quirks.py`, `kfold/case_forms.py`) | case-only errors dev 0.0052 / long 0.0104 / Miami 0.0077; "..."-only dev 0.0018; "&" never in dev refs; always-"I" rule = identical (our raw output never writes "i"); stripping our sentence punctuation dev −0.0002, long −0.0002, Miami +0.0001 | case errors are sentence-boundary mismatches (the scorer lowercases after every period) and annotator inconsistency ("christmas", "english" lowercase but "Zumba", "Target" capitalised); "..." marks cut-off words that are often real words ("la...", "that...") |
| cross-talk / oracle gap | oracle 0.1101 vs vote 0.1631 on the Miami half; of the vote's misses, 0.069 no member had, 0.052 a minority had, **0.005 a majority had** | text-only combiner fixes cap at ~0.005; consensus insertions (0.037) are the other speaker's words the refs omit |

---

## 3. Diagnostics that changed decisions

- **Parakeet's transfer**: its per-length gains are similar across length buckets, so the length mix does **not** explain the 27 % transfer.
  Parakeet never decodes > 30 s (pause-split) and helps on the 120 s set. Gating it out of long clips would cost that gain.
- **What Parakeet adds** (6-way vs 7-way, per changed clip): dev 17 better / 8 worse, Miami 122 / 86; gains are short Spanish
  function words and clause starts (ya, voy, darme, allá, y) and rescued very short clips; content words break even (61 vs 62).
  → acoustic, transferable in kind, ~0.001–0.002 in size.
- **What Cohere adds**: dev 25 / 13, Miami 126 / 82; **content words two to one** (19:8 dev, 81:41 Miami), correct-language choice
  ("el restaurante" → "the restaurant"), agreement forms. → the Cohere-class gain; the Omni delta looks like this class.
- **Sentence-initial capitals after "¿"**: organizer refs capitalise after ¿ 100 % (17 dev / 51 long) and the scorer keeps that
  capital; our output has no ¿ at all (~0.0014 dev lost). Predicting question starts is not a rule.
- **Unseen words** (`kfold/oov_check.py`; unseen = not in the Miami A+B fine-tune text, 13.6k word types). The members are
  pretrained on far more data, so they do write words they never saw in fine-tuning:

  | set | unseen share of ref words | vote gets unseen right | any member writes it | seen words right | unseen misses (WER pts) |
  |---|---|---|---|---|---|
  | dev | 5.1 % | 80 % | 86 % | 95 % | 0.0104 of 0.0755 |
  | Miami half | 3.3 % | 53 % | 63 % | 88 % | 0.0153 of 0.1832 |

  Dev's missed unseen words are about half reference artefacts (cut-off words like "tha...", digits, nonstandard spellings
  like "dispués", "cémo") and half real vocabulary (Nahuatl, Zumba, Marshall's, Pedialit, Chapultepec). The Miami half's
  are mostly conversation-specific names (Selena 25, Clare 12, Annie, Iria, Rupert) plus pseudonyms (0 % hit, unfixable).
  CIEMPIESS covers only 17–34 % of them. TTS would need the test set's names in advance, so it cannot target them.
- **Vote runtime** grows fast with clip length: 2,022 s for the 58 glued 120 s clips vs 45 s for dev.

---

## 4. Pending (running on GPU 2, results before the package is decided)

| experiment | logic | status |
|---|---|---|
| Omni A+B LoRA, **2 epochs** (`miami_r16_AB_e2`, 2,734 steps) | the 1-epoch run improved from step 1000 to the end as the LR decayed | **done, no gain**: member dev 0.1141 / Miami 0.1888 vs 1-epoch 0.1115 / 0.1867; vote dev 0.0765 / Miami 0.1832 vs 0.0761 / 0.1838 (steps 2000/2500 same picture) = noise. 1-epoch delta stays |
| **Voxtral v2** (fp32 LoRA master weights, lr 5e-5, accum 16, otherwise identical) | v1 validation swung ±0.03 between checkpoints, A+B run collapsed to 0.406 at step 1500, more data made it worse | training on GPU 2 since 18:03 (`kfold/logs/train_voxtral_AB_v2.log`) |
| **Omni checkpoint average** ("model soup", from the Pasketti winners): soupAB = 1-epoch final + 2-epoch final, soupE2 = 2-epoch steps 2000/2500/final (`kfold/soup_make.py`, exact average of the B@A products) | averaging runs that end in different places usually beats each of them; no training, same merge path as the shipped delta | **done, no gain**: member dev 0.1109 / 0.1111 (1-epoch 0.1115) but Miami 0.1905 / 0.1903 (1-epoch 0.1867); vote dev 0.0759 both (0.0761), Miami 0.1844 / 0.1842 (0.1838) = worse on the Miami half. 1-epoch delta stays |
| **Nemotron 3.5 ASR, Miami A+B fine-tune** (`kfold/nemotron_train.py`, lr 5e-5, bs 4 x accum 4, clips <= 20 s, early stop on Tsel600) | zero-shot errors look like translation / language choice, which Miami fine-tuning fixed for Voxtral and Cohere; zero-shot Tsel600 0.408 (shipped Parakeet after FT 0.241) | **done, dead**: Tsel600 0.408 -> 0.328 (250) -> 0.320 -> **0.308 (750)** -> 0.313, plateau far above Parakeet 0.241; member dev 0.170 / Miami 0.289 (Parakeet 0.104 / 0.204). Vote at a fixed weight 8: dev 0.0765 -> 0.0767, Miami 0.1845 -> 0.1841 as 8th member; replacing the conformer dev 0.0771 = noise or worse. Training stopped at step ~1100. Its zero-shot mistakes were not only language choice: it is simply a weaker recogniser of this speech |
| weight sensitivity with the delta Omni (omni 16–32 × parakeet 12–16) | Omni went from 2nd-worst to best member at an unchanged weight | **done, flat**: omni 16 vs 20 = −0.001 Miami / −0.0004 dev; omni ≥ 24 worse on both (Miami 0.1856 → 0.1878). Weights stay at 20/16 (no tuned settings) |

Replacement rule for the pending ones: the 2-epoch delta replaces the 1-epoch delta only if it wins on all three sets;
Voxtral v2 replaces v1 only if it wins on all three sets.

---

## 5. Slot plan (user decisions)

- Two real slots left before the deadline (~2026-10-03); both built and validated as final packages.
- Slot A: v8 (delta + súper + gap-fill + runtime fixes). Slot B: v8 plus whatever pending item wins on all three sets.
- Súper is not worth a single-rule leaderboard A/B: an estimated ≈ 0.0005, below the pipeline's own
  vLLM run-to-run noise (~0.001).

Paths: code `submission_v8_code` (builder `omni_rt/kfold/make_v8_code.py`), delta `qwen3_omni/lora_runs/miami_r16_AB/miami.pt`,
delta-aware stage `omni_rt/v7_stage/stage_omni.py`, harness `real_vote/`, analysis scripts `omni_rt/kfold/*.py`.

---

## 6. Night of 2026-09-27 (GPUs 1 and 2 only; GPU 0 left to the user)

**Morning summary: nothing beat v8 on all three sets. v8 (1-epoch Omni delta + súper + gap-fill) remains the package.**
The only Miami-half gains (Qwen3-ASR full FT / soup, -0.0012 to -0.0014) come with dev/long losses on organizer-style audio.
Noise floor measured tonight: re-decoding one member moves the vote ~0.0005 on each set.

Every candidate is compared against v8 (dev 0.0755 / Miami half 0.1846 / long 0.0880) with the same code on all three sets.
The v8 vote itself moves by ~0.0002 between runs (Miami 0.1846 vs 0.1849 on a rerun), so differences that small are ties.

| run | GPU | logic | status |
|---|---|---|---|
| Voxtral v2 (fp32 LoRA, lr 5e-5, accum 16) | 2 | calmer trainer | **done, NOT adopted**. Tsel600 still swings (0.259, 0.254, 0.287, **0.213 @1000**, 0.269, 0.242). Member vs tonight's v1 re-decode: dev 0.1137 vs 0.1117, Miami 0.2208 vs 0.2362 (deletions 0.053 -> 0.038, subs equal), long 0.1137 vs 0.1176; no empties, 8 vs 5 looping clips, no extra language flips. **v8 vote: dev 0.0775 vs 0.0751, Miami 0.1855 vs 0.1841, long 0.0896 vs 0.0886 = worse on all three** (vote insertions +39 on Miami, +7 dev): v2 writes more words and agrees with the other members' cross-talk insertions. Current Voxtral stays |
| current Voxtral re-decoded with the same eval code | 1 | clean baseline for v2 (same stage, same vLLM noise) | **done**: member dev 0.1117 (package 0.1125) / Miami 0.2365 (old hyps 0.2397) / long 0.1176 (0.1163); v8 vote dev **0.0751** / Miami **0.1841** / long **0.0886** (v8 with the old Voxtral outputs 0.0755 / 0.1846 / 0.0880). **Re-decoding one member moves the vote ~0.0005: that is the noise floor for tonight's comparisons**; v2 is compared to this re-decode |
| **Qwen3-ASR-1.7B full fine-tune** (`kfold/kf_qwen3asr_full.py`, fp32 master, lr 1e-5, 2 epochs, 1,366 steps) | 1 | the Pasketti winners' recipe | **done, NOT adopted**. Tsel600 0.285 -> 0.258 -> 0.253 -> 0.245 -> 0.228 -> 0.224 -> **0.219 final** (LoRA 0.243). Member vs the current model decoded the same way (beam 4, split > 45 s): dev **0.1294 vs 0.1205**, Miami 0.1912 vs 0.1991, long **0.1329 vs 0.1284** — it learned Miami's style and got worse on organizer-style audio (subs +0.007, ins +0.007 on dev; no runaways, no filler/case shift). v8 vote: dev 0.0761 vs 0.0757, Miami 0.1837 vs 0.1849, long 0.0884 vs 0.0880. **Soup of steps 1000/1250/1366**: member dev 0.1270 / Miami 0.1899 / long 0.1325; vote dev 0.0757 (=), Miami **0.1835** (-0.0014), long 0.0887 (+0.0007). Only the Miami half improves; the test set is organizer audio like dev, so a member that is worse on dev and long is a risk. Current Qwen3-ASR stays |
| **Omni concat-mix LoRA** (`omni_lora_train_concat.py`, +4 h = 170 glued 45–110 s Miami clips) | 2 | the delta is worse than base on long clips because it only saw short ones | **done, dead**. Member (delta on whole clips): dev 0.1117 (shipped 0.1115), Miami 0.2144 (0.1867), long 0.1139 (shipped 0.1166, **base 0.1070 still better**). The Miami loss is 4 runaway clips (+105 to +149 words each; insertions 0.045 -> 0.074): long-clip training taught it to keep talking. v8 vote: dev 0.0755 (=), Miami 0.1852 (+0.0005), long 0.0882 with the delta on whole 120 s clips (v8 0.0880). The 45 s base-weight switch stays |
| CPU: remove-one on v8 + base Omni as extra voter (weight 10) | CPU | does any member hurt now that Omni is fine-tuned? | **done** (dev / Miami half; v8 0.0755 / 0.1849 on this run): no_whisper 0.0801 / 0.1913, no_cohere 0.0777 / 0.1896, no_omni 0.0795 / 0.1882, no_parakeet 0.0781 / 0.1867 = all needed; no_voxtral 0.0803 / 0.1842 and no_qwen3asr 0.0761 / 0.1838 = needed on dev, slightly harmful on Miami; **no_conformer 0.0755 / 0.1848, long 0.0881 (v8 0.0880) = contributes nothing on any set now**; dropping it is a runtime saving, not a score change; plus base Omni 0.0757 / 0.1870 = dead |

## 7. v8 package change (2026-09-28)

- **Safety net removed** (user decision: a failed run returns the slot). A failed Omni delta merge now aborts the whole job with
  no submission file instead of finishing with base weights (`kfold/patch_v8_nofallback.py`). Tested end to end with main.py on
  3 clips and a corrupt delta: stage exit 3, main.py exit 1, no submission.csv.
- What the delta fixes on dev (vote with base Omni vs delta Omni): 30 clips change, 13 better / 8 worse, net -5 errors (~0.001).
  Fixes are grammar and function words (speak -> speaking, había -> habían, vemos -> debemos, a missing "if"); losses are
  lowercased names/places (La Cañada, Target, Zumba -> sumba) and homophones (know -> no, tender -> atender). On the Miami half
  net -90 errors (fillers 47:20, function 91:56, content 89:48) - partly Miami transcription conventions that will not transfer.
- **VibeVoice (Miami LoRA outputs from 2026-09-23) re-checked in the v8 vote** (`kfold/vibe_votes.sh`, weight fixed at 8):
  member dev 0.1512 / Miami 0.2530; v8 ref 0.0755 / 0.1848, conformer -> VibeVoice 0.0771 / 0.1855, VibeVoice as 8th
  0.0763 / 0.1858 = worse on both. Dead.
- **Hojo-ASR-Multi-V1** (Qwen3-4B decoder, Apache 2.0, own `hojo-asr` package): model card lists English only as future
  support, no fine-tuning recipe. Not a fit for code-switched Spanish-English; not tested.
- Board context 2026-09-28: leader 0.099, us 0.1016 (gap 0.0026).
- **Language of v8's errors** (`kfold/lang_errors.py`, word language from the Miami word-level langid lexicon): English and
  Spanish words are equally error-prone (dev 0.060 vs 0.061 errors per reference word; long 0.074 vs 0.073; Miami half
  0.163 vs 0.178). English errors lean to insertions (dev 39 vs 21), Spanish to substitutions (89 vs 68). Over-represented:
  fillers (1 % of words, ~10 % of errors) and names/unknown words (11 % of words, 20 % of errors). Wrong-language
  substitutions are 8.5 % of dev errors. No language-specific weakness to target.
- **Cohere beam 4 vs greedy** (`kfold/stage_cohere_beam.py` wraps model.generate: transcribe() hard-codes num_beams=1;
  GPU 2, package split > 30 s). Member: dev 0.1115 -> 0.1091, Miami 0.1909 -> 0.1858, long 0.1182 -> 0.1142 (better on
  all three). v8 vote: dev 0.0757 -> 0.0757, Miami 0.1849 -> 0.1849 - exactly net zero (dev 12 clips change, 4 better /
  4 worse, 55 -> 55 errors; Miami 110 clips, 37 / 33, 447 -> 447); **long vote 0.0880 -> 0.0874**. Non-negative on all three -> ADDED to v8 (`kfold/patch_v8_cohere_beam.py`: stage arg beams, main.py passes 4 when > 1200 s remain else 1, a failing beam batch retries greedily; package stage tested on dev: 4 beams, 155/155, 0.1097 whole clips).
- **Filler spellings** (`kfold/filler_spellings.py`): Miami A+B labels write uh 215 / ah 510 / er 144 / um 505; dev refs
  uh 15 / ah 0 / er 0 / um 9 / eh 13; untrained models write uh like the organizer (base Omni dev uh 7 ah 0; Parakeet Ultra
  zero-shot uh 9 ah 0); Miami-fine-tuned members switch to ah/er (Voxtral uh 1 / ah 13 / er 9, Whisper 2 / 13 / 4). The
  v5 rule ah/er -> uh undoes this and was board-verified (0.0062 with the casing guard). Untrained models also write fewer
  fillers per 1k words (base Omni 5.9, Parakeet zero-shot 4.0) than the refs (7.4) or the trained members (8.7-10.4).
  Listening page (private): https://claude.ai/artifact/RCuwa9JJ3CCqa3GCtXAyN8 (dev + Miami clips, members, vote, untrained).
- **vLLM randomness, measured** (`kfold/vllm_noise.sh`, v8 Omni stage twice on identical inputs, GPU 2): Omni member dev
  0.1123 vs 0.1105 (18/155 clips differ), Miami half 0.1904 vs 0.1957 (104/1447), long 0.1081 vs 0.1087 (41/58).
  **After the v8 vote: dev 0.0759 vs 0.0759, Miami 0.18470 vs 0.18464** - the vote absorbs almost all of it. The vote code
  itself is not deterministic: same inputs twice -> Miami 0.18470 vs 0.18452, output files differ (parallel workers).
  Noise floor for vote comparisons: ~0.0002 from the vote code, up to ~0.0005 when a member is re-decoded.
- **Error browser** (same private artifact, v4): seven error types of the v8 vote with clips and audio. Dev vote errors:
  near-miss 73, extra 72, names/rare 63, misheard 52, dropped 45, fillers 41, wrong language 32 (378 total).
- **Cut-off words ("ha...")**: the default Miami labels (`miami_prep/manifest.jsonl`) have 0 dotted words; the frag
  versions have ~1,020 (dev refs 0.50 % of words, Miami 0.44 %). Only Cohere trained on frag labels (KF_MANIFEST);
  Omni/Voxtral/Qwen3-ASR/Parakeet/Conformer never saw a dotted word. All 25 dev cut-offs are v8 errors (0.0050).
  Cohere's dotted words: dev 1/6 exact, Miami 3/15 exact - but several "wrong" ones are real cut-offs spelled
  differently (ma... vs me..., wha... vs hua..., pre... vs pedo..., hor... vs horr...). Detecting a cut-off is not
  enough: the fragment's spelling must match exactly. Listening tab added to the artifact (v5).

### 7.x Length skew of the Miami half (2026-09-28, `kfold/short_clips.py`, `kfold/bucket_wer.py`)

Dev: 155 clips, median 13 s, min 2 s, 0 clips under 1 s, 0 clips with <= 2 reference words. Miami half: 1449 clips, median
2.2 s, 306 under 1 s, 651 under 2 s, 376 clips with <= 2 words (2.9 % of words). The board tracks dev, not Miami. Miami-half WER by length bucket:

| bucket | clips | words | Qwen LoRA (current) | Qwen full FT | vote w/ LoRA | vote w/ full | v8 vote |
|---|---|---|---|---|---|---|---|
| 0-2 s | 649 | 1995 | 0.4125 | 0.3684 | 0.3373 | 0.3293 | 0.3444 |
| 2-5 s | 460 | 4704 | 0.2600 | 0.2383 | 0.2351 | 0.2326 | 0.2313 |
| 5-10 s | 224 | 5191 | 0.1368 | **0.1483** | 0.1402 | 0.1406 | 0.1402 |
| 10-30 s | 114 | 5398 | 0.1271 | 0.1271 | 0.1275 | 0.1282 | 0.1280 |

The full FT's whole Miami gain sits in clips under 5 s (backchannels, one-word turns); on 5-10 s it is worse and on 10-30 s
equal, matching its dev/long loss. **Rule confirmed: judge Miami-half results on the 5-30 s buckets (10.6k words), not the
total.** The 10-30 s Miami bucket scores like dev (0.127 vs 0.12), so it is a usable dev-like second judge.

### 7.y Chunked vote for clips over 30 s — ADDED to v8 (2026-09-28)

**Problem.** Same words, glued: v8 vote dev 0.0757 vs long 0.0874 (+0.012), while members lose only 0.003-0.008 and the loss
is flat over 60/120/220 s (`kfold/long_vs_dev.py`). Cause: `median_extended` builds candidates only on the edit path between
TWO members, so on a 300-word clip it cannot mix member X in one sentence with member Y in another.

**Fix** (`kfold/patch_v8_chunkvote.py`, applied to `submission_v8_code/vote.py`, backup `/tmp/vote.v8bak`, marker
`CHUNK_VOTE_PATCH`): for clips over 30 s (the package's existing split threshold), cut every member at anchors = pivot words
(heaviest member) that every non-empty member has at the aligned position, >= 30 pivot words per chunk (= dev's mean
words per clip), vote each chunk with the unchanged median_extended, join, then gap-fill + option2 as before. Cut is lossless
(406/406 member texts rebuilt; the code falls back to the whole clip if not).

| real v8 vote.py, 3 runs | dev | Miami half | long |
|---|---|---|---|
| v8 before | 0.0757 | 0.1848 | 0.0874 |
| v8 + chunked vote | 0.0753 / 0.0753 / 0.0755 | 0.1849 / 0.1849 / 0.1850 (0 clips chunked = vote noise) | **0.0826 / 0.0824 / 0.0826** |

Not tuned: chunk size 20 / 30 / 50 / 80 -> long 0.0829 / 0.0824 / 0.0838 / 0.0848, all better than 0.0874. Errors on long:
sub 821 -> 760, ins 308 -> 276, del 181 -> 198. Vote time on the long set 34 min -> 15 s. Expected board effect ~0.001. **Re-apply after any rebuild from make_v8_code.py** (like the no-fallback and
Cohere-beam patches).

**Chunked vote is not a glue artifact** (`kfold/chunk_boundary.py`, `kfold/chunk_lc.py`): natural dev clips > 30 s (5 clips,
530 words, no glue) 0.0943 -> 0.0887; only 7 % of cuts land at a glue boundary (about chance); forbidding cuts within 4 words of
any glue boundary still gives long 0.0874 -> 0.0832; lowercased scoring keeps the full gain (long 0.0762 -> 0.0714).

**Most of the "long vs dev" gap is a scoring artifact of the glued set** (`kfold/long_decomp.py`, `kfold/glue_artifact.py`):
the official normalizer keeps capitals except at the start of the text and after a period, so gluing clips makes mid-text
capitals count. Same member texts, per clip vs glued: official +0.0032..+0.0035 for every member, lowercased 0.0000..-0.0002.
Lowercased, v8+chunked: dev 0.0699 vs long 0.0714 -> the real long-audio loss is ~0.0015 (vote on glued short decodes 0.0708,
+0.0009; decoding long audio +0.0006). Char vs word vote rechecked on v8 (`kfold/char_vs_word.sh`): word-level wins everywhere
(dev 0.0757 vs 0.0765, long 0.0825 vs 0.0844, Miami 10-30 s 0.1280 vs 0.1295); Parakeet adds dev -0.0018, Miami 0-5 s -0.0031,
5-10 s +0.0008, 10-30 s -0.0008, long -0.0002.

### 7.z v8 STRICT mode + build (2026-09-28)

User rule: any failure must end the job with **no submission file** (the platform refunds the slot), never a degraded
submission. `kfold/patch_v8_strict.py` (marker `STRICT_V8`, backups `/tmp/strict_bak/`) removed every quiet fallback:
the empty submission written at start, Omni-as-submission fallback, dropped members (main + vote.py), time-based stage
skips and beam downgrades (Qwen3-ASR and Cohere now always 4 beams), split/merge fallbacks, the final "restore a member"
block, the refiner block, vendor_tf unpack fallback, Omni budget-hit empty clips, Voxtral HF fallback, Cohere beam->greedy
and clip-by-clip-empty, Parakeet clip-by-clip-empty and missing-generation-config default, Whisper OOM retry. New pre-flight:
all 7 model dirs + key files, delta, unpacked vendor_tf, every audio file, before any model loads; `check_member` after each
stage (all clips present, >= 90 % non-empty); final check (all rows, >= 98 % non-empty). Kept: soundfile->librosa audio
loader (identical waveform), mistral_common import alias, stage timeouts (now abort).

Build: `kfold/build_v8pkg.sh` (GPU 3, HDD) -> `$T/submission_v8/`, models hardlinked from submission_v7 (file count and
bytes asserted equal), delta = `qwen3_omni/lora_runs/miami_r16_AB/miami.pt` (md5 76be418e4549), vendor_tf.zip from
sub_mech2 (as in id-2783). Each validation run must pass `kfold/verify_v8_run.py` (every v8 ingredient visibly
active: delta split/merge, súper, gap-fill, Cohere 4 beams, chunked vote count, 7 members ok, no ABORT/fallback text,
vendored Parakeet, job log < 400 lines, no transcript text in the job log). Failure injection: `kfold/inject_cpu.sh`
(missing model file, missing audio: both PASS, stale submission removed), `kfold/inject_gpu.sh` (garbage Cohere weights,
garbage delta; queued after the build). Local stack = runner stack (python 3.12.14, torch 2.11.0+cu130, transformers 4.57.6,
vLLM 0.23.0). vendor_tf.zip + layout are the ones id-2783 already used.

## 8. v8 on the board + experiments for the final (2026-09-28)

**v8 = 0.0993** (v7 0.1016), rank #3, leader 0.0990; **21 min runtime**.

Extra-voter tests (real v8 vote via `kfold/vote_any.py` + `kfold/xeval.py`, weights fixed before results: TTA / sampled
pass = half the member's weight, second seed = full weight). v8 base: dev 0.0753, long 0.0826, Miami 0.1846 (5-10 s
0.1412, 10-30 s 0.1278).

| candidate | dev | long | Miami | 5-10 s | 10-30 s | verdict |
|---|---|---|---|---|---|---|
| Cohere seed 2 (8th voter, w16) | 0.0753 | 0.0825 | 0.1825 | 0.1389 | 0.1262 | passes, but superseded by the seed-3 retrain below |
| TTA Whisper 0.9/1.1 (w8+w8) | 0.0761 | 0.0816 | 0.1808 | 0.1362 | 0.1237 | borderline: dev 9 better/16 worse (+4 words, n.s.), long 24/11, Miami 98/62 |
| TTA Cohere | 0.0759 | 0.0820 | 0.1824 | 0.1381 | 0.1263 | fails dev |
| TTA Omni | 0.0773 | 0.0830 | 0.1847 | 0.1404 | 0.1288 | dead |
| TTA Parakeet | 0.0771 | 0.0831 | 0.1852 | 0.1402 | 0.1286 | dead |
| TTA Voxtral | 0.0769 | 0.0830 | 0.1860 | 0.1406 | 0.1297 | dead |
| Omni sampled x2 (T 0.7, top-p 0.8, top-k 20) | 0.0781 | 0.0832 | 0.1860 | 0.1393 | 0.1282 | dead |
| Omni beam 4 (vLLM offline beam search) | 30 dev clips: 0.1861 vs greedy 0.1360 (77 vs 18 deletions), 9x slower | | | | | dead |
| Voxtral fp32 / lower LR retrain | = "Voxtral v2" (09-27): vote worse on all three | | | | | dead (already done) |

Running: Cohere concat-mix seed 3 (GPU 3, `cohere_train_concat.py`: p 0.5, same-speaker consecutive utterances glued to
5-28 s) and Cohere plain seed 3 (GPU 2, fresh weights = replacement for the seed-2 result).

**Cohere retrains (2026-09-28, GPU 3 + GPU 2, same recipe as seed 1: A+B frag labels, lr 1e-5, early stop on Tsel600):**

| Cohere member | Tsel600 best | dev | long | Miami | 5-10 s | 10-30 s |
|---|---|---|---|---|---|---|
| seed 1 (shipped) | 0.2161 | 0.1091 | 0.1142 | 0.1857 | 0.1310 | 0.1182 |
| seed 2 | 0.2168 | 0.1141 | 0.1187 | 0.1857 | 0.1356 | 0.1099 |
| seed 3 plain (`cohere/AB_frag_seed3`) | 0.2130 | 0.1141 | 0.1145 | 0.1887 | 0.1358 | 0.1189 |
| concat-mix seed 3 (`cohere/AB_frag_concat_s3`, same-speaker glue 5-28 s, p 0.5) | 0.2148 | 0.1179 | 0.1215 | 0.1833 | 0.1285 | 0.1139 |

As 8th voter (w16): seed 3 dev 0.0767 / long 0.0824 / Miami 0.1834 (fails dev); concat-mix 0.0771 / 0.0833 / 0.1827
(fails dev + long); TTA Whisper + seed 3: 0.0761 / 0.0814 / 0.1799. **Seed 2 vs seed 3 as the 8th voter moved dev by
0.0000 vs +0.0014 -> dev cannot resolve ~0.001 voter effects.** Concat-mix helps Miami mid-length clips but hurts the
organizer-style audio (dev and long). A single-Cohere concat-mix package was built (packaged dev 0.1023 vs 0.1001 for the plain seed-2 single Cohere) -> predicted worse.

## 9. New-member night (2026-09-28/29): canary, Granite NAR, Phi-4, MOSS

8th voter added to the v8 seven through the real v8 vote (`kfold/xeval.py`), weight fixed in advance at 8 (16 as a
secondary check). Base in these runs: dev 0.0755 / long 0.0826 / Miami 0.1848. Decodes: `kfold/cand_zs.py` +
`kfold/zs_one.sh` (clips > 30 s cut with the package's seg_long), outputs `$T/xexp/<tag>/`.

| candidate | member dev | long | Miami | 8th voter w8 dev / long / Miami | verdict |
|---|---|---|---|---|---|
| canary-1b-v2 Miami FT (09-23 model) | 0.1798 | 0.1792 | 0.2447 | 0.0775 / 0.0831 / 0.1859 | dead |
| granite-speech-4.1-2b-nar zero-shot | 0.2839 | 0.2792 | 0.3369 | 0.0761 / 0.0834 / 0.1850 | dead |
| granite NAR Miami full FT (Tsel600 0.352->0.262, `granite_nar_train.py`) | 0.1862 | 0.1915 | 0.2375 | 0.0773 / 0.0835 / 0.1845 (w16 Miami 0.1825) | dead on dev/long |
| Phi-4-multimodal (ZFTurbo fork) zero-shot | 0.4449 | 0.4121 | 0.3252 | **0.0753 / 0.0822 / 0.1847** | neutral-positive -> fine-tune |

Phi-4 zero-shot dev failures are language drift (French / translation to English), which Miami FT fixed for canary.
Granite NAR is extremely fast (all 1,967 pieces in ~2 min) but tops out at Tsel600 0.26 (Cohere 0.21).
Phi-4 fine-tune: `kfold/phi4_train.py` (speech LoRA r320 + audio encoder + speech projection, 915M trainable, fp32
masters, bf16 frozen LM; empty image tensors must be dropped or train mode runs a fake vision forward).

**Phi-4 Miami FT (best step 1500, Tsel600 0.3526 -> 0.2433):** member dev 0.1518 / long 0.1571 / Miami 0.2197.

| 8th-voter set (weights fixed in advance) | dev | long | Miami | 5-10 s | 10-30 s |
|---|---|---|---|---|---|
| v8 base | 0.0755 | 0.0826 | 0.1846 | 0.1410 | 0.1280 |
| + Phi-4 FT w8 | 0.0757 | 0.0819 | 0.1839 | 0.1406 | 0.1265 |
| + TTA Whisper (0.9/1.1, w8 each) | 0.0761 | 0.0816 | 0.1807 | 0.1362 | 0.1237 |
| **+ TTA Whisper + Phi-4 FT w8** | **0.0751** | **0.0808** | **0.1799** | 0.1360 | 0.1236 |

Per clip (TTA Whisper + Phi-4 vs base): dev 9 better / 11 worse (net -2 words), Miami 111 / 57 (net -82), long 19 / 8 (net -26).
**First candidate since v8 that is non-negative on all three sets with no tuned setting -> ship candidate for the final main submission.**
MOSS zero-shot (0.9B): member dev 0.1978 (best zero-shot tonight) but vote 0.0759 / 0.0836 / 0.1850 -> dead as-is; Miami FT running.

### v9 package (built 2026-09-29 00:43, NOT submitted)
`$T/submission_v9_code` from `kfold/v9/make_v9_code.py` (asserted patches on the submitted v8 code): + `stage_phi4.py`
(Phi-4 base + Miami delta, 1137 tensors, batch 8, split > 30 s), + `stage_tta.py` (soxr speed perturbation 0.9 / 1.1,
same as `make_tta.py`) followed by `stage_whisper.py` on each, vote WEIGHTS + `whisper_s090` 8 / `whisper_s110` 8 / `phi4` 8,
strict vote refuses any member-name mismatch (names outside WEIGHTS used to be ignored silently), `vendor_phi/backoff`.
v9 vote.py on the lab member files = xeval exactly (long 0.080830 both; dev flips 0.07509 / 0.07529 in both = known
nondeterminism). `kfold/v9/build_v9pkg.sh` (GPU 3, models hardlinked from submission_v8 and byte-checked):
**packaged dev 0.0751 (v8 0.0755), long 0.0814 (v8 0.0826)**, verify_v9_run.py OK on both (Phi-4 delta loaded, both
TTA sets complete, 10 voters, no ABORT/fallback, 94 / 103 job-log lines). Stage times on dev: phi4 168 s, whisper_s090 75 s,
whisper_s110 72 s (cohere 86 s) -> about +8-10 min on the test run vs v8's 21 min. Package 72 GB (`submission_v9.zip`).
Gemma-4-E4B LoRA+audio FT stopped by hand (Tsel600 0.576 -> 0.352, flat). MOSS full FT: best Tsel600 0.2304 @ step 250
(early stop at 1000); decode + vote test running.

**MOSS-Transcribe-Diarize Miami full FT** (`kfold/moss_train.py`, keeps its "[t][S01] text[t]" format, lr 1e-5, best step
250 Tsel600 0.2304, zero-shot 0.3126): member **dev 0.1169** / long 0.1205 / Miami 0.2014 - the strongest new member of
the night (Cohere 0.1091) - but no vote gain: 8th voter w8 0.0755 / 0.0838 / 0.1842 (long per clip 8 better / 21 worse);
on top of v9 0.0765 / 0.0812 / 0.1798 (dev +0.0014); replacing the conformer inside v9 0.0757 / 0.0807 / 0.1804 (dev +0.0006).
Dead: a strong member that is redundant with the others (Whisper-style encoder). Decode is also slow (batch 1, 44 min on 1,967 pieces).

## 10. Concat-mix Cohere check + new-family night (2026-09-29/30)

**Concat-mix single Cohere vs plain single Cohere:** 0.1184 vs 0.1201 (user-confirmed 2026-09-30; the old
0.1223 note was wrong). Same main.py, clean run.
Dev had predicted concat-mix worse (0.1023 vs 0.1001), Miami better. The difference is within test-sampling noise and within
Cohere seed-to-seed spread (0.005 on dev), so it does not prove concat-mix helps. Inside the vote it hurts everywhere:
v9 + concat Cohere w16 0.0763 / 0.0822 / 0.1803; v9 with Cohere swapped for it 0.0759 / 0.0822 / 0.1818; v8 swapped
0.0767 / 0.0843 / 0.1852 (v9 0.0753 / 0.0811 / 0.1802). Those weights also cannot ship on main. -> v9 unchanged.

New families tested on top of v9 (weight 8, fixed in advance; v9 on the same runs 0.0751-0.0753 / 0.0809-0.0811 / 0.1799):

| candidate | recipe | member dev / long / Miami | v9 + candidate dev / long / Miami | verdict |
|---|---|---|---|---|
| Phi-4 FT on 0.9x / 1.1x speed audio (w4 + w4) | decode only (`kfold/phi4_tta.sh`) | s090 0.1512 / 0.1582 / 0.2165, s110 0.1966 / 0.1747 / 0.2381 | 0.0755 / 0.0810 / 0.1800 | dead (slightly worse on all three) |
| SeamlessM4T-v2-large full FT (`kfold/seamless_spen_train.py`, tgt spa, lr 2e-5, bs 4x2) | Tsel600 0.936 zero-shot -> 0.2794 (step 4500) | 0.1664 / 0.1799 / 0.2589 | 0.0751 / **0.0820** / 0.1800 | dead (long +0.001) |
| w2v-BERT 2.0 CTC (`kfold/w2vbert_train.py`, scorer-normalised chars, lr 5e-5) | Tsel600 0.853 (500) -> 0.567 (2500), flat at 3000 | - | - | stopped by hand: stalled far above voter range |

Seamless note: the tokenizer writes labels as [</s>, __spa__, text, </s>]; the leading </s> must be dropped so training
matches generate() (the Javanese trainer did not do this). w2v-BERT needed ctc_zero_infinity and dropping ~IPA-labelled clips.

## 11. Last-shot retrains of VibeVoice + Nemotron (2026-09-30)

Both families had short Miami fine-tunes before (VibeVoice LoRA r32 attn-only 1200 steps; Nemotron killed at 1175) and
were vote-negative/neutral on v8. Longer / bigger retrains, same protocol (`kfold/last_chain.sh`, Tsel600 selection,
9th voter on top of v9 at weight 8 fixed in advance). miner1 rebooted at ~10:31 (new Jupyter token) -> both relaunched.
- **Nemotron** (lr 3e-5, warmup 300, patience 4, 5000 steps): Tsel600 0.4054 -> 0.2891 @ 4250 (old run 0.308).
  Member dev 0.1650 / long 0.1765 / Miami 0.2601 (29/1449 Miami empty). v9 + it: 0.0747 / **0.0818** / **0.1815**
  vs v9 0.0751 / 0.0807 / 0.1801 -> dead (long and Miami worse).
- **VibeVoice** (LoRA r64/a128 on attn + MLP, 161.5M, accum 8, lr 1e-4, GPU 3): Tsel600-200 0.7284 -> 0.4338 @ 2000,
  early stop 3500; JSON scaffolding gone from step 500 (78/200 -> 0). Member dev 0.1400 / long 0.1446 / Miami 0.2373.
  v9 + it: 0.0751 / **0.0824** / 0.1789 (T 0-5 s 0.2569 vs 0.2600) vs v9 0.0751 / 0.0811 / 0.1802 -> fails the rule
  on long (+0.0013); first candidate to help Miami by >0.001, but dev flat and long worse. Not shipped.
v9 stays final. Adapter `$T/vibevoice/AB_last/adapter`, Nemotron `$T/nemotron/miami_AB_last/best`.

## 12. Night of 2026-09-30/10-01: omniASR + cheap vote checks (GPUs 2+3)

All on top of v9 (dev 0.0751 / long ~0.0811 / Miami ~0.1800), weights fixed before results. **Nothing passes; v9 stays.**
- CPU: Omni weight 16 0.0759 / 0.0808 / 0.1802 (dev worse); drop Conformer 0.0769 / 0.0809 / 0.1803 (Conformer
  now earns its slot); Cohere 0.9/1.1 TTA pair w8+w8 0.0755 / 0.0813 / 0.1784 (Miami gain only < 5 s, 10-30 s worse).
- omniASR (Meta Omnilingual, never tried on SP-EN; `kfold/omni_dec.py`, `train_omni_spen.py` = JV trainer with eval
  lang from held rows + selection on lowercased WER, `omni_chain.sh`, casing lexicon `omni_casing.json`, 442 words).
  Zero-shot dev: 7B spa_Latn 0.269, eng 0.285, none 0.630 (loops), 3B spa 0.282. Miami A+B FT (fp32 masters, top-12
  encoder layers, lr 1e-5, accum 2, 1500 steps): Tsel200 0.96 -> 0.699 @1000 (7B); 3B crashed at eval step 500 on
  GPU 3 (known in-process-decode fault), kept step 250. Members: 7B dev 0.188 / long 0.186 / Miami 0.380; 3B 0.293 /
  0.298 / 0.529. On v9: 7B 0.0753 / 0.0813 / 0.1801 (10-30 s 0.1245 vs 0.1236), 3B 0.0759 / 0.0814 / 0.1804 -> dead.
  Decode needs a >= 4 s ballast piece in all-short batches (CUDA invalid argument otherwise).
- Second Whisper speed pair 0.95/1.05 (members dev 0.1059 / 0.1079): w8 each 0.0765 / 0.0820 / 0.1815 (too much
  Whisper weight); w4 each 0.0753 / 0.0809 / 0.1800 = noise -> dead.

## 13. Long clips on v9 (2026-10-01): verified, nothing left to recover

`kfold/long_v9.py` (long set = the 155 dev clips glued into >= 60 / 120 / 220 s groups, so every long clip has an exact
short twin). v9, per tier, A = dev vote per clip, B = dev vote glued, C = short decodes glued then voted, D = package path:
| tier | official A / B / C / D | lowercased A / B / C / D |
|---|---|---|
| 60 s | 0.0751 / 0.0783 / 0.0805 / 0.0809 | 0.0693 / 0.0693 / 0.0695 / 0.0705 |
| 120 s | 0.0751 / 0.0787 / 0.0813 / 0.0801 | 0.0693 / 0.0693 / 0.0697 / 0.0689 |
| 220 s | 0.0751 / 0.0793 / 0.0817 / 0.0823 | 0.0693 / 0.0693 / 0.0693 / 0.0699 |
| all | 0.0751 / 0.0788 / 0.0812 / 0.0811 | 0.0693 / 0.0693 / 0.0695 / 0.0698 |
The whole official gap (+0.0060) is capital letters at the glue points of our own constructed references (B-A is +0.0037
official and 0.0000 lowercased with identical texts). Lowercased, long audio costs v9 +0.0005 in total (+0.0012 / -0.0004 /
+0.0006 by tier) = noise. Members lowercased, short-glued vs long 60/120/220: whisper 0.0953 vs 0.0947/0.0979/0.0939,
omni 0.1007 vs 0.1001/0.0925/0.0921 (better on long), qwen3asr 0.1113 vs 0.1131/0.1163/0.1167 and conformer +0.005
(both low weight). `kfold/join_case.py`: piece joins are clean - 96-97 % of the ~299 joins per member follow a piece that
ends with . ? !, and 0 joins produce a spurious mid-text capital; a join-case rule changes the v9 long vote by -0.0002
(2 clips) = nothing. Limit: the long set measures LENGTH, not natural long-conversation difficulty (only 5 natural dev
clips > 30 s exist).

## 14. Night of 2026-10-01: Whisper variants, public HF models, omniASR v2, Voxtral Mini, Qwen TTA (all on v9, w fixed)

v9 reference on these runs: dev 0.0751-0.0753 / long 0.0809-0.0811 / Miami 0.1797-0.1804 (10-30 s 0.1236).
| candidate (weight) | member dev / long / Miami | v9 + it: dev / long / Miami (10-30 s) | verdict |
|---|---|---|---|
| Whisper fragment-label sibling `full_frag_s2` (8) | 0.1159 / 0.1161 / 0.2462 | 0.0755 / 0.0813 / 0.1800 (0.1236) | dead |
| Whisper seed-1 sibling `full_s1` (8) | 0.1087 / 0.1144 / 0.2810 | 0.0753 / 0.0808 / 0.1796 (0.1239), gain only < 5 s | dead |
| Whisper large-v3 FULL FT (lr 1e-5, 3 ep, ckpt 870, Tsel600 0.2335 vs shipped 0.2618) extra (8) | 0.1071 / 0.1134 / 0.2733 | 0.0757 / 0.0812 / 0.1800 (0.1250) | dead |
| same, swapped for Whisper (16) | | 0.0749 / 0.0816 / 0.1809 (0.1247) | dead |
| HF drewoodward/whisper-small-spanglish (8; tokenizer files from openai/whisper-small) | 0.1984 / 0.2048 / 0.4720 | 0.0769 / 0.0822 / 0.1802 | dead |
| HF chengyili2005 whisper-medium en-es LoRA merged (8) | 0.1872 / 0.2516 / 0.3983 | 0.0757 / 0.0814 / 0.1808 | dead |
| HF chengyili2005 whisper-medium es-en LoRA merged (8) | 0.1876 / 0.2494 / 0.3408 | 0.0767 / 0.0818 / 0.1809 | dead |
| omniASR-LLM-7B **v2** (written tokenizer, cased targets, ckpt 250; zero-shot dev 0.191 spa / 0.232 eng) (8) | 0.1734 / 0.1769 / 0.3248 | 0.0763 / 0.0819 / 0.1806 | dead |
| Qwen3-ASR 0.9/1.1 speed pair (5+5) | 0.1203 / 0.1312 / 0.2031 (s090) | 0.0747 / 0.0808 / 0.1813 (0.1243) | dead (Miami worse) |
| Voxtral-Mini-3B LoRA (shipped Voxtral recipe, best step 1000, Tsel600 0.7109 -> 0.2514) (8) | 0.1239 / 0.1207 / 0.2145 | 0.0753 / 0.0810 / 0.1816 (0.1245) | dead (same family as Voxtral Small) |
tangering-ai distil model skipped (no license).
| Whisper large-v3-TURBO full FT (same recipe; ckpts 435/870/1305 Tsel600 0.2155/0.2185/**0.2132**) (8) | 0.1135 / 0.1141 / 0.3132 (0-5 s 0.604, 10-30 s **0.1199** = best of any member) | 0.0767 / 0.0808 / 0.1794 (0.1241) | dead (dev +0.0016) |
| Phi-4 Miami FT seed 2 (same recipe, best step 1000, Tsel600 0.2446 vs shipped 0.2433) (8) | see night.log | 0.0761 / 0.0814 / 0.1790 (0.1239) | dead (dev +0.0010, long +0.0005; Miami gain < 10 s) |

**2026-10-01 morning follow-ups (honest-report gaps closed):**
- Whisper full FT, PROPER swap (main 16 + its own 0.9/1.1 speed copies 8+8 replacing all three Whisper slots):
  0.0759 / 0.0815 / 0.1808 (10-30 s 0.1252) vs v9 0.0751 / 0.0808 / 0.1804 (0.1236) -> worse everywhere, dead.
- omniASR 7B v2 at lr 5e-6, eval every 125: Tsel200 0.765 -> 0.872 (125) -> **0.593 (250)** -> 0.683 -> 0.739 (500),
  output length rising to 1.26x - same early peak and insertion drift as at 1e-5, so the LR was not the cause.
- Transcript read-out of every near-miss: `SPEN_EXPERIMENTS_ON_V7_2026-10-01_readout.md`.
- omniASR 7B v2 lr 5e-6 (ckpt 250): member dev 0.1838 / long 0.1813 / Miami 0.3412 (worse than the 1e-5 run's 0.1734);
  v9 + it 0.0765 / 0.0816 / 0.1808 (10-30 s 0.1249) -> dead. omniASR closed for good.


## 15. v9 error read-out against the references (2026-10-01 afternoon, CPU only)

Scripts `kfold/err_look.py`, `err_look2.py`, `case_rule.py`, `vocab_diff.py`; reports in `/kaggle/working/{err_look,err_look2,case_rule,vocab_diff}.txt`
(local copies `%TEMP%/spen_last/ea/`). v9 vote = `xexp/votes/tta_whisper_phi4_{dev,long,T}.csv`.

**dev (4,994 ref words, 375 errors = 221 sub / 61 del / 93 ins):**
- Where the ref word is missed (282): no member has it 148 (0.0296), a minority has it 128 (0.0256), a majority has it 6 (0.0012).
  Inserted words (93): 85 are written by a majority of the members. No member is under-trusted (each is right 24-40 times
  where the vote is wrong, wrong 76-645 times where the vote is right).
- ~43 % of the errors are writing conventions, not hearing: hesitation sounds ~36 (um/uh/eh precision 0.50/0.48/0.46 -> dropping
  them is 0/+2/+1), cut-off words `gro...` 25 (no system writes them: best candidate 3 of 25), capitals 27 (i after a ref full
  stop 7, capitals after `¿` / annotator capitals 10, names 6, english/christmas lower-case in the ref 4), ok/okay 7 (ref ok 7 :
  okay 13 : Okay 2), `I am`/`it is`/`she is` 14 (ref expands 6 of 53), digits 8 (ref 5 digits : 56 spelled out), hyphens 7,
  accents 12, annotator typos + apostrophes ~20. Spanish small word heard as English (o/or, en/in, y/and, es/it's) ~20.
- No clip-boundary problem on dev (first/last word 0.065/0.071 vs 0.055 interior). On the Miami half the first word is wrong
  25 %, 266 of 761 insertions sit before the first word (segment cuts, cross-talk).

**Miami half:** ~410 of 3,119 errors are our own dev-style rules (gonna/wanna/ok/o_k/ah, `I` after a ref full stop), ~130 are
anonymised names (ref Selena/Clare/Annie/Rupert vs the name actually spoken), ~80 underscore names. 56 % of the errors are in clips under 5 s.

**Rules measured on the v9 text (dev / long / Miami word errors):**
| rule | logic | result |
|---|---|---|
| `quien` -> `quién` | refs never write it without the accent: dev 8:0, Miami 4:0; v9 writes `quien` twice on dev | **-2 / -6 / 0** (dev 0.0751 -> 0.0747) |
| one casing per word from the members' majority | a name is a name in every clip | 0 / 0 / 0 (the vote already follows the majority; members split 166:204 on Zumba) |
| + drop our . ! ? before such a name | the scorer lower-cases after them | -1 / -2 / +1 |
| both directions (also force lower-case) | | +2 / +5 / -1 -> dead |
| drop um / uh / eh | annotator writes half of them | 0 / -1 / +49, +2 / -4 / -1, +1 / +11 / +3 -> dead |
| capital after a detected question start | ref `¿` 17: 3 clip-initial, 7 after . ! ? (lower-cased), 7 mid-sentence | coin flip, and only 1 in 3 ref questions has `¿` -> dead by counts |
| `I'm` -> `I am`, `or` -> `o`, `okay` -> `ok`, digits | | +26, +25, +6 on dev -> dead |
| dev-only lexicon: zumba -> Zumba (ref 4:0), Marshalls -> Marshall's (2:0), mmhm -> mm (2:0) | annotator spellings | -2, -2, -1 on dev, in-sample, 2-4 examples each |

Conclusion: text rules are exhausted except `quién` (~0.0004). The two classes with real size (cut-off words 0.005, minority-right
words 0.026) need the members to change (labels that keep `...`) or a vote that uses per-word confidence; neither fits before the deadline.

**v9b (2026-10-01 12:35 UTC): v9 + `quien` -> `quién`, nothing else.** `$T/submission_v9b/` = hardlink copy of `submission_v9/`, only
`spelling_rules.py` replaced (`kfold/v9/patch_v9b_quien.py`; `fix_quien` is the last step of `option2`). `kfold/v9/check_v9b.py`
(real v9b `vote.py` on the stored member outputs, old vs new rules on the SAME raw vote): dev 376 -> 374 errors, long 1217 -> 1211,
Miami 3115 -> 3115; every changed word is quien -> quién; spelling/vote self-tests and `test_rules.py` pass. Side finding: the vote
is not bit-reproducible between runs (parallel workers): 1 / 7 / 19 clips differ from the lab v9 run, +-1 / +-7 / +-4 word errors.
Zip + CRC test + entry-by-entry comparison with `submission_v9.zip`, and a full packaged dev run from a second hardlink copy
(`submission_v9b_val`, GPU 3): `kfold/v9/v9b_step2.sh`, log `$T/build_v9b.log`. `submission_v9.zip` is untouched and stays the fallback.
