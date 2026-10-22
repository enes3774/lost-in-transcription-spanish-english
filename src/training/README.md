# Training: how every shipped weight was produced

Everything here is the code **as it was run** (copied verbatim from the training machine on 2026-10-21), plus this
guide. The scripts contain the absolute paths of that machine; the table below maps them so the commands can be
re-run elsewhere with a `sed`-style replacement or by recreating the same folders. Nothing was retyped or cleaned up,
so what you read is what produced the weights.

| path in the scripts | meaning | what to put there |
| --- | --- | --- |
| `/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa` (`$I`) | data root | `raw/miami/` = the Bangor Miami download; `miami_prep/`, `miami_kfold/` are created by the scripts |
| `/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track` (`$T`) | track workspace: training runs, packages | any empty folder |
| `/kaggle/working/omni_rt` (`$S`) | code folder with the main venv, `score.py`, `dev/`, `dev_long/` | this repository's `src/training/` folder + the organizers' dev set in `dev/` |
| `/kaggle/working/omni_rt/kfold` (`$K`) | the fold trainers | this folder's scripts |
| `/kaggle/working/omni_rt/venv/bin/python` | main training venv | see `environment/README.md` |
| `/kaggle/working/compat_venv/bin/python` | transformers-5 venv (Parakeet) | see `environment/README.md` |
| `/kaggle/working/verify_venv/bin/python` | NeMo venv (Conformer, official scorer) | see `environment/README.md` |
| `/home/jovyan/work/Enes/lost_in_transcription/cache/huggingface/hub/models--*/snapshots/*` | Hugging Face cache of the base models | `huggingface-cli download <repo>` |
| `/home/jovyan/work/Enes/lost_in_transcription/models/checkpoints/whisper-large-v3` | `openai/whisper-large-v3` | same |
| `/home/jovyan/work/Enes/miami_windows` | Whisper training windows (`prep_miami_windows.py`) | created by step 1 |
| `/home/jovyan/work/Enes/phi_extra` | folder holding the `backoff` package for the Phi-4 remote code | `pip download backoff` or `src/inference/vendor_phi` |

GPU: every job runs on a single card (`CUDA_VISIBLE_DEVICES`). Memory: Omni LoRA 65 GB, Voxtral LoRA ~50 GB, Cohere 49 GB, the rest under 40 GB.

## 1. Data preparation (`data_prep/`)

Input: the Bangor Miami corpus as distributed on the Mozilla Data Collective (56 `.cha` CHAT transcripts in `chat/`,
56 `.mp3` recordings in `audios/`, `word_level_tsvs/`, `Miami_doc.pdf`), placed in `$I/raw/miami/`. Also the
competition dev set (155 clips + `test_metadata.csv` + references) in `$S/dev/` as `clips/`, `test_metadata.csv`, `refs.csv`.

```bash
PY=$S/venv/bin/python
# 1. CHAT -> utterance clips. Cleans CHAT markup, drops utterances with xxx/yyy/www, merges consecutive same-speaker
#    utterances up to 20 s (gap <= 2 s), cuts 16 kHz mono wavs:  $I/miami_prep/clips/<id>.wav + manifest.jsonl (22,290 clips, 24.1 h)
$PY data_prep/miami_prep.py --cut --max_s 20 --gap_s 2
# 2. Speaker-disjoint split of the 56 conversations into groups A / B / T (named speakers never cross groups):
#    $I/miami_prep/split_spk3.json  (A 9,203 clips, B 10,430, T 2,633)
$PY data_prep/miami_split.py
# 3. Stage-compatible folders miami_A / miami_B / miami_T (+ miami_Tval600, a fixed 600-clip sample of T):
$PY data_prep/make_groups.py
# 4. T split into a selection part and an untouched test half:
#    miami_Tsel600 (200 clips each from herring16, sastre12, zeledon4; used for early stopping) and
#    miami_Ttest (all 1,449 clips of herring3, sastre13, zeledon6; the "Miami half" validation set, never used for any choice)
$PY data_prep/make_tsplit.py
# 5. Second label variant with the competition's cut-off-word convention ("&gro" -> "gro..."):
#    manifest_frag.jsonl and refs_frag.csv next to every refs.csv. Used by Cohere, Parakeet and Phi-4.
$PY data_prep/miami_prep_frag.py
# 6. Name list of the A+B clips (used to filter the Omni trainer):
sh data_prep/launch_omniAB.sh      # the first three lines only; the rest launches the Omni training, see below
# 7. Whisper training windows: consecutive same-speaker utterances joined into <= 24 s windows (6,951 windows, 21.6 h)
$PY data_prep/prep_miami_windows.py --miami-dir $I/miami_prep --out-dir /home/jovyan/work/Enes/miami_windows --max-seconds 24
# 8. The glued long validation set (58 clips of 60-220 s made from dev clips; every dev clip used once per target length)
$PY data_prep/make_long_dev.py
```

`common_kf.py` is imported by every fold trainer: it reads the manifest and split, filters clips by duration, applies
the early-stopping rule (`EarlyStop`: stop when the selection WER has not improved for `patience` evaluations) and scores
with the official normalizer. Environment variables it reads: `KF_VAL` (selection folder, default `miami_Tval600`),
`KF_MANIFEST` (`manifest.jsonl` or `manifest_frag.jsonl`), `KF_REFS`, `KF_TAG` (suffix for the run folder), `KF_AUG`
(phone-style augmentation, **0 for every shipped member**), `KF_EXTRA_FRAC` (CIEMPIESS mix, **0 for every shipped member**).

Which selection set each member used: Voxtral, Qwen3-ASR and the Conformer were trained first and early-stopped on
`miami_Tval600`; 300 of its 600 clips lie inside the later `miami_Ttest` half. Whisper, Cohere, Parakeet and Phi-4 used
`miami_Tsel600`, which is disjoint from the test half. The Omni LoRA ran for a fixed single epoch with no selection.
No member ever saw the competition dev set during training or selection.

## 2. The eight fine-tunes

Base-model snapshots (all public, downloaded with `huggingface-cli download`; the Conformer from NGC):

| member | base repo |
| --- | --- |
| omni | `Qwen/Qwen3-Omni-30B-A3B-Instruct` (training, bf16) and `cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit` (inference experts) |
| voxtral | `mistralai/Voxtral-Small-24B-2507` (training) and `RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic` (inference) |
| whisper | `openai/whisper-large-v3` |
| conformer | NGC `stt_enes_conformer_ctc_large_codesw` (NeMo `.nemo` file) |
| qwen3asr | `Qwen/Qwen3-ASR-1.7B` |
| cohere | `CohereLabs/cohere-transcribe-03-2026` (snapshot `b1eacc26`) |
| parakeet | `moondream/parakeet-ultra` (weights) + `nvidia/parakeet-tdt-0.6b-v3` (processor / generation config) |
| phi4 | `ZFTurbo/Phi-4-multimodal-instruct` (microsoft weights + a `generate()` fix in the remote code) |

Common environment for the GPU jobs: `export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false; cd $S/work`.

### 2.1 Qwen3-Omni (`omni/`)  → `models/omni` + `models/omni_delta/delta.pt`

```bash
# base checkpoint used by the trainer: the bf16 thinker + audio tower packed for vLLM (qwen3_omni/submission_vllm/model =
#   Qwen/Qwen3-Omni-30B-A3B-Instruct without talker/code2wav; see MODEL in omni_lora_train.py)
# LoRA r16 / alpha 32 / dropout 0.05 on the thinker's attention q,k,v,o; lr 1e-4; 1 epoch over the A+B clips >= 2 s
# (10,933 clips, 18.6 h); accumulation 8; seed 1; fp32 master weights for the LoRA; 1,367 steps, 2.6 h on one 96 GB card.
$PY omni/omni_lora_train.py --run miami_r16_AB --miami 30000 --miami_names $K/miamiAB_names.txt --seed 1 --epochs 1 --accum 8 --save_every 500
#   -> $T/qwen3_omni/lora_runs/miami_r16_AB/miami.pt   (the final step; step-1000 looped on one dev clip, so the last one ships)
cp $T/qwen3_omni/lora_runs/miami_r16_AB/miami.pt  models/omni_delta/delta.pt
# inference base = hybrid checkpoint: AWQ int4 experts from cyankiwi/..., attention q/k/v/o restored to bf16 from the original
# so the LoRA delta merges exactly; talker + code2wav dropped (21 GB)
$PY omni/make_hybrid_omni.py <cyankiwi_awq4_snapshot> <Qwen_bf16_snapshot> models/omni
```

### 2.2 Voxtral Small (`voxtral/`)  → `models/voxtral` (FP8 base + `lora/`)

```bash
# LoRA r16 / alpha 32 / dropout 0.05 on language_model self_attn q,k,v,o; lr 1e-4; accumulation 8; 50 warm-up steps;
# A+B clips >= 2 s; eval every 250 steps on KF_VAL, patience 2; max 2,500 steps.  Selection set: miami_Tval600 (default).
$PY voxtral/kf_voxtral.py AB 2500 250 2                        # -> $T/kfold/vox_AB/best  (adapter_config.json, adapter_model.safetensors)
#   log: early stop at step 1500, best step 1000 (selection WER 0.2386), 2.4 h on one 96 GB card
# inference base: RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic converted to the Mistral layout vLLM loads
# (consolidated.safetensors, params.json, tekken.json: the repo's own convert_voxtral_hf_to_mistral.py); the adapter is
# served by vLLM as a LoRA on top of it.  voxtral/v4_build.sh lines 12-14 show the assembly.
cp $T/kfold/vox_AB/best/adapter_*  models/voxtral/lora/
```

### 2.3 Whisper large-v3 (`whisper/`)  → `models/whisper` (also used for the two speed-perturbed voters)

```bash
# whisper_full.sh <gpu> 2  does all of this: LoRA r32 (attention + MLP), lr 1e-4, batch 8 x accum 2, 3 epochs over the
# 24 s windows, warm-up 100, gradient checkpointing, seed 2; then every saved checkpoint is merged in fp32
# (whisper_merge.py) and decoded on miami_Tsel600 (kf_decode_whisper.py + whisper_pick.py); the best is kept.
sh whisper/whisper_full.sh <gpu> 2
#   checkpoint scores on the selection half: step 435 0.3141, step 870 0.2618 (kept), step 1305 0.2716
#   -> $T/kfold/whisper_folds/full_s2  (merged fp32, 5.8 GB)
$PY whisper/whisper_to_fp16.py $T/kfold/whisper_folds/full_s2 models/whisper      # fp16 on disk, 3.1 GB
```

The underlying trainer is `whisper/train_whisper_miami.py` (HF Seq2SeqTrainer + peft); the exact arguments are in `whisper_full.sh`.

### 2.4 Conformer-CTC (`conformer/`)  → `models/conformer`

```bash
# NeMo stt_enes_conformer_ctc_large_codesw, full fine-tune in a plain PyTorch loop (NeMo's loaders crash on the training box):
# batch 16, lr 5e-5, A+B clips >= 2 s, eval every 500 steps on KF_VAL (miami_Tval600), patience 3, max 16,000 steps (~6 epochs).
# Saves the best weights as a plain state dict + the SentencePiece tokenizer, which src/inference/conformer_port.py loads.
CUDA_LAUNCH_BLOCKING=1 /kaggle/working/verify_venv/bin/python conformer/kf_conformer.py AB 16000 500 3
#   log: early stop at step 15500, best step 14000 (selection WER 0.3632), 6.2 h
cp $T/kfold/conformer_AB/serve/{conformer_miami.pt,tokenizer.model}  models/conformer/
```

### 2.5 Qwen3-ASR 1.7B (`qwen3asr/`)  → `models/qwen3asr`

```bash
# LoRA on the thinker's attention (same recipe as Voxtral: r16, lr 1e-4, accum 8, eval every 250 on miami_Tval600, patience 2).
# Targets keep the model's own "language X<asr_text>" prefix: the base model first tags every training clip's language.
$PY qwen3asr/kf_qwen3asr.py AB 2500 250 2                      # -> $T/kfold/qwen3asr_AB/best
#   log: early stop at step 1000, best step 500 (selection WER 0.2429), 0.9 h
$PY qwen3asr/merge_qwen_ab.py $T/kfold/qwen3asr_AB/best models/qwen3asr    # merged bf16, greedy generation config
```

### 2.6 Cohere Transcribe (`cohere/`)  → `models/cohere`

```bash
export KF_VAL=miami_Tsel600 KF_TAG=_frag KF_MANIFEST=manifest_frag.jsonl KF_REFS=refs_frag.csv KF_EXTRA_FRAC=0 KF_AUG=0
# full fine-tune of the 2B conformer encoder-decoder; fp32 master weights under bf16 autocast, AdamW, BatchNorm statistics
# frozen; targets = the model's own Spanish decoder prompt + the Miami text + EOS; clips 1-35 s (15,954); lr 1e-5, batch 8 x
# accum 2, warm-up 100, eval every 250 on miami_Tsel600 (looping clips collapsed), patience 4.  Early stop at 2000, best = step 1000 (0.2161), 46 min.
$PY cohere/cohere_train.py <cohere_snapshot> $T/cohere/AB_frag_lr1e5 --group AB --lr 1e-5 --seed 1 --patience 4
$PY cohere/save_bf16.py $T/cohere/AB_frag_lr1e5/best models/cohere          # bf16 copy with the remote-code files
```

`fold_chain.sh` / `frag2_chain.sh` are the launch chains that ran this and the fold / seed / concat variants
that were measured and rejected (see the write-up).

### 2.7 Parakeet Ultra (`parakeet/`)  → `models/parakeet`

```bash
export KF_VAL=miami_Tsel600 KF_TAG=_frag KF_MANIFEST=manifest_frag.jsonl KF_REFS=refs_frag.csv
# full fine-tune of ParakeetForTDT (0.6B) in the transformers-5 venv; fp32 masters + bf16 autocast; clips <= 30 s;
# lr 5e-5, batch 8 x accum 2, warm-up 100, eval every 200 on miami_Tsel600, patience 3.
# run_parakeet_train.sh sets HF_HOME / PYTHONPATH and calls parakeet_train.py in compat_venv.
sh parakeet/run_parakeet_train.sh $T/parakeet/ultra_miami_lr5e5 --init moondream/parakeet-ultra --lr 5e-5 --group AB
#   the step-1000 checkpoint was the best on miami_Tsel600 and was copied to $T/parakeet/lr5e5_step1000
$PY parakeet/parakeet_bf16.py $T/parakeet/lr5e5_step1000 models/parakeet     # bf16 copy (needs vendor_tf on sys.path, see the script)
```

### 2.8 Phi-4-multimodal (`phi4/`)  → `models/phi4` + `models/phi4_delta`

```bash
export KF_VAL=miami_Tsel600 KF_TAG=_frag KF_MANIFEST=manifest_frag.jsonl KF_REFS=refs_frag.csv KF_EXTRA_FRAC=0 KF_AUG=0
# trains the speech LoRA (r320), the audio encoder and the speech projection (915M parameters, fp32 masters, bf16
# autocast); the 3.8B language model and all vision parts stay frozen; lr 4e-5, batch 8 x accum 2, warm-up 100,
# eval every 250 on miami_Tsel600, patience 3.  Early stop at 2250, best = step 1500 (0.2433), 1.75 h. Saves only the trained tensors as delta.safetensors.
env PYTHONPATH=/home/jovyan/work/Enes/phi_extra $PY phi4/phi4_train.py <ZFTurbo_snapshot> $T/phi4/AB_frag
# packaging: write the delta into the base shards (bit-verified), zero-fill the unused image tower so it compresses away,
# and keep a one-tensor delta file so the inference stage is unchanged.
ZERO_VISION=1 $PY phi4/merge_phi4.py <ZFTurbo_snapshot> $T/phi4/AB_frag/best/delta.safetensors models/phi4 models/phi4_delta
```

`phi4_chain.sh` is the launch chain (training + the decode of dev, Miami half and long set that qualified it as a voter).

## 3. Validation (`validation/`)

- `score.py`: the organizers' scorer (normalizer + corpus WER), unchanged. `python score.py refs.csv --predicted-path submission.csv`.
- `kf_score.py`, `vote_any.py`, `xeval.py`: score a set, vote any combination of member CSVs with the shipped `vote.py`
  (fixed weights), and the candidate-voter evaluation on the three sets (dev, glued long, Miami half) with weights fixed in advance.
- `make_tta.py`: speed-perturbed copies of the validation sets (the inference-time `stage_tta.py` does the same for the test audio).
- `verify_v9_run.py`: checks a finished package run (every member present and complete, delta merged, chunked vote ran, no transcript text in the job log).
- `kf_queue.sh`, `kf_final_queue.sh`, `zs_one.sh`, `cand_zs.py`: the queues that launched the fold trainers and the candidate decodes.

Shipped scores (packaged pipeline, `score.py`): dev 0.0751, glued long set 0.0814, Miami half 0.1799. Noise floor: vLLM
decoding is not bit-reproducible run to run, about 0.001 WER on dev.
