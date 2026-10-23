# Solution - Lost in Transcription: Spanish-English

Team: **WAXAL** — Enes Kulak (enesmahmut3774@gmail.com) and Roman (ZFTurbo, https://github.com/zfturbo)

**2nd place.** Private leaderboard WER **0.0998**, public 0.0993 (final submission `v9c`, 2026-10-01).
[Competition page](https://competitions.mozilladatacollective.com/competitions/2/lost-in-transcription-sp-en/) (Mozilla Data Collective, 2026):
verbatim transcription of Spanish-English code-switched voice notes, scored by word error rate (WER) after the organizers'
normalizer, inside a code-execution sandbox (one A100 80 GB, 120 minutes, no network).

Code: [github.com/enes3774/lost-in-transcription-spanish-english](https://github.com/enes3774/lost-in-transcription-spanish-english) ·
Weights: [huggingface.co/enesssssw/lost-in-transcription-spanish-english-ensemble](https://huggingface.co/enesssssw/lost-in-transcription-spanish-english-ensemble) (public, 74.6 GB).

The repository follows the [DrivenData prize-winner template](https://github.com/drivendataorg/prize-winner-template):
`src/` holds the code, `models/` the weights, `data/` the data; this README is organised as its `README_template.md`.

## Summary

Eight open speech models were each fine-tuned on the same two speaker-disjoint folds of the **Bangor Miami**
Spanish-English code-switching corpus (the only training data), decoded in a strict no-fallback pipeline, and merged word
by word with a weighted ROVER-style median vote; two extra Whisper passes on speed-perturbed audio act as test-time
augmentation. A few verified writing rules then map Miami's transcription conventions to the organizers'.

```
audio clip ──┬─► Qwen3-Omni + Miami LoRA (vLLM) ─────────────┐
             ├─► Voxtral Small FP8 + Miami LoRA (vLLM) ──────┤
             ├─► Whisper large-v3 + Miami LoRA ──────────────┤
             ├─► Whisper on 0.9× speed ──────────────────────┤      repetition guard
             ├─► Whisper on 1.1× speed ──────────────────────┼─►  chunking of clips > 30 s   ─►  writing rules  ─►  transcript
             ├─► Cohere Transcribe (full FT, 4 beams) ───────┤      word-level weighted median vote
             ├─► Parakeet Ultra (full FT) ───────────────────┤      majority gap-fill
             ├─► Qwen3-ASR + Miami LoRA (4 beams) ───────────┤
             ├─► Conformer-CTC (full FT) ────────────────────┤
             └─► Phi-4-multimodal + Miami speech FT ─────────┘
   (clips longer than a member's limit are cut at pauses before that member and re-joined after)
```

- **Data preprocessing.** The Miami CHAT transcripts were cleaned and cut into 22,290 single-speaker clips (24.1 h, 16 kHz),
  split by conversation into speaker-disjoint groups A / B (training) and T (600-clip early-stopping sample + an untouched
  1,449-clip validation half). A second label variant uses the competition's cut-off-word convention; Whisper trains on
  24 s windows of joined utterances. Inference does no preprocessing beyond reading the audio and cutting long clips at pauses.
- **Models and pretrained backbones.** Qwen3-Omni-30B-A3B (AWQ int4 + bf16 attention, LoRA), Voxtral-Small-24B FP8 (LoRA),
  Whisper large-v3 (LoRA, merged), Cohere Transcribe 2B (full fine-tune), Parakeet Ultra 0.6B (full), Qwen3-ASR-1.7B (LoRA,
  merged), NeMo Conformer-CTC es-en (full, ported to plain PyTorch), Phi-4-multimodal (speech LoRA + audio encoder).
  All fine-tunes on Miami A + B only. The competition dev set was never trained on and never used to pick a checkpoint.
- **Ensemble.** Yes: a word-level weighted median vote over ten voters (weights 20/20/16/16/16/10/8/8/8/8), chunked for clips
  over 30 s, followed by a majority gap-fill. Word level (instead of character level) was the single largest gain.
- **Post-processing of the output.** A repetition guard per member, then nine writing rules on the final text
  (`gonna → going to`, no `.!?` before `I`, `ah/er → uh`, underscores, `súper`, `quién`, ...), each verified on the dev
  references.

Every change shipped only if it was non-negative on three validation sets at once (official dev, a glued long-clip set,
the untouched Miami half) with weights fixed before measuring. Time spent: about three weeks (2026-09-10 to 2026-10-01),
two people plus an AI coding assistant. The full narrative (design decisions, measurements, everything that did not work)
is in [`docs/WRITEUP.md`](docs/WRITEUP.md); the day-by-day experiment log with every number is
[`docs/SPEN_EXPERIMENTS_ON_V7_2026-09-27.md`](docs/SPEN_EXPERIMENTS_ON_V7_2026-09-27.md). The approach in more detail
is in [Approach](#approach) below.

# Setup

Assumes a fresh Linux x86_64 machine with an NVIDIA driver that supports CUDA 13 (we used 582.08 / CUDA 13.0) and nothing else.

1. **Python 3.12** (we ran 3.12.14). Inference is defined by the organizers' runtime image; two routes:

   ```bash
   # a) Docker (the exact scored environment)
   git clone https://github.com/drivendataorg/lost-in-transcription-runtime && cd lost-in-transcription-runtime/runtime
   cp <this repo>/environment/{Dockerfile,pyproject.toml,uv.lock,apt.txt,.python-version} .   # the versions we ran against
   docker build --target runtime -t lit-runtime .

   # b) no Docker: the same locked environment with uv
   sudo apt-get install -y $(cat environment/apt.txt)        # ffmpeg, libsndfile, build tools
   curl -LsSf https://astral.sh/uv/install.sh | sh
   cp environment/{pyproject.toml,uv.lock,.python-version} . && uv sync --frozen && source .venv/bin/activate
   ```

   `requirements.txt` at the root is the complete pip freeze (260 packages) of the environment that trained six of the
   eight members and ran all validation: the runtime's packages plus the training extras. For a pip-only install:
   `pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 && pip install -r requirements.txt`.
   Two members train in Python 3.10 environments (`environment/compat_venv_freeze.txt` for Parakeet, transformers 5.15;
   `environment/verify_venv_freeze.txt` for the Conformer, nemo-toolkit 3.0). Details: [`environment/README.md`](environment/README.md).
2. **Non-Python tools:** `ffmpeg` and `libsndfile1` (in `environment/apt.txt`; `sudo apt-get install -y ffmpeg libsndfile1`),
   Docker with the NVIDIA container toolkit for route (a), `zip`/`unzip` for packaging. The Parakeet stage needs transformers 5.17,
   which is shipped as `src/inference/vendor_tf.zip` and unpacked by `main.py` on first run; nothing to install.
3. **Weights** (74.6 GB) from the Hugging Face Hub into `models/` (resumable, no token needed):
   `pip install -U huggingface_hub && python src/inference/download_weights.py`
4. **Data.** Inference: a folder in the competition layout (`clips/` + `test_metadata.csv`), see `data/README.md`.
   Training: the Bangor Miami download in `data/raw/miami/` and the competition dev set in `data/raw/dev/` (measurement only).

Expected layout before training or inference:

```
lost-in-transcription-spanish-english
├── data                       <- not in git, see data/README.md
│   ├── raw/miami              <- Bangor Miami corpus (chat/, audios/) — training only
│   ├── raw/dev                <- competition dev set — validation only
│   └── test                   <- clips/ + test_metadata.csv of the set to transcribe
├── models                     <- the ten member folders from the Hub (download_weights.py), see models/README.md
├── src
│   ├── inference              <- the exact code of the final submission (v9c) + 3 helpers
│   │   ├── main.py                entry point: pre-flight, ten members as subprocesses, vote, strict checks
│   │   ├── stage_*.py             one decoder per member (omni, voxtral, whisper, conformer, qwen3asr, cohere, parakeet, phi4, tta)
│   │   ├── vote.py                word-level weighted median vote, chunking for long clips, majority gap-fill
│   │   ├── spelling_rules.py      the writing rules applied once to the final text (self-test: python spelling_rules.py)
│   │   ├── seg_long.py            pause-based cutting of long clips and re-joining of the piece transcripts
│   │   ├── common.py, conformer_port.py   shared helpers; the NeMo Conformer ported to plain PyTorch
│   │   ├── vendor/, vendor_phi/, vendor_tf.zip   vendored libraries (vote library, Levenshtein, backoff, transformers 5.17)
│   │   ├── download_weights.py    pulls models/ from the Hub
│   │   ├── prepare_data.py        folder of audio files -> data dir in the competition layout
│   │   └── run_inference.sh       one-command run
│   ├── training               <- every script that produced the weights, as run, + its README.md
│   │   └── data_prep/ omni/ voxtral/ whisper/ conformer/ qwen3asr/ cohere/ parakeet/ phi4/ validation/
│   └── packaging              <- scripts that assembled, validated, zipped (60.6 GB) and uploaded the submission; HF model card
├── environment                <- organizers' runtime (Dockerfile, pyproject.toml, uv.lock, apt.txt) + the three venv freezes
├── docs                       <- write-up and experiment logs
├── README.md
├── requirements.txt           <- pip freeze of the main environment
└── LICENSE                    <- MIT for our code
```

# Hardware

Training and validation ran on one workstation: Ubuntu 22.04.5 (Docker on a WSL2 host), 3 × NVIDIA RTX PRO 6000 Blackwell
96 GB + 1 × RTX A6000 48 GB, 8 usable CPU cores, 251 GB RAM, driver 582.08 / CUDA 13.0. Every job used a single GPU.

| step | GPU | time |
| --- | --- | --- |
| data preparation (CHAT → clips, splits, Whisper windows, long set) | CPU | ~20 min |
| Qwen3-Omni LoRA (65 GB of GPU memory, needs a card > 80 GB) | 1 × 96 GB | 2.6 h |
| Voxtral LoRA | 1 × 96 GB | 2.4 h |
| Whisper LoRA + fp32 merge + checkpoint decodes | 1 × 48 GB | ~1.5 h |
| Conformer-CTC full fine-tune | 1 × 48 GB | 6.2 h |
| Qwen3-ASR LoRA + merge | 1 × 48 GB | 0.9 h |
| Cohere Transcribe full fine-tune | 1 × 48 GB | 0.8 h |
| Parakeet Ultra full fine-tune | 1 × 48 GB | ~1 h |
| Phi-4-multimodal speech fine-tune + merge | 1 × 48 GB | 1.75 h |
| validation decodes of the three sets, all members | 1 × 48 GB | ~2 h |
| **training total** | one 96 GB card | **~18 h** |
| inference, full test set | A100 80 GB (competition runner) | **~35 min** (Omni 3, Voxtral 5, Whisper 3 × 2, Phi-4 3, others < 1 each, vote 0.5; model loading ~8) |

Inference needs one GPU with at least 48 GB (validated on RTX PRO 6000 96 GB and RTX A6000 48 GB, scored on an A100 80 GB),
32 GB RAM and 160 GB of free disk (75 GB of weights, the vLLM cache, work files).

# Run training

`src/training/README.md` is the complete guide: for every member the base model, the exact command, the hyperparameters,
the selection set, the best step and the packaging step, plus the data pipeline and a map of the absolute paths the
scripts contain (they are the scripts **as run**, copied verbatim from the training machine; nothing was retyped).
Set the paths of that map (or recreate the folders) and run, from `src/training/`:

```bash
# data (once): CHAT -> clips, speaker-disjoint split A/B/T, selection sample, frag labels, Whisper windows, glued long set
python data_prep/miami_prep.py --cut --max_s 20 --gap_s 2; python data_prep/miami_split.py; python data_prep/make_groups.py
python data_prep/make_tsplit.py; python data_prep/miami_prep_frag.py; python data_prep/prep_miami_windows.py --max-seconds 24 ...; python data_prep/make_long_dev.py
# members (one GPU each; the two Python 3.10 members are marked)
python omni/omni_lora_train.py --run miami_r16_AB --miami 30000 --miami_names miamiAB_names.txt --seed 1 --epochs 1 --accum 8
python voxtral/kf_voxtral.py AB 2500 250 2
sh     whisper/whisper_full.sh <gpu> 2
python conformer/kf_conformer.py AB 16000 500 3                                       # NeMo venv
python qwen3asr/kf_qwen3asr.py AB 2500 250 2 && python qwen3asr/merge_qwen_ab.py
python cohere/cohere_train.py <snapshot> cohere/AB_frag_lr1e5 --group AB --lr 1e-5 --seed 1 --patience 4
sh     parakeet/run_parakeet_train.sh parakeet/ultra_miami_lr5e5 --init moondream/parakeet-ultra --lr 5e-5 --group AB   # transformers-5 venv
python phi4/phi4_train.py <snapshot> phi4/AB_frag
```

**Data sources and how each is used**

| dataset | source | role |
| --- | --- | --- |
| **Bangor Miami corpus** | [Mozilla Data Collective dataset page](https://datacollective.mozillafoundation.org/datasets/cmmccftl300ejmu072175nr00) (canonical home [bangortalk.org.uk](https://bangortalk.org.uk/), GPL-3.0-or-later; cite Deuchar et al. 2014). 56 informal Spanish-English conversations recorded in Miami, 84 speakers, about 35 h, CHAT transcripts with audio. | **The only training data.** Parsed to 22,290 clips. Speaker-disjoint groups A (9,203 clips) + B (10,430) train every member; group T (2,633) provides the 600-clip early-stopping sample and the untouched 1,449-clip validation half. |
| **Competition dev set** | the track's data download page (155 clips, 36 min, two speakers, with references) | **Measurement only**: never trained on, never used to pick a checkpoint. Also glued into 58 synthetic 60–220 s clips to validate long-clip handling. |

No other speech corpus, no synthetic speech, no public fine-tuned checkpoints. The test set was only ever seen by the platform.

**Weights produced.** Each trainer writes its run folder (checkpoints + selection WER log) under the track workspace, and a
packaging step (`whisper/whisper_merge.py` + `whisper_to_fp16.py`, `qwen3asr/merge_qwen_ab.py`, `cohere/save_bf16.py`,
`parakeet/parakeet_bf16.py`, `phi4/merge_phi4.py`, `omni/make_stage_omni_delta.py`, the Conformer state dict written by its
trainer) writes the member folder that goes into `models/<member>/`. Sizes on disk:

| folder | GB | folder | GB |
| --- | --- | --- | --- |
| `models/omni` (hybrid AWQ/bf16 base, not fine-tuned itself) | 21.8 | `models/omni_delta` (our LoRA) | 0.05 |
| `models/voxtral` (FP8 base as published + `lora/`, our adapter) | 28.5 | `models/whisper` | 3.1 |
| `models/phi4` (base shards with our fine-tune written in) | 11.2 | `models/phi4_delta` | < 0.01 |
| `models/cohere` | 4.1 | `models/qwen3asr` | 4.1 |
| `models/parakeet` | 1.3 | `models/conformer` | 0.5 |

74.6 GB in all (92 files).

**Trained weights are public**, so inference never needs retraining:
`python src/inference/download_weights.py` pulls all ten folders from
[huggingface.co/enesssssw/lost-in-transcription-spanish-english-ensemble](https://huggingface.co/enesssssw/lost-in-transcription-spanish-english-ensemble)
into `models/` (resumable; `--only omni,whisper` for a subset).

**Network access during training:** yes, once, to download the public base models from the Hugging Face Hub / NGC
(`Qwen/Qwen3-Omni-30B-A3B-Instruct` + `cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit`, `RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic`,
`openai/whisper-large-v3`, NGC `stt_enes_conformer_ctc_large_codesw`, `Qwen/Qwen3-ASR-1.7B`,
`CohereLabs/cohere-transcribe-03-2026` (gated: accept its terms), `moondream/parakeet-ultra` + `nvidia/parakeet-tdt-0.6b-v3`,
`ZFTurbo/Phi-4-multimodal-instruct`). Everything else is offline.

Fine-tuning is stochastic (data order, dropout, bf16 kernels); re-trained members landed within about ±0.005 member WER
and ±0.001 vote WER of the shipped ones in our seed experiments.

# Run inference

One entry point, `src/inference/run_inference.sh`, produces the submission from the raw competition data and the
downloaded weights with no retraining and no manual step. It works on any new data in the test-set format.

```bash
# data: the official test/dev download unzipped as data/test/clips/* + data/test/test_metadata.csv
#       (audio_filename,file_duration_seconds,language); for your own audio files:
python src/inference/prepare_data.py ~/my_voice_notes data/test      # wav/mp3/flac/ogg/m4a, any sample rate, any length

# run, Docker route (GPU 0, repo mounted at /solution)
docker run --rm --gpus all -v "$PWD":/solution lit-runtime \
    bash /solution/src/inference/run_inference.sh /solution/data/test /solution/out/submission.csv /solution/out/work 0
# run, uv route (environment activated)
bash src/inference/run_inference.sh data/test out/submission.csv out/work 0
```

`out/submission.csv` has the columns `audio_filename,transcript` (the competition's `submission_format.csv`).

- **Data used at inference:** only the audio in `clips/` and the durations in `test_metadata.csv` (durations steer batching
  and the long-clip cutting; an approximate value is fine). No reference data, no network (`HF_HUB_OFFLINE=1`).
- **Preprocessing of the test set:** none to run separately. `main.py` cuts clips longer than a member's limit (30/45/60 s)
  at the quietest pause before that member and re-joins the piece transcripts; the two speed-perturbed Whisper passes
  resample the audio on the fly.
- **Interim files** (work dir, 3rd argument, default `src/inference/work`): per-member transcripts `<member>.csv`
  (kept so any subset can be re-voted), the pause-cut pieces `seg_<member>/` and speed-perturbed audio `tta_*/` (a few
  hundred MB for the test set), the vote before the writing rules `vote_raw.csv`, every stage's full log in `logs/`, and the
  unpacked `src/inference/vendor_tf/` (3,482 files, 120 MB, created once). Plus the vLLM compile cache in `~/.cache/vllm`.
  Budget about 5 GB of disk for interim files on top of the 75 GB of weights.
- **Output:** written once, at the end, by the vote, to the 2nd argument (default when the `OMNI_*` variables are unset:
  `submission/submission.csv` relative to the working directory, exactly as the organizers' runner expected).
- **Runtime:** ~35 min for the full test set on an A100; about 8 min of that is model loading, so a single clip costs nearly as
  much as 600. The run is deterministic up to vLLM's kernel non-determinism (about ±0.001 WER on dev).

To score a run on a labelled set: `python src/training/validation/score.py refs.csv --predicted-path out/submission.csv`
(the organizers' scorer; needs `jiwer pandas typer`). One-model use: each `stage_*.py` is a stand-alone script
(`python src/inference/stage_whisper.py models/whisper data/test out.csv 16`) and `python src/inference/vote.py data/test out.csv name=csv ...`
combines any subset (`V8_STRICT=0` allows a partial ensemble; members are named as in `WEIGHTS` in `vote.py`).

On the platform the same code ran as `submission.zip` = the contents of `src/inference/` with `models/` inside it at the
archive root; `src/packaging/build_v9pkg.sh` and `v9c_build.sh` assembled, validated and zipped it. `run_inference.sh`
reproduces that layout by linking `src/inference/models` to `models/` (set `OMNI_MODELS_DIR` to point elsewhere).

**Troubleshooting**

- The run is strict: if any member fails or returns fewer than 90 % non-empty transcripts, no output file is written, the
  exit code is 1 and the last `ABORT:` line of the console names the cause; the member's full log is in `<work>/logs/<member>.log`.
- `pre-flight: missing ...` → a model folder is incomplete; re-run `download_weights.py` (incremental).
- vLLM out of memory on a 48 GB card → close other GPU processes; Omni (22 GB) and Voxtral (28 GB) each need the whole card.
  `VLLM_ATTENTION_BACKEND=TRITON_ATTN` is set by the script (FlashInfer/FlashAttention kernels differ across GPUs).
- `vendor_tf unpacked ...` appears once; if `src/inference/vendor_tf/` is half-unpacked after an interrupted run, delete it.
- Audio the stages cannot read (`soundfile`/`librosa` fail) aborts the pre-flight: convert to 16 kHz wav with ffmpeg.
- Stage timeouts are sized for the 120-minute sandbox (`OMNI_TIME_BUDGET_S`, default 7200 s in the script); raise it on a slower GPU.

---

## Approach

### The ten voters

Eight model families, all fine-tuned on Miami folds A + B, plus two speed-perturbed Whisper passes. "Alone" = dev WER of the member by itself after the writing rules; the vote reaches 0.0751.

| voter | base model | fine-tune (Miami A+B) | decoding | clips cut above | vote weight | alone |
| --- | --- | --- | --- | --- | --- | --- |
| omni | Qwen3-Omni-30B-A3B thinker; AWQ int4 experts with bf16 attention | LoRA r16 on attention q/k/v/o, lr 1e-4, 1 epoch; merged into the running vLLM engine at start-up, base weights kept for clips over 45 s | vLLM, greedy, whole clips | never | 20 | 0.0949 |
| voxtral | Voxtral-Small-24B, FP8 (RedHatAI) | LoRA r16 on attention, served as a vLLM LoRA adapter | vLLM, greedy, whole clips | never | 20 | 0.0951 |
| whisper | Whisper large-v3 | LoRA r32 (attention + MLP), 2 epochs on 24 s windows, merged in fp32 | HF pipeline, greedy, batch 16 | 30 s | 16 | 0.0907 |
| whisper_s090 | same weights | none | same, on audio slowed by 10 % | 30 s | 8 | 0.0901 |
| whisper_s110 | same weights | none | same, on audio sped up by 10 % | 30 s | 8 | 0.0907 |
| cohere | Cohere Transcribe (2B conformer enc-dec) | full fine-tune, lr 1e-5, best step 1000 | 4 beams, repetition guard | 30 s | 16 | 0.0919 |
| parakeet | Parakeet Ultra 0.6B TDT | full fine-tune, lr 5e-5, step 1000 | greedy, batch 16 | 30 s | 16 | 0.1039 |
| qwen3asr | Qwen3-ASR-1.7B | LoRA r16 on the thinker, merged | 4 beams | 45 s | 10 | 0.1059 |
| conformer | NeMo Conformer-CTC large, code-switched es-en | full fine-tune, about 6 epochs, ported to plain PyTorch | greedy CTC | 60 s | 8 | 0.2121 |
| phi4 | Phi-4-multimodal-instruct | speech LoRA r320 + audio encoder + speech projection (915M parameters), lr 4e-5, step 1500 | greedy, batch 8 | 30 s | 8 | 0.1334 |

A member earned its slot only by lowering the vote on all three validation sets at a weight fixed before the
measurement (8 for a new member, half the parent's weight for each speed copy). The weak voters still matter: Phi-4 at
0.133 helps because its errors are independent of the encoder-decoder ASR models and the Qwen-family audio-LLMs.

### Combination (`src/inference/vote.py`)

1. **Repetition guard** on every member: one word at most 4 times in a row, a 2–6-word phrase at most twice.
2. **Chunking for long clips.** The vote library (`asr_ensemble`, `median_extended`) builds candidates on the edit path between two members, so on a 300-word clip it cannot take one member's words in one sentence and another's in the next. Clips over 30 s are cut at anchor words (a word of the heaviest member that every member has at the aligned position), at least 30 words apart, voted chunk by chunk and re-joined; the cut is checked to be lossless. Glued long set 0.0874 → 0.0825.
3. **Weighted word-level median vote** with the weights above. Word level matters for code-switching: character-level merging can fuse a Spanish and an English word into a word that is neither. Switching from character to word level was the single largest gain of the competition (dev 0.0805 → 0.0775, Miami half 0.1776 → 0.1712).
4. **Majority gap-fill.** Every member is aligned to the vote; where members holding at least 60 % of the weight insert the same phrase in the same gap, it is inserted.

### Writing rules (`src/inference/spelling_rules.py`, applied once to the final text)

`gonna/wanna/gotta/kinda/ok → going to/want to/got to/kind of/okay`; no `. ! ?` directly before `I / I'm / I'll / I've / I'd`
(the scorer lower-cases the first letter after sentence punctuation); `ah/er → uh`; underscores (`Curious_George → Curious George`,
`T_V → TV`); `super → súper`; `quien → quién`. Each rule was measured against the dev references and, for the first
seven, measured on dev when each rule was added and recorded in the docstrings of `src/inference/spelling_rules.py`: five spelling rules, v5 five-way vote 0.0993 → 0.0897; casing guard 0.0897 → 0.0853; filler rule a further −0.0008 (the first seven together about −0.015); underscores neutral in the vote; `súper` 0.0773 → 0.0765 on the v8 vote; `quién` 0.0751 → 0.0747 on the v9 vote. The v4 → v5 leaderboard step (0.1190 → 0.1033) bundled the rules with member changes, so it is not an isolated measurement of them.

### Long clips (`src/inference/seg_long.py`)

Three members break on clips over 30 s (Whisper drops words at chunk borders, Qwen3-ASR truncates, the Conformer's
position table ends at 200 s). Every clip over a per-member limit is cut at the quietest 0.3 s inside a 12–28 s
window, the pieces are decoded by the unchanged member script and the texts re-joined in order. Omni and Voxtral are
never cut; the Omni LoRA switches to base weights above 45 s because it only saw short clips.

### Engineering

- **Strict, no-fallback `main.py`**: a pre-flight checks every model and audio file; each member must return at least 90 % non-empty clips; every stage has a timeout; the output file is written once, by the vote, at the end. Any miss aborts with no output file (the platform then refunds the submission slot) and an `ABORT:` line naming the cause.
- One subprocess per member (GPU memory fully released between models), full stage logs in files, only summary lines in the job log (the platform keeps 500 lines).
- Package size: the platform ran a 63 GB stored archive and hung on a 76 GB one. The final package merged the Phi-4 delta into its base shards, zero-filled Phi-4's unused image tower and was deflated instead of stored: 60.6 GB (74.6 GB unpacked), built by five parallel `zip` processes stitched without recompression (`src/packaging/`).

## Validation protocol and scores

A change shipped only if it was non-negative on three sets at once, with every weight and threshold fixed before the measurement.

| set | size | what it guards against | final (v9c) WER |
| --- | --- | --- | --- |
| competition dev | 155 clips, 2 speakers | the organizers' audio and writing style | 0.0751 |
| glued long set | 58 clips of 60–220 s built from dev | long-clip regressions (the long tail) | 0.0814 |
| Miami half (group T, held-out conversations) | 1,449 clips, speaker-disjoint from every fine-tune | overfitting to the two dev speakers | 0.1799 |

Dev gains reached the leaderboard at roughly a quarter to a third of their size. Noise floor: re-decoding one member moves the vote by about 0.0005; vLLM runs differ by about 0.001 on dev.

## Submission history

| version | date | public WER | what changed |
| --- | --- | --- | --- |
| v4 | Sep 15 | 0.1190 | 5 members (Omni, Voxtral, Qwen3-ASR, Conformer, Whisper), character-level vote; Omni AWQ + Voxtral FP8 to fit the size limit |
| v5 | Sep 20 | 0.1033 | Miami-trained members with no checkpoint chosen on dev, the five spelling rules, casing guard, filler rule, pause-splitting of long clips |
| v6 | Sep 23 | 0.1043 | v5 + an Omni "refiner" stage; scored worse, dropped |
| v7 | Sep 26 | 0.1016 | Cohere Transcribe as 6th voter, underscore rule, **word-level** vote, Parakeet as 7th voter; per-stage logs |
| v8 | Sep 28 | 0.0993 | Omni Miami LoRA delta, `súper` rule, gap-fill, Cohere beam 4, chunked vote, strict no-fallback mode |
| **v9c** | Oct 1 | **0.0993** (private **0.0998**, 2nd place) | Phi-4 Miami fine-tune and the Whisper speed pair as voters 8–10, `quién` rule; deflated 60.6 GB package |

## What did not work

Measured and rejected (each at a fixed weight, on all three sets); details and numbers in `docs/WRITEUP.md` and the experiment log.

- Extra model families as voters: Nemotron 3.5 ASR, VibeVoice, Granite-speech NAR, MOSS, canary, Gemma-4 audio, SeamlessM4T-v2, w2v-BERT CTC, omniASR-LLM 3B/7B, Voxtral Mini, Whisper turbo, three public Spanglish Whisper checkpoints. Every one neutral or worse on at least one set.
- Second copies of a member (Cohere seeds 2 and 3, Whisper seed and label-format siblings, Phi-4 seed 2, Qwen3-ASR and a second Whisper speed pair): Miami-only gains, dev flat.
- An Omni "refiner" that re-transcribes each clip given the audio and all member texts (v6): better on dev, worse on the board.
- Fine-tuning on the dev set (member and vote level): −0.0026 out-of-fold on dev, but it could not be checked on any held-out set that contains the dev speakers, so it was rejected as an overfitting risk.
- Smarter combiners: plain `median`, minimum-Bayes-risk selection, CTC beam + Miami trigram, LLM rewrite, per-clip oracle selection. All lose to the word-level vote.
- Full fine-tunes instead of LoRA (Qwen3-ASR, Whisper, Voxtral): better on Miami, worse on dev and long.
- More Miami (2-epoch Omni, checkpoint soups, concatenation-trained long clips, retraining on all of Miami), other data (CIEMPIESS mix, synthetic TTS): flat.
- Weight and threshold tuning, and text rules beyond the nine (dropping fillers, de-duplicating stutters, digits, casing after `¿`): within noise or coin flips.

## Licenses and attribution

- Our code (`src/inference/*.py` except the vendored libraries, `src/training/`, `src/packaging/`): MIT License, see `LICENSE`.
- Weights are derivatives of public base models and inherit their licenses: Qwen3-Omni, Qwen3-ASR, Voxtral, Whisper, Cohere Transcribe (gated on the Hub, accept its terms to download the base) — Apache-2.0; Phi-4-multimodal — MIT; Parakeet Ultra / parakeet-tdt-0.6b-v3 — CC-BY-4.0; NeMo Conformer-CTC es-en — NGC Terms of Use. Each member's folder on the Hub carries its base model's license.
- Training data: Bangor Miami corpus (GPL-3.0-or-later; Deuchar, M., Davies, P., Herring, J. R., Parafita Couto, M. C. & Carter, D. (2014). *Building bilingual corpora*. In Thomas & Mennen (eds.), *Advances in the Study of Bilingualism*, Multilingual Matters). All fine-tuning data is public.
- Vendored libraries: `asr_ensemble` (vote), `Levenshtein`/`editdistance` pure-Python ports, `backoff`, transformers 5.17 (Apache-2.0).

Authors (team WAXAL): Enes Kulak, enesmahmut3774@gmail.com, and Roman (ZFTurbo), https://github.com/zfturbo.
