---
language:
  - es
  - en
tags:
  - automatic-speech-recognition
  - code-switching
  - ensemble
  - lost-in-transcription
license: other
license_name: mixed-per-member
license_link: https://github.com/drivendataorg/competition-winners
---

# Lost in Transcription: Spanish-English — 2nd place ensemble weights (team WAXAL)

The ten fine-tuned ensemble members of the 2nd-place solution to the Mozilla Data Collective
[Lost in Transcription: Spanish-English](https://competitions.mozilladatacollective.com/competitions/2/lost-in-transcription-sp-en/)
track (private WER 0.0998, public 0.0993). All members were fine-tuned on the **Bangor Miami** Spanish-English
code-switching corpus only (folds A + B, speaker-disjoint from validation); the competition dev set was never trained on.

The inference code, training scripts and the full write-up are on GitHub:
[github.com/enes3774/lost-in-transcription-spanish-english](https://github.com/enes3774/lost-in-transcription-spanish-english)
(the same content as `solution_code.zip` in this repo, which was delivered to the organizers). `python src/inference/download_weights.py --repo <this repo>` pulls this tree into `models/`.

| folder | member | base model | base license | what is inside |
| --- | --- | --- | --- | --- |
| `models/omni` | Qwen3-Omni-30B-A3B thinker | `cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit` + `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Apache-2.0 | hybrid checkpoint: AWQ int4 experts, bf16 attention, talker/code2wav removed (21 GB). **Not fine-tuned itself**; the Miami LoRA is in `omni_delta` |
| `models/omni_delta` | Miami LoRA for Omni | — | Apache-2.0 | `delta.pt`: LoRA r16 (attention q/k/v/o) trained 1 epoch on Miami A+B, merged into the engine at start-up |
| `models/voxtral` | Voxtral-Small-24B | `RedHatAI/Voxtral-Small-24B-2507-FP8-dynamic` (Mistral layout) | Apache-2.0 | FP8 base **as published** + `lora/` = our Miami LoRA r16 adapter (peft) |
| `models/whisper` | Whisper large-v3 | `openai/whisper-large-v3` | Apache-2.0 | Miami LoRA r32 merged, fp16 |
| `models/conformer` | Conformer-CTC large es-en code-switched | NGC `stt_enes_conformer_ctc_large_codesw` | NGC Terms of Use | full fine-tune, plain PyTorch state dict + SentencePiece model (loaded by `conformer_port.py`) |
| `models/qwen3asr` | Qwen3-ASR-1.7B | `Qwen/Qwen3-ASR-1.7B` | Apache-2.0 | Miami LoRA merged, bf16 |
| `models/cohere` | Cohere Transcribe 2B | `CohereLabs/cohere-transcribe-03-2026` | Apache-2.0 | full fine-tune, bf16, with the model's remote code |
| `models/parakeet` | Parakeet Ultra 0.6B TDT | `moondream/parakeet-ultra` (+ `nvidia/parakeet-tdt-0.6b-v3` processor) | CC-BY-4.0 | full fine-tune, bf16 |
| `models/phi4` | Phi-4-multimodal-instruct | `ZFTurbo/Phi-4-multimodal-instruct` (= `microsoft/Phi-4-multimodal-instruct` weights + a generate fix) | MIT | base shards with our speech LoRA / audio encoder / speech projection fine-tune written in; the unused image tower is zero-filled |
| `models/phi4_delta` | — | — | MIT | one-tensor delta kept so the inference stage is unchanged |

Decoding, voting and the writing rules are in the solution code; the members are meant to be used together
(word-level weighted ROVER vote, weights 20/20/16/16/16/10/8/8/8/8). Single-member dev WERs range from 0.090 (Whisper) to 0.212 (Conformer); the vote reaches 0.075.

Training data: Bangor Miami corpus (GPL-3.0-or-later), as distributed on the Mozilla Data Collective. Please cite
Deuchar, M., Davies, P., Herring, J. R., Parafita Couto, M. C. & Carter, D. (2014), *Building bilingual corpora*, in
Thomas & Mennen (eds.), *Advances in the Study of Bilingualism*, Multilingual Matters.

Each fine-tuned member inherits the license of its base model (column above); the Conformer and Cohere members require
accepting the respective base model's terms. Authors (team WAXAL): Enes Kulak (enesmahmut3774@gmail.com) and Roman (ZFTurbo, https://github.com/zfturbo).
