# Environments

Three Python environments were used. Inference needs only the first one; training used the other two for two members.

| name | used for | Python | key packages | spec file |
| --- | --- | --- | --- | --- |
| **competition runtime** | running `src/inference/main.py` (this is the organizers' image; the platform ran our submission in it) | 3.12 | torch 2.11 cu130, vLLM 0.23.0, transformers 4.57, peft, qwen-asr, soundfile, librosa, sentencepiece | `Dockerfile`, `pyproject.toml`, `uv.lock`, `apt.txt` (verbatim from the organizers' runtime repository, tag used 2026-10-01) |
| **main training venv** (`omni_rt/venv`) | fine-tuning Omni, Voxtral, Whisper, Qwen3-ASR, Cohere, Phi-4; all validation and voting; building the package | 3.12.14 | same torch/vLLM/transformers as the runtime, peft 0.20.0, accelerate 1.12, jiwer 4.0, mistral_common 1.11.7, qwen-asr 0.0.6 | `omni_venv_freeze.txt` (260 packages) |
| **compat venv** (`compat_venv`) | fine-tuning Parakeet (`ParakeetForTDT` needs transformers 5.x) | 3.10.12 | torch 2.8.0 cu128, transformers 5.15.1 | `compat_venv_freeze.txt` |
| **NeMo venv** (`verify_venv`) | fine-tuning the Conformer-CTC; running the official `score.py` | 3.10.12 | torch 2.8.0 cu128, nemo-toolkit 3.0.0, transformers 4.57.6, jiwer 4.0.0 | `verify_venv_freeze.txt` |

## Inference environment, from scratch

The exact environment is the organizers' Docker image. From a fresh Linux machine with an NVIDIA driver (CUDA 13 capable) and Docker:

```bash
git clone https://github.com/drivendataorg/lost-in-transcription-runtime   # organizers' runtime repository
cd lost-in-transcription-runtime/runtime
# the four files in this folder are the versions we ran against; copy them over if the upstream repo moved on
cp /path/to/this/solution/environment/{Dockerfile,pyproject.toml,uv.lock,apt.txt,.python-version} .
docker build --target runtime -t lit-runtime .
```

Without Docker, the same environment is one `uv sync` away:

```bash
sudo apt-get install -y $(cat apt.txt)
curl -LsSf https://astral.sh/uv/install.sh | sh
cp pyproject.toml uv.lock .python-version <workdir>/ && cd <workdir>
uv sync --frozen            # creates .venv with the locked versions (Python 3.12, torch cu130, vllm 0.23.0)
source .venv/bin/activate
```

Two libraries the runtime does not have are shipped inside `src/inference/` and need no installation:

- `src/inference/vendor_tf.zip`: transformers 5.17.0 + tokenizers 0.23.2 + huggingface_hub 1.28.0, used only by the Parakeet stage (`ParakeetForTDT` landed in transformers 5.6). `main.py` unpacks it next to itself on first run.
- `src/inference/vendor_phi/backoff`: the pure-Python `backoff` package the Phi-4 remote code imports.
- `src/inference/vendor/`: the vote library (`asr_ensemble`) and pure-Python Levenshtein / editdistance.

## Training environments

`omni_venv_freeze.txt` is the complete list; to rebuild it on Python 3.12:

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0
pip install vllm==0.23.0 transformers==4.57.6 peft==0.20.0 accelerate==1.12.0 qwen-asr==0.0.6 qwen-omni-utils==0.0.9 \
            mistral_common==1.11.7 jiwer==4.0.0 soundfile==0.14.0 librosa==1.0.0 soxr==1.1.0 sentencepiece==0.2.2 \
            safetensors==0.8.0 datasets==5.0.1 huggingface_hub==0.36.2
```

Parakeet (Python 3.10): `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128` then `pip install transformers==5.15.1 soundfile jiwer`.

Conformer (Python 3.10): `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128` then `pip install "nemo-toolkit[asr]==3.0.0" jiwer==4.0.0`.

## Hardware used

Training and validation ran on one workstation: Ubuntu 22.04.5 inside Docker (WSL2 host), 3 × NVIDIA RTX PRO 6000 Blackwell 96 GB + 1 × RTX A6000 48 GB, 8 usable CPU cores, 251 GB RAM, NVIDIA driver 582.08 / CUDA 13.0. Every training job ran on a single GPU. The largest single fine-tune (Qwen3-Omni LoRA, 65 GB of GPU memory) needs a card with more than 80 GB; every other member trains on 48 GB or less.

Inference was validated on the same cards and scored on the competition runner: one NVIDIA A100 80 GB, 120-minute limit, no network.
