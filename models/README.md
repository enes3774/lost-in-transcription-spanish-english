# models/

The ten trained ensemble members live here, one folder per member, exactly as `src/inference/main.py` expects:

```
models/omni  models/omni_delta  models/voxtral  models/whisper  models/conformer
models/qwen3asr  models/cohere  models/parakeet  models/phi4  models/phi4_delta
```

They are not in git (74.6 GB). Download them from the public Hugging Face repo
[enesssssw/lost-in-transcription-spanish-english-ensemble](https://huggingface.co/enesssssw/lost-in-transcription-spanish-english-ensemble):

```bash
pip install -U huggingface_hub
python src/inference/download_weights.py          # resumable; --only omni,whisper for a subset
```

`src/inference/run_inference.sh` links `src/inference/models` to this folder on its first run (the as-run code reads
`models/` next to `main.py`). Set `OMNI_MODELS_DIR` to use a different location.

What each folder contains, its base model and license: see the model card on the Hub (`src/packaging/HF_MODEL_CARD.md`)
and the "Licenses and attribution" section of the top-level README. How each one was trained: `src/training/README.md`.
