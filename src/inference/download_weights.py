"""Download the ten trained ensemble members from the Hugging Face Hub into <repo root>/models/ (the folder src/inference/run_inference.sh links to).

usage:  python download_weights.py [--repo <hf_repo_id>] [--models-dir models] [--only omni,whisper]

After this, models/ holds exactly the tree main.py expects (run_inference.sh links src/inference/models -> ../../models):
    models/omni models/omni_delta models/voxtral models/whisper models/conformer models/qwen3asr
    models/cohere models/parakeet models/phi4 models/phi4_delta
(about 75 GB unpacked). Re-running is incremental: files already present are skipped.
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO = "enesssssw/lost-in-transcription-spanish-english-ensemble"
MEMBERS = ("omni", "omni_delta", "voxtral", "whisper", "conformer", "qwen3asr", "cohere", "parakeet", "phi4", "phi4_delta")

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=DEFAULT_REPO)
ap.add_argument("--models-dir", default=str(Path(__file__).resolve().parents[2] / "models"))
ap.add_argument("--only", default=None, help="comma-separated subset of member folders")
a = ap.parse_args()
want = a.only.split(",") if a.only else list(MEMBERS)
bad = [m for m in want if m not in MEMBERS]
if bad:
    sys.exit(f"unknown member(s) {bad}; choose from {MEMBERS}")
out = Path(a.models_dir)
if out.name != "models":
    sys.exit("--models-dir must be a folder named models (the Hub repo stores models/<member>/ and main.py reads that name)")
out.mkdir(parents=True, exist_ok=True)
# the repo stores every member under models/<name>/ ; download straight into the target tree (no symlinked cache)
snapshot_download(a.repo, repo_type="model", local_dir=str(out.parent),
                  allow_patterns=[f"models/{m}/**" for m in want])
missing = [m for m in want if not (out / m).is_dir()]
if missing:
    sys.exit(f"download finished but these member folders are missing: {missing}")
print("weights ready:", ", ".join(want))
