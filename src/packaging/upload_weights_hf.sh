#!/bin/bash
# Upload the shipped model tree (the models/ folder of the final submission package, about 71 GB) to one public
# Hugging Face model repo, keeping the models/<member>/ layout that inference/main.py and download_weights.py expect.
#   bash upload_weights_hf.sh <package_dir> <hf_repo_id>      # <package_dir> is the folder that CONTAINS models/
# Resumable: hf upload-large-folder keeps its state in <package_dir>/.cache/huggingface and skips finished files.
set -eu
PKG=$1; REPO=$2
hf repo create "$REPO" --type model -y 2>/dev/null || true
hf upload-large-folder "$REPO" "$PKG" --repo-type model --include "models/**" --num-workers 8
