#!/bin/sh
# Miami fine-tune of parakeet-ultra. Checkpoints chosen on miami_Tsel600 only, never on the competition dev set.
export HF_HOME=/home/jovyan/work/Enes/lost_in_transcription/cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-2} TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/jovyan/work/Enes/tf_new
export KF_VAL=miami_Tsel600
cd /kaggle/working/omni_rt/work
/kaggle/working/compat_venv/bin/python -u /kaggle/working/omni_rt/parakeet_train.py "$@"
