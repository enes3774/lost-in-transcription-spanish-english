#!/bin/sh
# phi4_chain.sh <gpu> : Miami A+B fine-tune of Phi-4-multimodal (speech LoRA + audio encoder), then decode dev + Miami half + long
S=/kaggle/working/omni_rt; T=/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track
X=$T/xexp; OD=$T/phi4/AB_frag${PTAG:-}; E=/home/jovyan/work/Enes/phi_extra
SNAP=$(ls -d /home/jovyan/work/Enes/lost_in_transcription/cache/huggingface/hub/models--ZFTurbo--Phi-4-multimodal-instruct/snapshots/*)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export KF_VAL=miami_Tsel600 KF_TAG=_frag KF_MANIFEST=manifest_frag.jsonl KF_REFS=refs_frag.csv KF_EXTRA_FRAC=0 KF_AUG=0
mkdir -p $T/phi4; cd $S/work
echo "$(date +%T) [$1] phi4 train start ${PARGS:-}" >> $X/night.log
env PYTHONPATH=$E $S/venv/bin/python -u $S/kfold/phi4_train.py $SNAP $OD ${PARGS:-} > $OD.log 2>&1
echo "$(date +%T) [$1] phi4 train exit $? | $(grep -E 'DONE|early stop' $OD.log | tail -1)" >> $X/night.log
[ -f $OD/best/delta.safetensors ] || { echo "$(date +%T) phi4: no best checkpoint" >> $X/night.log; exit 1; }
TG=phi4_ft${PTAG:-}; mkdir -p $X/$TG; [ -d $X/$TG/seg ] || cp -al $X/cohere_seed2/seg $X/$TG/seg
PHI4_DELTA=$OD/best/delta.safetensors EXTRA=$E sh $S/kfold/zs_one.sh $1 phi4 $SNAP $TG omni 8
