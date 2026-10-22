#!/bin/sh
# zs_one.sh <gpu> <backend> <model_dir> <tag> <venv: new|omni> [batch] : seg lab_all (>30 s cut like the package), zero-shot decode, merge, split
GPU=$1; B=$2; MD=$3; TAG=$4; V=$5; BS=${6:-8}
S=/kaggle/working/omni_rt; T=/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track
C=$T/submission_v8_code; PY=$S/venv/bin/python; X=$T/xexp; LOG=$X/night.log
if [ $V = new ]; then CPY="env PYTHONPATH=/home/jovyan/work/Enes/tf_new /kaggle/working/compat_venv/bin/python"; else CPY="env PYTHONPATH=${EXTRA:-} $PY"; fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd $S/work
O=$X/$TAG; mkdir -p $O; t=$(date +%s)
[ -f $O/seg/test_metadata.csv ] || $PY -c "import sys; sys.path.insert(0, '$C'); import seg_long; seg_long.split('${DATA:-$S/lab_all}', '$O/seg', 30.0, 28.0, 12.0)" > /dev/null 2>&1
$CPY $S/kfold/cand_zs.py $B $MD $O/seg $O/pieces.csv $BS > $O/dec.log 2>&1
rc=$?
$PY -c "import sys; sys.path.insert(0, '$C'); import seg_long; seg_long.merge('$O/seg', '$O/pieces.csv', '$O/all.csv')" > /dev/null 2>&1
$PY $S/kfold/split_sets.py $O/all.csv $O >> $O/dec.log 2>&1
echo "$(date +%T) [$GPU] $TAG: exit $rc in $(( $(date +%s) - t ))s | $(grep -h 'non-empty' $O/dec.log | tail -3 | tr '\n' ' ')" >> $LOG
