#!/bin/sh
# usage: kf_final_queue.sh <gpu> <member> [max_steps]
# Final member: train on A+B, early stop on the selection half of T (miami_Tsel600), decode full T and dev.
set -u
GPU=$1; MEM=$2
KF=/kaggle/working/omni_rt/kfold
T=/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track
R=$T/kfold; D=/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/miami_kfold
V4=$T/submission_v4; Q=$T/quant; DEV=/kaggle/working/omni_rt/dev
PY=/kaggle/working/omni_rt/venv/bin/python; NEMOPY=/kaggle/working/verify_venv/bin/python
HYP=$R/hyps; L=$R/logs
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_CACHE_ROOT=/kaggle/working/omni_rt/vllm_cache
export KF_VAL=miami_Tsel600 KF_TAG="" KF_EXTRA_FRAC=0 KF_AUG=0
cd /kaggle/working/omni_rt/work
say() { echo "$(date +%T) [$MEM AB] $*"; }
say "train AB"
case $MEM in
  voxtral)   $PY $KF/kf_voxtral.py AB ${3:-2500} 250 2 > $L/train_${MEM}_AB.log 2>&1 ;;
  qwen3asr)  $PY $KF/kf_qwen3asr.py AB ${3:-2500} 250 2 > $L/train_${MEM}_AB.log 2>&1 ;;
  conformer) CUDA_LAUNCH_BLOCKING=1 $NEMOPY $KF/kf_conformer.py AB ${3:-16000} 500 3 > $L/train_${MEM}_AB.log 2>&1 ;;
esac
grep -hE "training items|DONE|early stop" $L/train_${MEM}_AB.log | tail -3
for tgt in T dev; do
  if [ $tgt = dev ]; then DD=$DEV; else DD=$D/miami_T; fi
  out=$HYP/${MEM}_trainAB_on$tgt.csv
  say "decode on $tgt"
  case $MEM in
    voxtral)
      S=$R/vox_AB/serve; mkdir -p $S
      for x in consolidated.safetensors params.json tekken.json; do ln -sf $Q/vox_fp8/$x $S/$x; done
      ln -sfn $R/vox_AB/best $S/lora
      VOX_BACKEND=vllm $PY $V4/stage_voxtral.py $S $DD $out 0.85 > $L/dec_${MEM}_AB_$tgt.log 2>&1 ;;
    qwen3asr)  $PY $KF/kf_decode_qwen3asr.py $R/qwen3asr_AB/best $DD $out 4 > $L/dec_${MEM}_AB_$tgt.log 2>&1 ;;
    conformer) CUDA_LAUNCH_BLOCKING=1 $PY $V4/stage_conformer.py $R/conformer_AB/serve $DD $out > $L/dec_${MEM}_AB_$tgt.log 2>&1 ;;
  esac
  if [ $? -eq 0 ]; then touch $out.done; say "decoded $tgt"; else say "DECODE FAILED $tgt"; fi
done
say "FINAL_QUEUE_DONE"
