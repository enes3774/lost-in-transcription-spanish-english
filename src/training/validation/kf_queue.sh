#!/bin/sh
# usage: kf_queue.sh <gpu> <member: voxtral|qwen3asr|conformer|omni>
# Train fold A -> decode B and T -> train fold B -> decode A. Omni (base, no training): decode A, B, T.
# Every step is skipped when its output already exists, so the queue can be restarted after a crash.
set -u
GPU=$1; MEM=$2
KF=/kaggle/working/omni_rt/kfold
T=/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track
R=$T/kfold; D=/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/miami_kfold
V4=$T/submission_v4; Q=$T/quant
PY=/kaggle/working/omni_rt/venv/bin/python; NEMOPY=/kaggle/working/verify_venv/bin/python
HYP=$R/hyps; mkdir -p $HYP $R/logs
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_CACHE_ROOT=/kaggle/working/omni_rt/vllm_cache
cd /kaggle/working/omni_rt/work
say() { echo "$(date +%T) [$MEM] $*"; }
other() { [ "$1" = "A" ] && echo B || echo A; }

decode() {  # fold, target group
  f=$1; g=$2; out=$HYP/${MEM}_train${f}_on${g}.csv
  if [ -f $out.done ]; then say "skip decode train$f on $g"; return; fi
  say "decode train$f on $g"
  case $MEM in
    voxtral)
      S=$R/vox_$f/serve; mkdir -p $S
      for x in consolidated.safetensors params.json tekken.json; do ln -sf $Q/vox_fp8/$x $S/$x; done
      ln -sfn $R/vox_$f/best $S/lora
      VOX_BACKEND=vllm $PY $V4/stage_voxtral.py $S $D/miami_$g $out 0.85 > $R/logs/dec_${MEM}_$f$g.log 2>&1 ;;
    qwen3asr)
      $PY $KF/kf_decode_qwen3asr.py $R/qwen3asr_$f/best $D/miami_$g $out 4 > $R/logs/dec_${MEM}_$f$g.log 2>&1 ;;
    conformer)
      CUDA_LAUNCH_BLOCKING=1 $PY $V4/stage_conformer.py $R/conformer_$f/serve $D/miami_$g $out > $R/logs/dec_${MEM}_$f$g.log 2>&1 ;;
  esac
  [ $? -eq 0 ] && touch $out.done && say "decoded -> $out" || say "DECODE FAILED train$f on $g"
}

train() {
  f=$1; log=$R/logs/train_${MEM}_$f.log
  if grep -q "^.*DONE best step" $log 2>/dev/null; then say "skip train $f"; return; fi
  say "train $f"
  case $MEM in
    voxtral)   $PY $KF/kf_voxtral.py $f 1500 250 2 > $log 2>&1 ;;
    qwen3asr)  $PY $KF/kf_qwen3asr.py $f 1500 250 2 > $log 2>&1 ;;
    conformer) CUDA_LAUNCH_BLOCKING=1 $NEMOPY $KF/kf_conformer.py $f 10000 500 3 > $log 2>&1 ;;
  esac
  grep -hE "EVAL|DONE|early stop" $log | tail -4
}

if [ "$MEM" = "omni" ]; then
  for g in T A B; do
    out=$HYP/omni_base_on$g.csv
    [ -f $out.done ] && { say "skip omni on $g"; continue; }
    say "omni base decode $g"
    OMNI_MODEL_DIR=$Q/omni_awq4 OMNI_MOE_BACKEND=auto OMNI_DATA_DIR=$D/miami_$g OMNI_OUT_PATH=$out OMNI_TIME_BUDGET_S=1000000 OMNI_GPU_UTIL=0.85 \
      $PY $V4/stage_omni.py > $R/logs/dec_omni_$g.log 2>&1 && touch $out.done && say "done $g" || say "OMNI DECODE FAILED $g"
  done
else
  train A; decode A B; decode A T
  train B; decode B A
fi
say "QUEUE_DONE"
