#!/bin/sh
# usage: whisper_full.sh <gpu> <seed>
# Our own Whisper large-v3 Miami LoRA on all of A+B (same recipe as the other agent's run, different seed).
# Every saved checkpoint is merged in fp32 and scored on the selection half of T (loops collapsed); the best
# one is kept as whisper_folds/full_s<seed>, then decoded on T and on dev (dev via long-clip splitting).
set -u
GPU=$1; SEED=$2
T=/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track
R=$T/kfold; W=$R/whisper_folds; HYP=$R/hyps; L=$R/logs
K=/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/miami_kfold
E=/home/jovyan/work/Enes; BASE=$E/lost_in_transcription/models/checkpoints/whisper-large-v3
PY=/kaggle/working/omni_rt/venv/bin/python; KF=/kaggle/working/omni_rt/kfold
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd /kaggle/working/omni_rt/work
say() { echo "$(date +%T) [whisper full s$SEED] $*"; }
RUN=$W/runfull_s$SEED
if [ ! -f $RUN/TRAIN_OK ]; then
  say "train 3 epochs on A+B, seed $SEED"
  $PY -u $E/lost_in_transcription/analysis/spanglish/train_whisper_miami.py --model $BASE --processor $BASE \
     --train-manifest $E/miami_windows/train.jsonl --val-manifest $E/miami_windows/val.jsonl --out-dir $RUN \
     --language es --epochs 3 --batch-size 8 --grad-accum 2 --lr 1e-4 --warmup 100 --lora --lora-rank 32 \
     --gradient-checkpointing --seed $SEED > $L/train_whisperfull_s$SEED.log 2>&1 && touch $RUN/TRAIN_OK || { say "TRAIN FAILED"; exit 1; }
fi
best=""; bestw=9; i=0
for step in $(ls -d $RUN/checkpoint-* | sed "s/.*checkpoint-//" | sort -n); do
  M=$W/mfull_s${SEED}_$i
  $PY -u $KF/whisper_merge.py $RUN $M $i > $L/mergefull_s${SEED}_$i.log 2>&1 || { say "MERGE FAILED step $step"; i=$((i+1)); continue; }
  $PY -u $KF/kf_decode_whisper.py $M $K/miami_Tsel600 $W/selfull_s${SEED}_$i.csv 16 > $L/decfull_s${SEED}_sel_$i.log 2>&1
  w=$($PY $KF/whisper_pick.py $W/selfull_s${SEED}_$i.csv 2>/dev/null | tail -1)
  case "$w" in 0.[0-9]*) ;; *) say "SCORING FAILED step $step: [$w]"; rm -rf $M; i=$((i+1)); continue;; esac
  say "step $step: selection-half WER $w"
  if [ "$(echo "$w $bestw" | awk '{print ($1 < $2)}')" = "1" ]; then [ -n "$best" ] && rm -rf $best; best=$M; bestw=$w; else rm -rf $M; fi
  i=$((i+1))
done
[ -n "$best" ] || { say "NO USABLE CHECKPOINT"; exit 1; }
say "chosen $(basename $best) at $bestw  (other agent's model on the same clips: 0.2810)"
rm -rf $W/full_s$SEED; mv $best $W/full_s$SEED
$PY -u $KF/kf_decode_whisper.py $W/full_s$SEED $K/miami_T $HYP/whisperS${SEED}_full_onT.csv 16 > $L/decfull_s${SEED}_T.log 2>&1 && say "decoded T"
$PY -u $KF/kf_decode_whisper.py $W/full_s$SEED /kaggle/working/omni_rt/dev_seg $HYP/whisperS${SEED}_full_ondev_pieces.csv 16 > $L/decfull_s${SEED}_dev.log 2>&1 \
  && $PY /kaggle/working/omni_rt/seg_long.py merge /kaggle/working/omni_rt/dev_seg $HYP/whisperS${SEED}_full_ondev_pieces.csv $HYP/whisperS${SEED}_full_ondev.csv && say "decoded dev"
say "WHISPER_FULL_DONE"
