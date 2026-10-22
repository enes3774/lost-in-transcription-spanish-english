#!/bin/sh
# v9c = v9b (all ten voters incl. Phi-4, quién rule) made SMALLER without changing what it computes:
#   - the Phi-4 Miami delta is merged into the base shards, Phi-4's unused image tower / vision LoRA are zero-filled
#     (merge_phi4.py, every tensor verified bit by bit), a one-tensor delta keeps the stage code unchanged;
#   - the zip is DEFLATED (-6) instead of stored: Voxtral FP8 shrinks 18 %, Phi-4 20 %, Omni 6 %.
# Code files are byte-identical to v9b. usage: v9c_build.sh folder | zip | val <gpu>
set -u
T=/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track
S=/kaggle/working/omni_rt; PY=$S/venv/bin/python; SCORE=/kaggle/working/verify_venv/bin/python; K=$S/kfold/v9
D=$T/submission_v9c; A=$T/submission_v9b; P=$T/phi4_shrunk; LOG=$T/build_v9c.log
say() { echo "$(date +%T) [v9c $1] $2" >> $LOG; }
case "$1" in
folder)
  grep -q "MERGE_OK" $P/merge.log || { say folder "FAILED: merge not finished"; exit 1; }
  [ -e $D ] && { say folder "FAILED: $D exists"; exit 1; }
  cp -al $A $D || { say folder "FAILED: hardlink copy"; exit 1; }
  rm -rf $D/models/phi4 $D/models/phi4_delta $D/vendor_tf; find $D -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
  mv $P/phi4 $D/models/phi4 && mv $P/phi4_delta $D/models/phi4_delta || { say folder "FAILED: move"; exit 1; }
  [ "$(ls $A/models/phi4 | wc -l)" = "$(ls $D/models/phi4 | wc -l)" ] || { say folder "FAILED: phi4 file count differs"; exit 1; }
  [ "$(stat -L -c %s $A/models/phi4_delta/delta.safetensors)" = "1830596760" ] || { say folder "FAILED: v9b delta changed"; exit 1; }
  for f in $(ls $A/models/phi4 | grep -v '^model-0'); do cmp -s $A/models/phi4/$f $D/models/phi4/$f || { say folder "FAILED: $f differs"; exit 1; }; done
  say folder "differences v9b -> v9c: $(diff -rq $A $D | sed "s#$T/##g" | tr '\n' ';' | cut -c1-600)"
  say folder "$(tail -1 $P/merge.log | cut -c1-200)"
  say folder "FOLDER_DONE content bytes $(find -L $D -type f -printf '%s\n' | awk '{s+=$1} END {print s}')"
  ;;
zip)
  Z=$T/submission_v9c.zip
  [ -e $Z ] && { say zip "FAILED: $Z exists"; exit 1; }
  [ -d $D/vendor_tf ] && { say zip "FAILED: vendor_tf dir in the package folder"; exit 1; }
  [ -n "$(find $D -name '__pycache__' | head -1)" ] && { say zip "FAILED: __pycache__ in the package folder"; exit 1; }
  say zip "start (deflate -6)"
  cd $D && zip -r -6 -q $Z . -x "*/__pycache__/*" "*.pyc" "*/.cache/*" || { say zip "FAILED: zip"; exit 1; }
  say zip "ZIP_OK size $(ls -l $Z | awk '{printf "%.3f GB (%d bytes)", $5/1e9, $5}') (v8 zip that ran: 63412004300 bytes; v9b zip that hung: 76414326730)"
  cd $T && unzip -tq submission_v9c.zip > $T/build_v9c_ziptest.log 2>&1; rc=$?
  echo "ZIPTEST_EXIT $rc" >> $T/build_v9c_ziptest.log
  [ "$rc" = "0" ] || { say zip "FAILED: unzip -t exit $rc"; exit 1; }
  say zip "TEST_OK ($(tail -2 $T/build_v9c_ziptest.log | head -1))"
  $PY $K/cmp_zips_v9c.py > $T/build_v9c_zipcmp.log 2>&1; rc=$?
  cut -c1-400 $T/build_v9c_zipcmp.log >> $LOG
  [ "$rc" = "0" ] || { say zip "FAILED: zip comparison"; exit 1; }
  say zip "ALL_DONE"
  ;;
val)
  GPU=$2; DV=$T/submission_v9c_val; V=$T/build_v9c_val
  [ -e $DV ] && { say val "FAILED: $DV exists"; exit 1; }
  say val "waiting for the GPU (the v8c dev run uses GPU $GPU first)"
  while ! grep -q "VAL_DONE\|\[v8c val\] FAILED" $T/build_v8c.log; do sleep 15; done
  cp -al $D $DV || { say val "FAILED: hardlink copy"; exit 1; }
  rm -rf $V; mkdir -p $V
  export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_CACHE_ROOT=$S/vllm_cache
  cd $S/work; t=$(date +%s)
  say val "full dev run from $DV on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU env -u OMNI_MODELS_DIR -u OMNI_DELTA -u OMNI_LIMIT -u PYTHONPATH OMNI_DATA_DIR=$S/dev OMNI_OUT_PATH=$V/submission.csv OMNI_WORK_DIR=$V/work OMNI_TIME_BUDGET_S=7200 $PY -u $DV/main.py > $V/run.log 2>&1
  rc=$?
  say val "dev run exit $rc in $(( $(date +%s) - t ))s, job log $(wc -l < $V/run.log) lines"
  [ "$rc" = "0" ] || { tail -8 $V/run.log >> $LOG; say val "FAILED: dev run"; exit 1; }
  $PY $S/kfold/verify_v9_run.py $V $S/dev 5 > $V/verify.log 2>&1; say val "verify_v9_run exit $? ($(grep -c PASS $V/verify.log) PASS, $(grep -c FAIL $V/verify.log) FAIL)"
  W=$($SCORE $S/score.py $S/dev/refs.csv --predicted-path $V/submission.csv | tail -1 | grep -oE "0\.[0-9]+")
  PW=$($SCORE $S/score.py $S/dev/refs.csv --predicted-path $V/work/phi4.csv | tail -1 | grep -oE "0\.[0-9]+")
  PW0=$($SCORE $S/score.py $S/dev/refs.csv --predicted-path $T/build_v9b_val/work/phi4.csv | tail -1 | grep -oE "0\.[0-9]+")
  SAME=$($PY -c "
import csv
a = {r['audio_filename']: r['transcript'] for r in csv.DictReader(open('$V/work/phi4.csv', encoding='utf-8'))}
b = {r['audio_filename']: r['transcript'] for r in csv.DictReader(open('$T/build_v9b_val/work/phi4.csv', encoding='utf-8'))}
print(sum(1 for k in b if a.get(k) == b[k]), len(b))")
  say val "PACKAGED v9c dev WER = $W (v9b packaged 0.0739); Phi-4 member: $PW vs $PW0 in the v9b run, identical transcripts $SAME; 'quien' in the output: $(grep -ciw 'quien' $V/submission.csv)"
  grep -E "pre-flight|stage [a-z0-9_]+ (exit|TIMEOUT)|v9 done|ABORT|\[phi4\]" $V/run.log | cut -c1-150 >> $LOG
  say val "VAL_DONE"
  ;;
esac
