#!/bin/sh
# Build + validate + zip the v8 package on ONE GPU (default 3). Everything on the HDD ($T), never on the SSD.
# v8 = v7 members (same weights, hardlinked from submission_v7/models) + Omni A+B delta + v8 code (súper, gap-fill,
# Cohere 4 beams, chunked vote, STRICT no-fallback, nested vendor_tf.zip). Zips only if both validation runs pass
# verify_v8_run.py (every v8 ingredient visibly active) and the scores are in range.
set -u
GPU=${GPU:-3}
T=/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track
D=$T/submission_v8; C=$T/submission_v8_code; V7M=$T/submission_v7/models
DELTA=$T/qwen3_omni/lora_runs/miami_r16_AB/miami.pt; VZ=$T/sub_mech2/vendor_tf.zip
S=/kaggle/working/omni_rt; PY=$S/venv/bin/python; SCORE=/kaggle/working/verify_venv/bin/python; K=$S/kfold
say() { echo "$(date +%T) [build v8] $*"; }
fail() { say "FAILED: $*"; echo BUILD_V8_FAILED; exit 1; }

say "pre-checks on $C"
grep -q STRICT_V8 $C/main.py || fail "strict patch missing"
grep -q CHUNK_VOTE_PATCH $C/vote.py || fail "chunked-vote patch missing"
grep -q COHERE_BEAMS_PATCH $C/stage_cohere.py || fail "cohere-beam patch missing"
grep -q "no base-weight fallback" $C/stage_omni.py || fail "no-fallback patch missing in stage_omni"
grep -q '("cohere", "stage_cohere.py", lambda: \["16", "4"\]' $C/main.py || fail "cohere 4 beams not wired in main PLAN"
grep -q "char_level=False" $C/vote.py || fail "vote is not word-level"
grep -q "fix_super" $C/spelling_rules.py || fail "súper rule missing"
grep -q 'GAPFILL = float(os.environ.get("V8_GAPFILL", "0.6"))' $C/vote.py || fail "gap-fill missing"
for f in $C/*.py; do $PY -m py_compile $f || fail "compile $f"; done
$PY $C/vote.py --selftest || fail "vote self-test"
(cd $C && $PY -c "import spelling_rules" ) || fail "spelling_rules import/self-test"
[ -f $DELTA ] || fail "no delta at $DELTA"; [ -f $VZ ] || fail "no vendor_tf.zip at $VZ"
say "delta $(md5sum $DELTA | cut -c1-12) $(du -h $DELTA | cut -f1) | vendor_tf.zip $(du -h $VZ | cut -f1)"

say "assemble $D"
rm -rf $D; mkdir -p $D/models
for f in main.py common.py vote.py seg_long.py spelling_rules.py stage_omni.py stage_voxtral.py stage_whisper.py stage_conformer.py stage_qwen3asr.py stage_cohere.py stage_parakeet.py conformer_port.py; do cp $C/$f $D/ || fail "copy $f"; done
cp -r $C/vendor $D/vendor; cp $VZ $D/vendor_tf.zip
for m in omni voxtral whisper conformer qwen3asr cohere parakeet; do
  (cd $V7M && find $m -type d) | while read d; do mkdir -p $D/models/$d; done
  (cd $V7M && find $m -type f) | while read f; do ln -f $V7M/$f $D/models/$f 2>/dev/null || cp -L $V7M/$f $D/models/$f; done
done
mkdir -p $D/models/omni_delta; cp $DELTA $D/models/omni_delta/delta.pt; echo "$DELTA" > $D/models/omni_delta/SOURCE.txt
find $D -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
for m in omni voxtral whisper conformer qwen3asr cohere parakeet; do
  a=$(cd $V7M/$m && find . -type f | wc -l); b=$(cd $D/models/$m && find . -type f | wc -l)
  sa=$(du -sbL $V7M/$m | cut -f1); sb=$(du -sbL $D/models/$m | cut -f1)
  [ "$a" = "$b" ] && [ "$sa" = "$sb" ] || fail "models/$m differs from v7 ($a/$b files, $sa/$sb bytes)"
done
cmp -s $DELTA $D/models/omni_delta/delta.pt || fail "delta copy differs"
du -shL $D/models/* | sed "s#$D/##"; say "total $(du -shL $D | cut -f1)"

export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_CACHE_ROOT=$S/vllm_cache
cd $S/work
run() {  # run <data_dir> <val_dir>
  rm -rf $2 $D/vendor_tf; mkdir -p $2; t=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU env -u OMNI_MODELS_DIR -u OMNI_DELTA -u OMNI_LIMIT OMNI_DATA_DIR=$1 OMNI_OUT_PATH=$2/submission.csv OMNI_WORK_DIR=$2/work OMNI_TIME_BUDGET_S=7200 $PY -u $D/main.py > $2/run.log 2>&1
  echo "$?" > $2/exit_code; echo "$(( $(date +%s) - t ))" > $2/elapsed_s
}
say "validate on dev (GPU $GPU)"; run $S/dev $T/build_v8_val
say "dev run: exit $(cat $T/build_v8_val/exit_code), $(cat $T/build_v8_val/elapsed_s)s"
[ "$(cat $T/build_v8_val/exit_code)" = "0" ] || { tail -15 $T/build_v8_val/run.log; fail "dev run exited non-zero"; }
$PY $K/verify_v8_run.py $T/build_v8_val $S/dev 5 || fail "dev run verification"
W=$($SCORE $S/score.py $S/dev/refs.csv --predicted-path $T/build_v8_val/submission.csv | tail -1 | grep -oE "0\.[0-9]+")
say "PACKAGED v8 dev WER = $W (lab 0.0753-0.0755, v7 packaged 0.0775)"
for m in omni voxtral whisper cohere parakeet qwen3asr conformer; do
  say "  member $m dev $($SCORE $S/score.py $S/dev/refs.csv --predicted-path $T/build_v8_val/work/$m.csv | tail -1 | grep -oE '0\.[0-9]+')"
done

say "validate on the glued long set (GPU $GPU)"; run $S/dev_long $T/build_v8_val_long
say "long run: exit $(cat $T/build_v8_val_long/exit_code), $(cat $T/build_v8_val_long/elapsed_s)s"
[ "$(cat $T/build_v8_val_long/exit_code)" = "0" ] || { tail -15 $T/build_v8_val_long/run.log; fail "long run exited non-zero"; }
$PY $K/verify_v8_run.py $T/build_v8_val_long $S/dev_long 58 || fail "long run verification"
WL=$($SCORE $S/score.py $S/dev_long/refs.csv --predicted-path $T/build_v8_val_long/submission.csv | tail -1 | grep -oE "0\.[0-9]+")
say "PACKAGED v8 long WER = $WL (lab 0.0824-0.0826, v7 packaged 0.0891)"
OK=$(echo "$W $WL" | awk '{print ($1 > 0.070 && $1 < 0.080 && $2 > 0.078 && $2 < 0.088) ? 1 : 0}')
[ "$OK" = "1" ] || fail "scores out of range (dev $W, long $WL)"
for v in build_v8_val build_v8_val_long; do say "$v job log: $(wc -l < $T/$v/run.log) lines"; grep -E "stage [a-z0-9]+ (exit|TIMEOUT)|member .* ok|v8 done" $T/$v/run.log | cut -c1-120 | sed "s#^#  [$v] #"; done
rm -rf $D/vendor_tf; find $D -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
echo VALIDATION_PASSED

say "zip (stored)"
rm -f $T/submission_v8.zip
cd $D && zip -r -0 -q $T/submission_v8.zip . -x "*/__pycache__/*" "*.pyc" "*/.cache/*" || fail "zip"
say "zip size $(ls -l $T/submission_v8.zip | awk '{printf "%.2f GB", $5/1e9}')"
unzip -l $T/submission_v8.zip | awk '{print $4}' | grep -E "^(main.py|vendor_tf.zip|models/[a-z0-9_]+/)" | sed -E "s#(models/[a-z0-9_]+/).*#\1#" | sort | uniq -c
cd $T && unzip -tq submission_v8.zip > $T/build_v8pkg_ziptest.log 2>&1; echo "ZIPTEST_EXIT $?" >> $T/build_v8pkg_ziptest.log
say "ziptest: $(tail -1 $T/build_v8pkg_ziptest.log)"
say BUILD_V8_DONE
