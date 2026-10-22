#!/bin/sh
# Build + validate (+ zip only with ZIP=1) the v9 package on ONE GPU (default 3). Everything on the HDD ($T).
# v9 = v8 (same models, hardlinked from submission_v8/models, incl. the Omni delta) + Phi-4-multimodal base (ZFTurbo
# fork snapshot) + our Miami delta + v9 code (TTA Whisper pair, phi4 stage, vendor_phi/backoff, strict vote names).
set -u
GPU=${GPU:-3}
T=/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track
D=$T/submission_v9; C=$T/submission_v9_code; V8M=$T/submission_v8/models
PHI=$(ls -d /home/jovyan/work/Enes/lost_in_transcription/cache/huggingface/hub/models--ZFTurbo--Phi-4-multimodal-instruct/snapshots/*)
PDELTA=$T/phi4/AB_frag/best/delta.safetensors; VZ=$T/sub_mech2/vendor_tf.zip
S=/kaggle/working/omni_rt; PY=$S/venv/bin/python; SCORE=/kaggle/working/verify_venv/bin/python; K=$S/kfold
say() { echo "$(date +%T) [build v9] $*"; }
fail() { say "FAILED: $*"; echo BUILD_V9_FAILED; exit 1; }

say "pre-checks on $C"
grep -q STRICT_V8 $C/main.py || fail "strict patch missing"
grep -q CHUNK_VOTE_PATCH $C/vote.py || fail "chunked-vote patch missing"
grep -q COHERE_BEAMS_PATCH $C/stage_cohere.py || fail "cohere-beam patch missing"
grep -q '"whisper_s090": 8.0, "whisper_s110": 8.0, "phi4": 8.0}' $C/vote.py || fail "v9 weights missing"
grep -q 'set(systems) != set(WEIGHTS)' $C/vote.py || fail "strict name check missing"
grep -q '("phi4", "stage_phi4.py", lambda: \["8"\], 600)' $C/main.py || fail "phi4 not in PLAN"
grep -q 'stage_tta.py' $C/main.py || fail "tta block missing"
for f in $C/*.py; do $PY -m py_compile $f || fail "compile $f"; done
$PY $C/vote.py --selftest || fail "vote self-test"
[ -f $PDELTA ] || fail "no phi4 delta"; [ -f $PHI/model.safetensors.index.json ] || fail "no phi4 snapshot"; [ -f $VZ ] || fail "no vendor_tf.zip"
say "phi4 delta $(md5sum $PDELTA | cut -c1-12) $(du -h $PDELTA | cut -f1)"

say "assemble $D"
rm -rf $D; mkdir -p $D/models
for f in main.py common.py vote.py seg_long.py spelling_rules.py stage_omni.py stage_voxtral.py stage_whisper.py stage_conformer.py stage_qwen3asr.py stage_cohere.py stage_parakeet.py stage_phi4.py stage_tta.py conformer_port.py; do cp $C/$f $D/ || fail "copy $f"; done
cp -r $C/vendor $D/vendor; cp -r $C/vendor_phi $D/vendor_phi; cp $VZ $D/vendor_tf.zip
for m in omni omni_delta voxtral whisper conformer qwen3asr cohere parakeet; do
  (cd $V8M && find $m -type d) | while read d; do mkdir -p $D/models/$d; done
  (cd $V8M && find $m -type f) | while read f; do ln -f $V8M/$f $D/models/$f 2>/dev/null || cp -L $V8M/$f $D/models/$f; done
done
mkdir -p $D/models/phi4 $D/models/phi4_delta
for f in $PHI/*; do b=$(basename $f); [ "$b" = "README.md" ] && continue; ln -f $(readlink -f $f) $D/models/phi4/$b 2>/dev/null || cp -L $f $D/models/phi4/$b; done
ln -f $PDELTA $D/models/phi4_delta/delta.safetensors 2>/dev/null || cp $PDELTA $D/models/phi4_delta/delta.safetensors
echo "$PDELTA (phi4_train.py, Miami A+B, best step 1500, Tsel600 0.2433)" > $D/models/phi4_delta/SOURCE.txt
find $D -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
for m in omni omni_delta voxtral whisper conformer qwen3asr cohere parakeet; do
  a=$(cd $V8M/$m && find . -type f | wc -l); b=$(cd $D/models/$m && find . -type f | wc -l)
  sa=$(du -sbL $V8M/$m | cut -f1); sb=$(du -sbL $D/models/$m | cut -f1)
  [ "$a" = "$b" ] && [ "$sa" = "$sb" ] || fail "models/$m differs from v8 ($a/$b files, $sa/$sb bytes)"
done
for f in $PHI/*; do b=$(basename $f); [ "$b" = "README.md" ] && continue; cmp -s $f $D/models/phi4/$b || fail "phi4/$b differs"; done
cmp -s $PDELTA $D/models/phi4_delta/delta.safetensors || fail "phi4 delta differs"
du -shL $D/models/* | sed "s#$D/##"; say "total $(du -shL $D | cut -f1)"

export CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_CACHE_ROOT=$S/vllm_cache
cd $S/work
run() {  # run <data_dir> <val_dir>   (PYTHONPATH cleared: backoff must come from vendor_phi, as on the runner)
  rm -rf $2 $D/vendor_tf; mkdir -p $2; t=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU env -u OMNI_MODELS_DIR -u OMNI_DELTA -u OMNI_LIMIT -u PYTHONPATH OMNI_DATA_DIR=$1 OMNI_OUT_PATH=$2/submission.csv OMNI_WORK_DIR=$2/work OMNI_TIME_BUDGET_S=7200 $PY -u $D/main.py > $2/run.log 2>&1
  echo "$?" > $2/exit_code; echo "$(( $(date +%s) - t ))" > $2/elapsed_s
}
say "validate on dev (GPU $GPU)"; run $S/dev $T/build_v9_val
say "dev run: exit $(cat $T/build_v9_val/exit_code), $(cat $T/build_v9_val/elapsed_s)s"
[ "$(cat $T/build_v9_val/exit_code)" = "0" ] || { tail -15 $T/build_v9_val/run.log; fail "dev run exited non-zero"; }
$PY $K/verify_v9_run.py $T/build_v9_val $S/dev 5 || fail "dev run verification"
W=$($SCORE $S/score.py $S/dev/refs.csv --predicted-path $T/build_v9_val/submission.csv | tail -1 | grep -oE "0\.[0-9]+")
say "PACKAGED v9 dev WER = $W (lab 0.0751-0.0753; packaged v8 0.0755)"
for m in omni voxtral whisper cohere parakeet qwen3asr conformer phi4 whisper_s090 whisper_s110; do
  say "  member $m dev $($SCORE $S/score.py $S/dev/refs.csv --predicted-path $T/build_v9_val/work/$m.csv | tail -1 | grep -oE '0\.[0-9]+')"
done

say "validate on the glued long set (GPU $GPU)"; run $S/dev_long $T/build_v9_val_long
say "long run: exit $(cat $T/build_v9_val_long/exit_code), $(cat $T/build_v9_val_long/elapsed_s)s"
[ "$(cat $T/build_v9_val_long/exit_code)" = "0" ] || { tail -15 $T/build_v9_val_long/run.log; fail "long run exited non-zero"; }
$PY $K/verify_v9_run.py $T/build_v9_val_long $S/dev_long 58 || fail "long run verification"
WL=$($SCORE $S/score.py $S/dev_long/refs.csv --predicted-path $T/build_v9_val_long/submission.csv | tail -1 | grep -oE "0\.[0-9]+")
say "PACKAGED v9 long WER = $WL (lab 0.0808; packaged v8 0.0826)"
OK=$(echo "$W $WL" | awk '{print ($1 > 0.070 && $1 < 0.0765 && $2 > 0.076 && $2 < 0.0835) ? 1 : 0}')
[ "$OK" = "1" ] || fail "scores out of range (dev $W, long $WL)"
for v in build_v9_val build_v9_val_long; do say "$v job log: $(wc -l < $T/$v/run.log) lines"; grep -E "stage [a-z0-9_]+ (exit|TIMEOUT)|member .* ok|v9 done" $T/$v/run.log | cut -c1-120 | sed "s#^#  [$v] #"; done
rm -rf $D/vendor_tf; find $D -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
echo VALIDATION_PASSED
[ "${ZIP:-0}" = "1" ] || { say "not zipping (ZIP=1 to zip)"; say BUILD_V9_DONE; exit 0; }

say "zip (stored)"
rm -f $T/submission_v9.zip
cd $D && zip -r -0 -q $T/submission_v9.zip . -x "*/__pycache__/*" "*.pyc" "*/.cache/*" || fail "zip"
say "zip size $(ls -l $T/submission_v9.zip | awk '{printf "%.2f GB", $5/1e9}')"
unzip -l $T/submission_v9.zip | awk '{print $4}' | grep -E "^(main.py|vendor_tf.zip|vendor_phi/|models/[a-z0-9_]+/)" | sed -E "s#(models/[a-z0-9_]+/|vendor_phi/).*#\1#" | sort | uniq -c
cd $T && unzip -tq submission_v9.zip > $T/build_v9pkg_ziptest.log 2>&1; echo "ZIPTEST_EXIT $?" >> $T/build_v9pkg_ziptest.log
say "ziptest: $(tail -1 $T/build_v9pkg_ziptest.log)"
say BUILD_V9_DONE
