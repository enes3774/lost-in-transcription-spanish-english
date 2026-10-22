#!/bin/sh
# Faster build of submission_v9c.zip: N Info-ZIP processes (deflate -6) write part archives into RAM (/dev/shm), the parts are
# stitched into one archive without recompressing (merge_zips.py), the result is checked with `zip -sf`
# and against the package folder's file count. If it is good, the slow single-process zip is stopped and this archive
# becomes submission_v9c.zip (a link to the RAM copy); then the same CRC test + comparison as the normal build run.
set -u
T=/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track
S=/kaggle/working/omni_rt; PY=$S/venv/bin/python; K=$S/kfold/v9; D=$T/submission_v9c
W=/dev/shm/v9c_parts; R=/dev/shm/submission_v9c.zip; LOG=$T/build_v9c_fast.log; B=$T/build_v9c.log; N=5
say() { echo "$(date +%T) [fast] $*" >> $LOG; }
: > $LOG
[ -e $R ] && { say "FAILED: $R exists"; exit 1; }
rm -rf $W; mkdir -p $W
$PY $K/merge_zips.py bins $D $W $N >> $LOG 2>&1 || { say "FAILED: bins"; exit 1; }
cd $D
i=0; while [ $i -lt $N ]; do
  ( zip -6 -q $W/part_$i.zip -@ < $W/bin_$i.lst; echo "$(date +%T) [fast] part $i exit $? $(ls -l $W/part_$i.zip | awk '{printf "%.2f GB", $5/1e9}')" >> $LOG ) &
  i=$((i+1))
done
( find . -type d ! -name . | sed 's#^\./##; s#$#/#' | sort | zip -0 -q $W/part_dirs.zip -@; echo "$(date +%T) [fast] dirs exit $?" >> $LOG ) &
wait
[ "$(grep -c 'exit 0' $LOG)" = "$((N + 1))" ] || { say "FAILED: a part zip failed"; exit 1; }
$PY $K/merge_zips.py merge $R $W/part_dirs.zip $W/part_[0-9].zip >> $LOG 2>&1 || { say "FAILED: merge"; exit 1; }
zip -sf $R > $T/build_v9c_fast_zipsf.log 2>&1 || { say "FAILED: zip -sf"; exit 1; }
NF=$(find -L $D -type f | wc -l); ND=$(find $D -type d ! -path $D | wc -l)
say "listing: $(tail -1 $T/build_v9c_fast_zipsf.log) ; folder has $NF files + $ND dirs"
grep -q "Total $((NF + ND)) entries" $T/build_v9c_fast_zipsf.log || { say "FAILED: entry count"; exit 1; }
[ -e $T/submission_v9c.zip ] && { say "the slow zip finished first -> keeping it, removing the RAM copy"; rm -f $R; exit 0; }
# the RAM archive is ready first: stop the slow zip, remove its temp file, link, and tell the upload gate
SP=$(pgrep -f "v9c_build.sh zip" | head -1); ZP=$(pgrep -f "zip -r -6 -q" | head -1)
TMP=""; [ -n "$ZP" ] && TMP=$(ls -l /proc/$ZP/fd 2>/dev/null | grep -oE "$T/zi[A-Za-z0-9]{6}$" | head -1)
[ -n "$SP" ] && kill $SP; [ -n "$ZP" ] && kill $ZP; sleep 3
[ -n "$TMP" ] && [ -f "$TMP" ] && rm -f "$TMP"
ln -s $R $T/submission_v9c.zip || { say "FAILED: link"; exit 1; }
say "slow zip stopped (pids $SP $ZP, temp $(basename "$TMP")); submission_v9c.zip -> $R"
echo "$(date +%T) [v9c zip] ZIP_OK size $(stat -L -c %s $R | awk '{printf "%.3f GB (%d bytes)", $1/1e9, $1}') - parallel build in RAM, stitched without recompression (v8 zip that ran: 63412004300 bytes; v9b zip that hung: 76414326730)" >> $B
cd $T && unzip -tq submission_v9c.zip > $T/build_v9c_ziptest.log 2>&1; rc=$?
echo "ZIPTEST_EXIT $rc" >> $T/build_v9c_ziptest.log
[ "$rc" = "0" ] || { echo "$(date +%T) [v9c zip] FAILED: unzip -t exit $rc" >> $B; exit 1; }
echo "$(date +%T) [v9c zip] TEST_OK ($(tail -2 $T/build_v9c_ziptest.log | head -1))" >> $B
$PY $K/cmp_zips_v9c.py > $T/build_v9c_zipcmp.log 2>&1; rc=$?
cut -c1-400 $T/build_v9c_zipcmp.log >> $B
[ "$rc" = "0" ] || { echo "$(date +%T) [v9c zip] FAILED: zip comparison" >> $B; exit 1; }
echo "$(date +%T) [v9c zip] ALL_DONE" >> $B
say "FAST_DONE"
