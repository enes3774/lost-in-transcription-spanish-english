#!/bin/sh
# assemble submission_v4 (Omni AWQ-4bit, Voxtral FP8 + LoRA, Qwen3-ASR, conformer), dev run, bench run
cd /kaggle/working/omni_rt
T=/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/lost-in-transcription-spanish-english-track
Q=$T/quant; V3=$T/submission_v3; V=$T/submission_v4
PY=/kaggle/working/omni_rt/venv/bin/python; SC=/kaggle/working/verify_venv/bin/python
rm -rf $V; mkdir -p $V/models
for f in main.py common.py vote.py stage_omni.py stage_voxtral.py stage_qwen3asr.py stage_conformer.py conformer_port.py; do cp $V3/$f $V/; done
cp -r $V3/vendor $V/; cp -r $V3/models/conformer $V/models/
ln -sfn $T/qwen3_asr_training/merged_qwen3asr_miami $V/models/qwen3asr
ln -sfn $Q/omni_awq4 $V/models/omni
mkdir -p $V/models/voxtral/lora
for f in consolidated.safetensors params.json tekken.json; do ln -sf $Q/vox_fp8/$f $V/models/voxtral/$f; done
cp $Q/vox_fp8/lora/* $V/models/voxtral/lora/
# AWQ Omni: let vLLM pick the quantized MoE kernel (bf16 needed the forced Triton path)
sed -i 's#"OMNI_TIME_BUDGET_S": str(max(600, remaining() - 1500))})#"OMNI_TIME_BUDGET_S": str(max(600, remaining() - 1500)), "OMNI_MOE_BACKEND": os.environ.get("OMNI_MOE_BACKEND", "auto")})#' $V/main.py
grep -c "OMNI_MOE_BACKEND" $V/main.py
du -shL $V/models/* | cut -f1 | tr "\n" " "; echo
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_CACHE_ROOT=/kaggle/working/omni_rt/vllm_cache OMNI_GPU_UTIL=0.77
mkdir -p v4work && cd v4work
echo "$(date +%T) v4 main.py on dev"; S=$(date +%s)
OMNI_DATA_DIR=/kaggle/working/omni_rt/dev OMNI_OUT_PATH=/kaggle/working/omni_rt/v4work/dev_final.csv OMNI_WORK_DIR=/kaggle/working/omni_rt/v4work/work_dev $PY $V/main.py > main_dev.log 2>&1
echo "DEV_WALL $(( $(date +%s) - S ))s"; grep -E "^\[" main_dev.log | grep -E "stage .* exit|models present" | cut -c1-100
for f in omni voxtral conformer qwen3asr; do printf "%s " $f; $SC ../score.py ../dev/refs.csv --predicted-path work_dev/$f.csv 2>/dev/null || echo missing; done
printf "FINAL "; $SC ../score.py ../dev/refs.csv --predicted-path dev_final.csv
echo "$(date +%T) v4 main.py on bench"; S=$(date +%s)
OMNI_DATA_DIR=/kaggle/working/omni_rt/bench OMNI_OUT_PATH=/kaggle/working/omni_rt/v4work/bench_final.csv OMNI_WORK_DIR=/kaggle/working/omni_rt/v4work/work_bench $PY $V/main.py > main_bench.log 2>&1
echo "BENCH_WALL $(( $(date +%s) - S ))s"; grep -E "^\[" main_bench.log | grep -E "stage .* exit|budget|restor" | cut -c1-100
$PY -c "
import csv; r=list(csv.DictReader(open('/kaggle/working/omni_rt/v4work/bench_final.csv'))); print('bench rows', len(r), 'empty', sum(1 for x in r if not x['transcript'].strip()))"
echo V4_CHAIN_DONE
