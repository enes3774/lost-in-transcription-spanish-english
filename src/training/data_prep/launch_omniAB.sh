#!/bin/sh
S=/kaggle/working/omni_rt; K=/home/jovyan/work/2026_08_Mozilla_Lost_In_Transcription/input_enspa/miami_kfold
tail -n +2 $K/miami_A/test_metadata.csv | cut -d, -f1 > $S/kfold/miamiAB_names.txt
tail -n +2 $K/miami_B/test_metadata.csv | cut -d, -f1 >> $S/kfold/miamiAB_names.txt
echo "names: $(wc -l < $S/kfold/miamiAB_names.txt)  T-half overlap: $(tail -n +2 $K/miami_Ttest/test_metadata.csv | cut -d, -f1 | grep -cxFf - $S/kfold/miamiAB_names.txt)"
cd $S/work
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
setsid nohup $S/venv/bin/python -u $S/omni_lora_train.py --run miami_r16_AB --miami 30000 --miami_names $S/kfold/miamiAB_names.txt --seed 1 --epochs 1 --accum 8 --save_every 500 > $S/train_omni_miamiAB.log 2>&1 < /dev/null &
echo "launched pid $!"
