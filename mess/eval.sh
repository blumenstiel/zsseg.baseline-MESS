#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate zsseg

# Benchmark
export DETECTRON2_DATASETS="datasets"
TEST_DATASETS="bdd100k_sem_seg_val dark_zurich_sem_seg_val mhp_v1_sem_seg_test foodseg103_sem_seg_test atlantis_sem_seg_test dram_sem_seg_test isaid_sem_seg_val isprs_potsdam_sem_seg_test_irrg worldfloods_sem_seg_test_irrg floodnet_sem_seg_test uavid_sem_seg_val kvasir_instrument_sem_seg_test chase_db1_sem_seg_test cryonuseg_sem_seg_test paxray_sem_seg_test_lungs paxray_sem_seg_test_bones paxray_sem_seg_test_mediastinum paxray_sem_seg_test_diaphragm corrosion_cs_sem_seg_test deepcrack_sem_seg_test pst900_sem_seg_test zerowaste_sem_seg_test suim_sem_seg_test cub_200_sem_seg_test cwfid_sem_seg_test"

# Run experiments
for DATASET in $TEST_DATASETS
do
 python train_net.py --num-gpus 1 --eval-only --config-file configs/ade20k-150/cross_dataset_test_only.yaml MODEL.WEIGHTS weights/model_final.pth OUTPUT_DIR output/ZSSeg/$DATASET DATASETS.TEST \(\"$DATASET\",\) MODEL.CLIP_ADAPTER.MASK_THR 0.5
done

# Combine results
python mess/evaluation/mess_evaluation.py --model_outputs output/ZSSeg

# Run evaluation with:
# nohup bash mess/eval.sh > eval.log &
# tail -f eval.log
