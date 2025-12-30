#!/bin/bash

# Updated training script for 70% UAR Target (Optimized for Kaggle GPU P100/T4)
# Note: If running locally on Mac M2, change batch-size to 4 to avoid OOM.
# MODIFIED FOR ViT-B/32 with a safer batch size

python main.py \
    --mode train \
    --exper-name prompt_tuning_vitb32_safer_batch \
    --gpu 0 \
    --epochs 50 \
    --batch-size 2 \
    --workers 4 \
    --gradient-accumulation-steps 2 \
    --use-amp True \
    --lr 0.003 \
    --lr-image-encoder 1e-6 \
    --lr-prompt-learner 0.001 \
    --weight-decay 0.0001 \
    --momentum 0.9 \
    --milestones 20 35 \
    --gamma 0.1 \
    --temporal-layers 1 \
    --num-segments 16 \
    --duration 1 \
    --image-size 224 \
    --seed 42 \
    --print-freq 10 \
    --root-dir /kaggle/input/raer-video-emotion-dataset/ \
    --train-annotation RAER/annotation/train.txt \
    --test-annotation RAER/annotation/test.txt \
    --data-percentage 1.0 \
    --clip-path ViT-B/32 \
    --bounding-box-face RAER/bounding_box/face.json \
    --bounding-box-body RAER/bounding_box/body.json \
    --text-type class_descriptor \
    --contexts-number 8 \
    --class-token-position end \
    --class-specific-contexts True \
    --load_and_tune_prompt_learner True \
    --lambda-mi 1.0 \
    --lambda-dc 2.0 \
    --mi-warmup 2 \
    --mi-ramp 5 \
    --dc-warmup 3 \
    --dc-ramp 5 \
    --label-smoothing 0.2 \
    --semantic-smoothing True \
    --smoothing-temp 0.1
