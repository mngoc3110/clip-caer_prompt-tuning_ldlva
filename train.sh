#!/bin/bash
set -e

python main.py \
  --mode train \
  --exper-name m2max_prompttuning_vitb32_wrs_effbs8 \
  --gpu mps \
  --epochs 50 \
  --batch-size 8 \
  --workers 4 \
  --gradient-accumulation-steps 1 \
  --use-amp True \
  \
  --lr 3e-4 \
  --lr-image-encoder 1e-6 \
  --lr-prompt-learner 1e-3 \
  --weight-decay 1e-4 \
  --momentum 0.9 \
  --milestones 20 35 \
  --gamma 0.1 \
  \
  --temporal-layers 1 \
  --num-segments 16 \
  --duration 1 \
  --image-size 224 \
  --seed 42 \
  --print-freq 10 \
  \
  --root-dir /kaggle/input/raer-video-emotion-dataset/ \
  --train-annotation /kaggle/input/raer-annot/annotation/train_80.txt \
  --val-annotation /kaggle/input/raer-annot/annotation/val_20.txt \
  --test-annotation /kaggle/input/raer-annot/annotation/test.txt \
  --data-percentage 1.0 \
  \
  --clip-path ViT-B/32 \
  --bounding-box-face /kaggle/input/raer-video-emotion-dataset/RAER/bounding_box/face.json \
  --bounding-box-body /kaggle/input/raer-video-emotion-dataset/RAER/bounding_box/body.json \
  \
  --text-type class_descriptor \
  --contexts-number 12 \
  --class-token-position end \
  --class-specific-contexts True \
  --load_and_tune_prompt_learner True \
  \
  --lambda-mi 0.5 \
  --lambda-dc 1.0 \
  --mi-warmup 3 \
  --mi-ramp 8 \
  --dc-warmup 5 \
  --dc-ramp 10 \
  \
  --label-smoothing 0.05 \
  --semantic-smoothing True \
  --smoothing-temp 0.1 \
  \
  --logit-adjust False \
  --logit-adjust-tau 0.8 \
  \
  --use-weighted-sampler True \
  --max-class-weight 5 \
  --use-class-weights False
