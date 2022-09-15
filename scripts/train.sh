#!/bin/bash
WANDB_KEY="" # paste your key here if wandb is enabled

python -u run.py \
  --task-name "imdb-cls" \
  --label-path config/labelspace_imdb \
  --pretrained ../pretrained_models/bert-base-uncased \
  --pooler-type cls \
  --batch-size 128 \
  --save-cache \
  --use-cache \
  --train-path data/imdb/train.csv \
  --valid-path data/imdb/test.csv \
  --ckpt-path ckpts \
  --max-epochs 10 \
  --patience 3 \
  --lr 2e-5 \
  --dropout 0.1 \
  --shuffle \
  --max-seq-len 128 \
  --use-wandb \
  --wandb-key $WANDB_KEY
