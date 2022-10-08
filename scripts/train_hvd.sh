#!/bin/bash
WANDB_KEY=$WANDB_KEY # paste your key here if wandb is enabled
TIMESTAMP=$(date +'%Y-%m-%d-%H-%M')
HOSTFILE="./hostfile"
LOGDIR="logs"
NUM_GPUS=4
TASK_NAME="imdb-cls"

mkdir -p $LOGDIR
NCCL_DEBUG=info NCCL_IB_DISABLE=1 NCCL_NET_GDR_LEVEL=2 \
horovodrun -np $NUM_GPUS -hostfile $HOSTFILE --timeline-filename $LOGDIR/timeline-$TIMESTAMP.json \
python -u run.py \
  --horovod \
  --task-name $TASK_NAME \
  --timestamp $TIMESTAMP \
  --label-path config/labelspace_imdb \
  --pretrained ../pretrained_models/bert-base-uncased \
  --pooler-type cls \
  --batch-size 64 \
  --save-cache \
  --use-cache \
  --train-path data/imdb/train.csv \
  --valid-path data/imdb/test.csv \
  --ckpt-path ckpts \
  --max-epochs 10 \
  --patience 3 \
  --lr 2e-5 \
  --no-scale-lr \
  --dropout 0.1 \
  --shuffle \
  --max-seq-len 128 \
  --log-interval 5 \
  --use-wandb \
  --wandb-key $WANDB_KEY \
  2>&1 | tee ${LOGDIR}/${TASK_NAME}-${TIMESTAMP}.log

