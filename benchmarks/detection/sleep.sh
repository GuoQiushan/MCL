#!/bin/bash
WEIGHT_PTH=$1
WEIGHTS=$2
until [ -f $1 ]
do
sleep 5s
done

python convert-pretrain-to-detectron2.py $WEIGHT_PTH $WEIGHTS

DET_CFG=$3
OUTPUT_DIR=$4

python $(dirname "$0")/train_net.py --config-file $DET_CFG \
    --num-gpus 8 MODEL.WEIGHTS $WEIGHTS OUTPUT_DIR $OUTPUT_DIR
