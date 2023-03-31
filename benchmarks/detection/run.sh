#!/bin/bash
DET_CFG=$1
WEIGHTS=$2
PY_ARGS=${@:3}

python $(dirname "$0")/train_net.py --config-file $DET_CFG \
    --num-gpus 8 --resume MODEL.WEIGHTS $WEIGHTS ${PY_ARGS}
