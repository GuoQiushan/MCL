#!/usr/bin/env bash

set -x

PARTITION=$1
CFG=$2
GPUS=$3
PY_ARGS=${@:4}
JOB_NAME="openselfsup"
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-"-x SH-IDC1-10-198-8-80"}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

GLOG_vmodule=MemcachedClient=-1 \
spring.submit arun \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --gpu \
    -n ${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    ${SRUN_ARGS} \
    "python -u tools/train.py ${CFG} \
        --work_dir ${WORK_DIR} --seed 0 ${PY_ARGS}"
