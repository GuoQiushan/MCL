#!/bin/bash
WEIGHT_PTH=$1
WEIGHTS=$2
until [ -f $1 ]
do
sleep 5s
done

python convert-sppretrain-to-detectron2.py $WEIGHT_PTH ./model_zoo/${WEIGHTS}.pkl

for loop_i in {1..3}
do
DET_CFG=coco_R_50_FPN_CONV_1x_moco

echo ./configs/${DET_CFG}.yaml
echo ./work_dir/${DET_CFG}_${WEIGHTS}

python $(dirname "$0")/train_net.py --config-file ./configs/${DET_CFG}.yaml \
    --num-gpus 8 --dist-url 'tcp://localhost:10001' MODEL.WEIGHTS ./model_zoo/${WEIGHTS}.pkl OUTPUT_DIR ./work_dir/${DET_CFG}_${WEIGHTS}_loop_${loop_i}
done


for loop_i in {1..3}
do
DET_CFG=coco_R_50_FPN_CONV_1x_moco_wd2.5

echo ./configs/${DET_CFG}.yaml
echo ./work_dir/${DET_CFG}_${WEIGHTS}

python $(dirname "$0")/train_net.py --config-file ./configs/${DET_CFG}.yaml \
    --num-gpus 8 --dist-url 'tcp://localhost:10001' MODEL.WEIGHTS ./model_zoo/${WEIGHTS}.pkl OUTPUT_DIR ./work_dir/${DET_CFG}_${WEIGHTS}_loop_${loop_i}
done



DET_CFG=coco_R_50_RetinaNet_1x_moco

echo ./configs/${DET_CFG}.yaml
echo ./work_dir/${DET_CFG}_${WEIGHTS}

python $(dirname "$0")/train_net.py --config-file ./configs/${DET_CFG}.yaml \
    --num-gpus 8 --dist-url 'tcp://localhost:10001' MODEL.WEIGHTS ./model_zoo/${WEIGHTS}.pkl OUTPUT_DIR ./work_dir/${DET_CFG}_${WEIGHTS}




DET_CFG=coco_R_50_FPN_CONV_2x_moco

echo ./configs/${DET_CFG}.yaml
echo ./work_dir/${DET_CFG}_${WEIGHTS}

python $(dirname "$0")/train_net.py --config-file ./configs/${DET_CFG}.yaml \
    --num-gpus 8 --dist-url 'tcp://localhost:10001' MODEL.WEIGHTS ./model_zoo/${WEIGHTS}.pkl OUTPUT_DIR ./work_dir/${DET_CFG}_${WEIGHTS}
