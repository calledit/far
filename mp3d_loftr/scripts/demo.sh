#!/bin/bash

data_cfg_path="configs/data/mp3d.py"
main_cfg_path="configs/loftr/indoor/loftr_ds_dense.py"

# pretrained model
ckpt_path="pretrained_models/far_8pt.ckpt"
echo "script takes fx and fy as inputs"
echo "all images are rescaled to 640x480 so fx and fy needs to be calcualted with x_fov and y_fov and dimensions 640x480"
python demo.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --img_path0 demo/mp3d_Vt2qJdWjCF2_0_12_2.png --img_path1 demo/mp3d_Vt2qJdWjCF2_0_12_32.png \
    --fx $1 --fy $2 --cx 320 --cy 240 --h 480 --w 640
