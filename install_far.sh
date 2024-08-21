#!/bin/bash

apt-get install vim unzip ffmpeg libsm6 libxext6  -y

conda env create -f environment.yml

source activate far

cd mp3d_loftr

wget https://fouheylab.eecs.umich.edu/~cnris/far/model_checkpoints/mp3d_loftr/pretrained_models.zip --no-check-certificate
unzip pretrained_models.zip

mkdir data

cd data

wget http://example.com/video.mp4
mkdir imgs_4fps

ffmpeg -i video.mp4 -vf 'scale=640:480' imgs_4fps/%04d.png

scripts/demo.sh
