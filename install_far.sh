#!/bin/bash

apt-get install vim unzip ffmpeg libsm6 libxext6  -y

conda env create -f environment.yml

source activate far

cd mp3d_loftr

wget https://fouheylab.eecs.umich.edu/~cnris/far/model_checkpoints/mp3d_loftr/pretrained_models.zip --no-check-certificate
unzip pretrained_models.zip
