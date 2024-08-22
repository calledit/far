#!/bin/bash

source activate far

cd mp3d_loftr
mkdir data
cd data
rm -r imgs_4fps
wget -O video.mp4 $3
mkdir imgs_4fps

ffmpeg -i video.mp4 -vf 'scale=640:480' imgs_4fps/%04d.png

cd ..
scripts/demo.sh $1 $2
