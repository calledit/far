#!/bin/bash

source activate far

cd mp3d_loftr
mkdir data
cd data
rm -r imgs_4fps
rm -r masks_4fps
rm -r depth_4fps
wget -O video.mp4 $3
wget -O mask.mp4 $4
wget -O depth.mp4 $5
mkdir imgs_4fps
mkdir depth_4fps
mkdir masks_4fps

ffmpeg -i video.mp4 -vf 'fps=4,scale=640:480' imgs_4fps/%06d.png
ffmpeg -i mask.mp4 -vf 'fps=4,scale=640:480' masks_4fps/%06d.png
ffmpeg -i depth.mp4 -vf 'fps=4,scale=640:480' depth_4fps/%06d.png

cd ..
scripts/demo.sh $1 $2
