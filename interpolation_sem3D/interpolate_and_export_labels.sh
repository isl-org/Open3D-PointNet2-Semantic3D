#!/bin/bash
display_usage() { 
	echo "This script interpolates the results on the preprocessed data to the raw point clouds." 
	echo "provide first the directory where you have the raw semantic_data, then where you have the results of prediction.py (in visu/semantic_test/full_scenes_predictions for example), then where results should be written, and finally the voxel size used for interpolation" 
	} 

# check whether user had supplied -h or --help . If yes display usage 
if [[ ( $# == "--help") ||  $# == "-h" ]] 
	then 
		display_usage
                exit 0
	fi 

if [  $# -le 2 ] 
	then 
		display_usage
		exit 1
	fi 

if [ ! -d $1 ]; then
  echo $1" not found"
     exit 1
fi
if [ ! -d $2 ]; then
  echo $2" not found"
     exit 1
fi

if [ ! -d $3 ]; then
  echo $3" not found"
     exit 1
fi

if [ ! -d "build" ]; then
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make
  cd ..
fi

if [  $# -le 3 ] 
	then 
                echo "voxel size set to 0.1 default"
		voxel_size=0.1
        else
                voxel_size=$4
	fi 

./build/interpolate $1 $2 $3 $voxel_size 1

