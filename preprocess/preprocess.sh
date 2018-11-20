#!/bin/bash
display_usage() {
    echo "This script preprocesses the point cloud with adaptive sampling : more points are taken where there is curvature"
    echo "provide first the directory where you have the raw semantic_data, then where you want the preprocessed data, and finally the voxel size which you want for sampling"
}

# check whether user had supplied -h or --help . If yes display usage
if [[ ($# == "--help") || $# == "-h" ]]; then
    display_usage
    exit 0
fi

if [ $# -le 1 ]; then
    display_usage
    exit 1
fi

if [ $# -le 2 ]; then
    echo "voxel size set to 0.05 default"
    voxel_size=0.05
else
    voxel_size=$3
fi

if [ ! -d $1 ]; then
    echo $1" not found"
    exit 1
fi

if [ ! -d $2 ]; then
    echo $2" not found"
    exit 1
fi

mkdir tmp_storage

if [ ! -d "build" ]; then
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make
    cd ..
fi

./build/sample $1 tmp_storage/ $voxel_size

./convert_folder_to_npz.sh tmp_storage/ $2

if [ "$(ls -A $tmp_storage)" ]; then
    if ! [ "$(ls -A $2)" ]; then
        echo "Something went wrong with the conversion from .txt to .npz, but you have your files in tmp_storage in .txt"
    else
        rm -r tmp_storage
    fi
else
    rm -r tmp_storage
fi
