# PointNet2 for semantic segmentation of 3d points clouds
By Mathieu Orhan and Guillaume Dekeyser (Ecole des Ponts et Chaussées, Paris, 2018).

## Introduction
This project is a student fork of PointNet2, by Charles R. Qi, Li (Eric) Yi, Hao Su, Leonidas J. Guibas from Stanford University.
You can refer to the original PointNet2 paper and code (https://github.com/charlesq34/pointnet2) for details.

This fork focused on semantic segmentation, with the goal of comparing three datasets : Scannet, Semantic-8 and Bertrand Le Saux aerial LIDAR dataset.
To achieve that, we clean, document, refactor, and improve the original project.
We will compare the same datasets later with SnapNet, another state-of-the-art semantic segmentation project.

## Dependancies and data
We work on Ubuntu 16.04 with 3 GTX Titan Black and a GTX Titan X. On older GPUs, like my GTX 860m, you can expect to lower the number of points and the batch size for the training, otherwise you will get a OutOfMemory from TensorFlow.
You have to install TensorFlow on GPU (we use TF 1.2, cuda 8.0, python 2.7, but it should also work on newer versions with minor changes). Then, you have to compile the custom TensorFlow operators in the tf_ops subdirectories, with the .sh files. You may have to install some additionnal Python modules.

Get the preprocessed data (you can also preprocess the semantic data from raw data in the directory dataset/preprocessing) :
- Scannet : https://onedrive.live.com/?authkey=%21AHEO5Ik8Hn4Ue2Y&cid=423FEBB4168FD396&id=423FEBB4168FD396%21136&parId=423FEBB4168FD396%21134&action=locate
- Semantic : https://drive.google.com/file/d/1-l2h3yh1xBAzR2JhqPz-YIzM4vtEOf4W/view?usp=sharing

Compiling the C++ parts if you want to preprocess the data or to calculate results on the raw data can result in the following error : "/usr/bin/ld: cannot find -lvtkproj4", but you can overcome this difficulty by using this trick : ln -s /usr/lib/x86_64-linux-gnu/libvtkCommonCore-6.2.so /usr/lib/libvtkproj4.so (see https://github.com/PointCloudLibrary/pcl/issues/1594 for details).

For downloading the raw data , go into dataset/ directory and use the command:
./downloadAndExtractSem8.sh

For preprocessing with this raw data and the voxel_size you want, go into the preprocessing directory and use the command:
./preprocess.sh ../dataset/raw_semantic_data ../dataset/semantic_data 'voxel_size'
(with the voxel_size you want, in m. default is 0.05)

For training, use python train.py --config=your_config --log=your_logs
Both scannet and semantic_8 should be trainable.

For interpolating results, first use predict.py --cloud=true --n=100 --ckpt=your_ckpt --dataset=semantic --set=test for example. Files will be created in visu/semantic_test/full_scenes_predictions and will contain predictions on sparse point clouds. The actual interpolation is done in interpolation directory with the command:
./interpolate path/to/raw/data visu/semantic_test/full_scenes_predictions /path/to/where/to/put/results 'voxel_size'
(with the voxel_size you want, in m. default is 0.1)

Please check the source files for more details.
