# PointNet2 for semantic segmentation of 3d points clouds
By Mathieu Orhan and Guillaume Dekeyser (Ecole des Ponts et Chaussées, Paris, 2018).

## Introduction
This project is a student fork of PointNet2, by Charles R. Qi, Li (Eric) Yi, Hao Su, Leonidas J. Guibas from Stanford University.
You can refer to the original PointNet2 paper and code (https://github.com/charlesq34/pointnet2) for details.

This fork focused on semantic segmentation, with the goal of comparing three datasets : Scannet, Semantic-8 and Bertrand Le Saux aerial LIDAR dataset.
To achieve that, we clean, document, refactor, and improve the original project. 
We will compare the same datasets later with SnapNet, another state-of-the-art semantic segmentation project.

## Dependancies and data
We work on Ubuntu 16.04 with 3 GTX Titan Black and a GTX Titan X. On old GPU, as my GTX 860m, expect to lower the number of point and the batch size for the training.
You have to install TensorFlow on GPU (we use TF 1.2, cuda 8.0, python 2.7, but it should work on newer versions with minor changes). Then, you have to compile the custom TensorFlow operators in the tf_ops subdirectories, with the sh files. You may have to install some Python modules.

Get the preprocessed data :
- Scannet : https://onedrive.live.com/?authkey=%21AHEO5Ik8Hn4Ue2Y&cid=423FEBB4168FD396&id=423FEBB4168FD396%21136&parId=423FEBB4168FD396%21134&action=locate
- Semantic : to do

## TODO
- Use tensorflow Dataset API
- Change/add metrics
- Succeed in training Semantic
