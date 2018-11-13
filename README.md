# PointNet2 for semantic segmentation of 3d points clouds

## Introduction

- This is a fork of Mathieu Orhan and Guillaume Dekeyser (Ecole des Ponts et
  Chaussées, Paris, 2018)'s
  [repository](https://github.com/mathieuorhan/pointnet2_semantic).
- This project is a student fork of PointNet2, by Charles R. Qi, Li (Eric) Yi,
  Hao Su, Leonidas J. Guibas from Stanford University.
  You can refer to the original PointNet2 paper and
  [code](https://github.com/charlesq34/pointnet2) for details.

This fork focused on semantic segmentation, with the goal of comparing three
datasets : Scannet, Semantic-8 and Bertrand Le Saux aerial LIDAR dataset.
To achieve that, we clean, document, refactor, and improve the original project.
We will compare the same datasets later with SnapNet, another state-of-the-art
semantic segmentation project.

## Semantic 3D dataset

### Dataset split
```
# Train (9)
bildstein_station1_xyz_intensity_rgb
bildstein_station3_xyz_intensity_rgb
bildstein_station5_xyz_intensity_rgb
domfountain_station1_xyz_intensity_rgb
domfountain_station2_xyz_intensity_rgb
domfountain_station3_xyz_intensity_rgb
neugasse_station1_xyz_intensity_rgb
sg27_station1_intensity_rgb
sg27_station2_intensity_rgb


# Validation (6)
sg27_station4_intensity_rgb
sg27_station5_intensity_rgb
sg27_station9_intensity_rgb
sg28_station4_intensity_rgb
untermaederbrunnen_station1_xyz_intensity_rgb
untermaederbrunnen_station3_xyz_intensity_rgb

# Test (15)
birdfountain_station1_xyz_intensity_rgb
castleblatten_station1_intensity_rgb
castleblatten_station5_xyz_intensity_rgb
marketplacefeldkirch_station1_intensity_rgb
marketplacefeldkirch_station4_intensity_rgb
marketplacefeldkirch_station7_intensity_rgb
sg27_station10_intensity_rgb
sg27_station3_intensity_rgb
sg27_station6_intensity_rgb
sg27_station8_intensity_rgb
sg28_station2_intensity_rgb
sg28_station5_xyz_intensity_rgb
stgallencathedral_station1_intensity_rgb
stgallencathedral_station3_intensity_rgb
stgallencathedral_station6_intensity_rgb
```

Currently, for the `Dataset` class:
```
# Todo: refactor this
"train"     == Train
"test"      == Validation
"full"      == Train + Validation
"test_full" == Test
```

See `dataset/semantic.py` for training/validation/test set split.

### Train (with training set)
```bash
python train.py --config semantic.json
```

### Test (with validation set)
```bash
python predict.py \
    --ckpt logs/semantic/best_model_epoch_060.ckpt \
    --dataset=semantic --set=test --config semantic.json

./interpolate.sh $HOME/data/semantic3d \
    $HOME/repo/Open3D-PointNet-Semantic/visu/semantic_test/full_scenes_predictions \
    $HOME/repo/Open3D-PointNet-Semantic/visu/semantic_test/full_scenes_predictions_all_points
```

### Train (with training + validation set)
TBD

### Test (with test set and submit)
TBD

## Dependancies and data
We work on Ubuntu 16.04 with 3 GTX Titan Black and a GTX Titan X. On older GPUs,
like my GTX 860m, you can expect to lower the number of points and the batch
size for the training, otherwise you will get a OutOfMemory from TensorFlow.
You have to install TensorFlow on GPU (we use TF 1.2, cuda 8.0, python 2.7, but
it should also work on newer versions with minor changes). Then, you have to
compile the custom TensorFlow operators in the tf_ops subdirectories, with the
.sh files. You may have to install some additional Python modules.

Get the preprocessed data (you can also preprocess the semantic data from raw
data in the directory dataset/preprocessing) :
- Scannet : https://onedrive.live.com/?authkey=%21AHEO5Ik8Hn4Ue2Y&cid=423FEBB4168FD396&id=423FEBB4168FD396%21136&parId=423FEBB4168FD396%21134&action=locate
- Semantic : https://drive.google.com/file/d/1-l2h3yh1xBAzR2JhqPz-YIzM4vtEOf4W/view?usp=sharing

Compiling the C++ parts if you want to preprocess the data or to calculate
results on the raw data can result in the following error:
`/usr/bin/ld: cannot find -lvtkproj4`, but you can overcome this difficulty by
using this trick (see
[this post](https://github.com/PointCloudLibrary/pcl/issues/1594) for details):
```bash
ln -s /usr/lib/x86_64-linux-gnu/libvtkCommonCore-6.2.so /usr/lib/libvtkproj4.so
```

For downloading the raw data , go into `dataset/` directory and use the command:
```bash
./downloadAndExtractSem8.sh
```

For preprocessing with this raw data and the `voxel_size` you want, go into the
preprocessing directory and use the command:
```bash
./preprocess.sh ../dataset/raw_semantic_data ../dataset/semantic_data 'voxel_size'
```

(with the `voxel_size` you want, in m. default is 0.05)

For training, use
```bash
python train.py --config=your_config --log=your_logs
```

Both scannet and semantic_8 should be trainable.

For interpolating results, first use for example
```bash
predict.py --cloud=true --n=100 --ckpt=your_ckpt --dataset=semantic --set=test
```

Files will be created in `visu/semantic_test/full_scenes_predictions` and will
contain predictions on sparse point clouds. The actual interpolation is done in
interpolation directory with the command:
```
./interpolate path/to/raw/data visu/semantic_test/full_scenes_predictions /path/to/where/to/put/results 'voxel_size'
```
(with the voxel_size you want, in m. default is 0.1)

Please check the source files for more details.
