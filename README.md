# Semantic3D semantic segmentation with Open3D and PointNet++

## Intro

Demo project for Semantic3D (semantic-8) segmentation with
[Open3D](https://github.com/IntelVCL/Open3D) and PointNet++. The
purpose of this project is to showcase the usage of Open3D in deep learning pipelines
and provide a clean baseline implementation for semantic segmentation on Semantic3D
dataset.  Here's
[our entry](http://www.semantic3d.net/view_method_detail.php?method=PointNet2_Demo)
on the semantic-8 test benchmark page.


[Open3D](https://github.com/IntelVCL/Open3D) is an open-source library that supports
rapid development of software that deals with 3D data. The Open3D frontend exposes a
set of carefully selected data structures and algorithms in both C++ and Python. The
backend is highly optimized and is set up for parallelization. We welcome contributions
from the open-source community.

In this project, Open3D was used for
- Point cloud data loading, writing, and visualization. Open3D provides efficient
  implementations of various point cloud manipulation methods.
- Data pre-processing, in particular, voxel-based down-sampling.
- Point cloud interpolation, in particular, fast nearest neighbor search for label
  interpolation.
- And more.

This project is forked from Mathieu Orhan and Guillaume Dekeyser's
[repo](https://github.com/mathieuorhan/pointnet2_semantic), which, is forked
from the original [PointNet2](https://github.com/charlesq34/pointnet2). We thank the
original authors for sharing their methods.

## Usage

### 1. Download

Download the dataset [Semantic3D](http://www.semantic3d.net/view_dbase.php) and extract it by running the following commands: 

`cd dataset/semantic_raw`.

`bash download_semantic3d.sh`.

```shell
Open3D-PointNet2-Semantic3D/dataset/semantic_raw
├── bildstein_station1_xyz_intensity_rgb.labels
├── bildstein_station1_xyz_intensity_rgb.txt
├── bildstein_station3_xyz_intensity_rgb.labels
├── bildstein_station3_xyz_intensity_rgb.txt
├── ...
```

### 2. Convert txt to pcd file

Run

```shell
python preprocess.py
```

Open3D is able to read `.pcd` files much more efficiently.

```shell
Open3D-PointNet2-Semantic3D/dataset/semantic_raw
├── bildstein_station1_xyz_intensity_rgb.labels
├── bildstein_station1_xyz_intensity_rgb.pcd (new)
├── bildstein_station1_xyz_intensity_rgb.txt
├── bildstein_station3_xyz_intensity_rgb.labels
├── bildstein_station3_xyz_intensity_rgb.pcd (new)
├── bildstein_station3_xyz_intensity_rgb.txt
├── ...
```

### 3. Downsample

Run

```shell
python downsample.py
```

The downsampled dataset will be written to `dataset/semantic_downsampled`. Points with
label 0 (unlabled) are excluded during downsampling.

```shell
Open3D-PointNet2-Semantic3D/dataset/semantic_downsampled
├── bildstein_station1_xyz_intensity_rgb.labels
├── bildstein_station1_xyz_intensity_rgb.pcd
├── bildstein_station3_xyz_intensity_rgb.labels
├── bildstein_station3_xyz_intensity_rgb.pcd
├── ...
```

### 4. Compile TF Ops
We need to build TF kernels in `tf_ops`. First, activate the virtualenv and make
sure TF can be found with current python. The following line shall run without
error.

```shell
python -c "import tensorflow as tf"
```

Then build TF ops. You'll need CUDA and CMake 3.8+.

```shell
cd tf_ops
mkdir build
cd build
cmake ..
make
```

After compilation the following `.so` files shall be in the `build` directory.

```shell
Open3D-PointNet2-Semantic3D/tf_ops/build
├── libtf_grouping.so
├── libtf_interpolate.so
├── libtf_sampling.so
├── ...
```

Verify that that the TF kernels are working by running

```shell
cd .. # Now we're at Open3D-PointNet2-Semantic3D/tf_ops
python test_tf_ops.py
```

### 5. Train

Run

```shell
python train.py
```

By default, the training set will be used for training and the validation set
will be used for validation. To train with both training and validation set,
use the `--train_set=train_full` flag. Checkpoints will be output to
`log/semantic`.

### 6. Predict

Pick a checkpoint and run the `predict.py` script. The prediction dataset is
configured by `--set`. Since PointNet2 only takes a few thousand points per
forward pass, we need to sample from the prediction dataset multiple times to
get a good coverage of the points. Each sample contains the few thousand points
required by PointNet2. To specify the number of such samples per scene, use the
`--num_samples` flag.

```shell
python predict.py --ckpt log/semantic/best_model_epoch_040.ckpt \
                  --set=validation \
                  --num_samples=500
```

The prediction results will be written to `result/sparse`.

```
Open3D-PointNet2-Semantic3D/result/sparse
├── sg27_station4_intensity_rgb.labels
├── sg27_station4_intensity_rgb.pcd
├── sg27_station5_intensity_rgb.labels
├── sg27_station5_intensity_rgb.pcd
├── ...
```

### 7. Interpolate

The last step is to interpolate the sparse prediction to the full point cloud.
We use Open3D's K-NN hybrid search with specified radius.

```shell
python interpolate.py
```

The prediction results will be written to `result/dense`.

```shell
Open3D-PointNet2-Semantic3D/result/dense
├── sg27_station4_intensity_rgb.labels
├── sg27_station5_intensity_rgb.labels
├── ...
```

### 8. Submission

Finally, if you're submitting to Semantic3D benchmark, we've included a handy
tools to rename the submission file names.

```shell
python renamer.py
```

### Summary of directories

- `dataset/semantic_raw`: Raw Semantic3D data, .txt and .labels files. Also contains the
  .pcd file generated by `preprocess.py`.
- `dataset/semantic_downsampled`: Generated from `downsample.py`. Downsampled data,
   contains .pcd and .labels files.
- `result/sparse`: Generated from `predict.py`. Sparse predictions, contains .pcd and
   .labels files.
- `result/dense`: Dense predictions, contains .labels files.
- `result/dense_label_colorized`: Dense predictions with points colored by label type.
