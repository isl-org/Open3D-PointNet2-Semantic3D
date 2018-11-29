# Semantic3D segmentation with Open3D and PointNet2

## Reference

This project is forked from Mathieu Orhan and Guillaume Dekeyser's
[repo](https://github.com/mathieuorhan/pointnet2_semantic), which, is forked
from the original [PointNet2](https://github.com/charlesq34/pointnet2).

## Usage

### 1. Download

Download the dataset from [Semantic3D](http://www.semantic3d.net/view_dbase.php)
and extract to `dataset/semantic_raw`.

```shell
Open3D-PointNet-Semantic/dataset/semantic_raw
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
Open3D-PointNet-Semantic/dataset/semantic_raw
├── bildstein_station1_xyz_intensity_rgb.labels
├── bildstein_station1_xyz_intensity_rgb.pcd (new)
├── bildstein_station1_xyz_intensity_rgb.txt
├── bildstein_station3_xyz_intensity_rgb.labels
├── bildstein_station3_xyz_intensity_rgb.pcd (new)
├── bildstein_station3_xyz_intensity_rgb.txt
├── ...
```

### 3. Downsample

Points with label 0 (unlabled) are excluded during downsampling.

```shell
python downsample.py
```

The downsampled dataset will be written to `dataset/semantic_downsampled`.

```shell
Open3D-PointNet-Semantic/dataset/semantic_downsampled
├── bildstein_station1_xyz_intensity_rgb.labels
├── bildstein_station1_xyz_intensity_rgb.pcd
├── bildstein_station3_xyz_intensity_rgb.labels
├── bildstein_station3_xyz_intensity_rgb.pcd
├── ...
```

### 4. Train

First, we'll need to build TF kernels in `tf_ops`. Run `.sh` build scripts for
each op in the `tf_ops` folder respectively. The dataset split is defined in
`dataset/semantic_dataset.py`.

Then, run

```shell
python train.py
```

By default, the training set will be used for training and the validation set
will be used for validation. To train with both training and validation set,
use the `--train_set=train_full` flag. Checkpoints will be output to
`log/semantic`.

### 5. Predict

Pick a checkpoint and run the `predict.py` script. The prediction dataset is
configured by `--set`. Since PointNet2 only takes a few thousand points per
forward pass, we need to sample from the prediction dataset multiple times to
get a good coverage of the points. Each sample contains the few thousand points
required by PointNet2. To specify the number of such samples per scene, use the
`--num_samples` flag.

```shell
python predict.py --ckpt log/semantic/best_model_epoch_040.ckpt \
                  --set=validation \
                  --num_samples=200
```

The prediction results will be written to `result/sparse`.

```
Open3D-PointNet-Semantic/result/sparse
├── sg27_station4_intensity_rgb.labels
├── sg27_station4_intensity_rgb.pcd
├── sg27_station5_intensity_rgb.labels
├── sg27_station5_intensity_rgb.pcd
├── ...
```

### 6. Interpolate

The last step is to interpolate the sparse prediction to the full point cloud.
We use Open3D's K-NN hybrid search with specified radius.

```shell
python interpolate.py
```

The prediction results will be written to `result/dense`.

```
Open3D-PointNet-Semantic/result/dense
├── sg27_station4_intensity_rgb.labels
├── sg27_station4_intensity_rgb.pcd
├── sg27_station5_intensity_rgb.labels
├── sg27_station5_intensity_rgb.pcd
├── ...
```
