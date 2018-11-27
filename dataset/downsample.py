import open3d
import os
import numpy as np
from utils.point_cloud_util import load_labels, write_labels


voxel_size = 0.05
raw_data_dir = "/home/ylao/data/semantic3d"
sparse_data_dir = "/home/ylao/repo/Open3D-PointNet-Semantic/dataset/down_sampled"

train_set = [
    "sg27_station4_intensity_rgb",
    "sg27_station5_intensity_rgb",
    "sg27_station9_intensity_rgb",
    "sg28_station4_intensity_rgb",
    "untermaederbrunnen_station1_xyz_intensity_rgb",
    "untermaederbrunnen_station3_xyz_intensity_rgb",
]
valid_set = [
    "bildstein_station1_xyz_intensity_rgb",
    "bildstein_station3_xyz_intensity_rgb",
    "bildstein_station5_xyz_intensity_rgb",
    "domfountain_station1_xyz_intensity_rgb",
    "domfountain_station2_xyz_intensity_rgb",
    "domfountain_station3_xyz_intensity_rgb",
    "neugasse_station1_xyz_intensity_rgb",
    "sg27_station1_intensity_rgb",
    "sg27_station2_intensity_rgb",
]
test_set = [
    "birdfountain_station1_xyz_intensity_rgb",
    "castleblatten_station1_intensity_rgb",
    "castleblatten_station5_xyz_intensity_rgb",
    "marketplacefeldkirch_station1_intensity_rgb",
    "marketplacefeldkirch_station4_intensity_rgb",
    "marketplacefeldkirch_station7_intensity_rgb",
    "sg27_station10_intensity_rgb",
    "sg27_station3_intensity_rgb",
    "sg27_station6_intensity_rgb",
    "sg27_station8_intensity_rgb",
    "sg28_station2_intensity_rgb",
    "sg28_station5_xyz_intensity_rgb",
    "stgallencathedral_station1_intensity_rgb",
    "stgallencathedral_station3_intensity_rgb",
    "stgallencathedral_station6_intensity_rgb",
]
all_set = train_set + valid_set + test_set


for file_prefix in all_set:
    # Paths
    dense_pcd_path = os.path.join(raw_data_dir, file_prefix + ".pcd")
    dense_label_path = os.path.join(raw_data_dir, file_prefix + ".labels")
    sparse_pcd_path = os.path.join(sparse_data_dir, file_prefix + ".pcd")
    sparse_label_path = os.path.join(sparse_data_dir, file_prefix + ".labels")

    # Skip if done
    if os.path.isfile(sparse_pcd_path) and (
        not os.path.isfile(dense_label_path) or os.path.isfile(sparse_label_path)
    ):
        print("Skipped:", file_prefix)
        continue
    else:
        print("Processing:", file_prefix)

    # Downsample points
    pcd = open3d.read_point_cloud(dense_pcd_path)
    min_bound = pcd.get_min_bound() - voxel_size * 0.5
    max_bound = pcd.get_max_bound() + voxel_size * 0.5

    sparse_pcd, cubics_ids = open3d.voxel_down_sample_and_trace(
        pcd, voxel_size, min_bound, max_bound, False
    )
    print("Number of points before %d" % np.asarray(pcd.points).shape[0])
    print("Number of points after %d" % np.asarray(sparse_pcd.points).shape[0])
    open3d.write_point_cloud(sparse_pcd_path, sparse_pcd)

    # Downsample lables
    try:
        dense_labels = np.array(load_labels(dense_label_path))
    except:
        continue
    sparse_labels = []
    for cubic_ids in cubics_ids:
        cubic_ids = cubic_ids[cubic_ids != -1]
        cubic_labels = dense_labels[cubic_ids]
        sparse_labels.append(np.bincount(cubic_labels).argmax())
    sparse_labels = np.array(sparse_labels)
    write_labels(sparse_label_path, sparse_labels)
