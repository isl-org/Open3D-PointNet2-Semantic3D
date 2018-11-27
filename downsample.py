import open3d
import os
import numpy as np
from utils.point_cloud_util import load_labels, write_labels
from dataset.semantic_dataset import all_file_prefixes


def down_sample(
    dense_pcd_path, dense_label_path, sparse_pcd_path, sparse_label_path, voxel_size
):
    # Skip if done
    if os.path.isfile(sparse_pcd_path) and (
        not os.path.isfile(dense_label_path) or os.path.isfile(sparse_label_path)
    ):
        print("Skipped:", file_prefix)
        return
    else:
        print("Processing:", file_prefix)

    # Downsample points
    pcd = open3d.read_point_cloud(dense_pcd_path)
    min_bound = pcd.get_min_bound() - voxel_size * 0.5
    max_bound = pcd.get_max_bound() + voxel_size * 0.5

    sparse_pcd, cubics_ids = open3d.voxel_down_sample_and_trace(
        pcd, voxel_size, min_bound, max_bound, False
    )
    print("Number of points before:", np.asarray(pcd.points).shape[0])
    print("Number of points after:", np.asarray(sparse_pcd.points).shape[0])
    print("Point cloud written to:", sparse_pcd_path)
    open3d.write_point_cloud(sparse_pcd_path, sparse_pcd)

    # Downsample labels
    try:
        dense_labels = np.array(load_labels(dense_label_path))
    except:
        return
    sparse_labels = []
    for cubic_ids in cubics_ids:
        cubic_ids = cubic_ids[cubic_ids != -1]
        cubic_labels = dense_labels[cubic_ids]
        sparse_labels.append(np.bincount(cubic_labels).argmax())
    sparse_labels = np.array(sparse_labels)
    write_labels(sparse_label_path, sparse_labels)
    print("Labels written to:", sparse_label_path)


if __name__ == "__main__":
    voxel_size = 0.05
    current_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_dir = os.path.join(current_dir, "dataset")
    raw_dir = os.path.join(dataset_dir, "semantic_raw")
    downsampled_dir = os.path.join(dataset_dir, "semantic_downsampled")

    # Create downsampled_dir
    os.makedirs(downsampled_dir, exist_ok=True)

    for file_prefix in all_file_prefixes:
        # Paths
        dense_pcd_path = os.path.join(raw_dir, file_prefix + ".pcd")
        dense_label_path = os.path.join(raw_dir, file_prefix + ".labels")
        sparse_pcd_path = os.path.join(downsampled_dir, file_prefix + ".pcd")
        sparse_label_path = os.path.join(downsampled_dir, file_prefix + ".labels")

        # Put down_sample in a function for garbage collection
        down_sample(
            dense_pcd_path,
            dense_label_path,
            sparse_pcd_path,
            sparse_label_path,
            voxel_size,
        )
