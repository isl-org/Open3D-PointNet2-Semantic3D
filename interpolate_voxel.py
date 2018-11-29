import os
import numpy as np
import open3d
import time
import multiprocessing

from util.metric import ConfusionMatrix
from util.point_cloud_util import load_labels
from dataset.semantic_dataset import validation_file_prefixes


class LabelCounter:
    """
    A simple but non-optimized implementation
    """

    def __init__(self):
        self.labels = []
        self.majority_label = None
        self.finalized = False

    def increment(self, label):
        if self.finalized:
            raise RuntimeError("Counter finalized")
        self.labels.append(label)

    def get_label(self):
        if not self.finalized:
            self.finalize()
        return self.majority_label

    def finalize(self):
        self.majority_label = np.bincount(self.labels).argmax()
        self.finalized = True


def get_voxel(point, voxel_size=0.1):
    """
    Returns a voxel tuple
    point: [x, y, z]
    """
    return tuple([np.floor(float(val) / voxel_size) + 0.5 for val in point])


if __name__ == "__main__":
    # TODO: handle test set
    sparse_dir = "result/sparse"
    dense_dir = "result/dense"
    gt_dir = "dataset/semantic_raw"

    for file_prefix in validation_file_prefixes:
        print("Interpolating:", file_prefix)

        # Paths
        sparse_points_path = os.path.join(sparse_dir, file_prefix + ".pcd")
        sparse_labels_path = os.path.join(sparse_dir, file_prefix + ".labels")
        dense_points_path = os.path.join(gt_dir, file_prefix + ".pcd")
        dense_gt_labels_path = os.path.join(gt_dir, file_prefix + ".labels")

        # Sparse points
        sparse_pcd = open3d.read_point_cloud(sparse_points_path)
        sparse_points = np.asarray(sparse_pcd.points)
        print("sparse_pcd loaded")

        # Sparse labels
        sparse_labels = load_labels(sparse_labels_path)
        print("sparse_labels loaded")

        # Dense points
        dense_pcd = open3d.read_point_cloud(dense_points_path)
        dense_points = np.asarray(dense_pcd.points)
        print("dense_pcd loaded")

        # Dense Ground-truth labels
        dense_gt_labels = load_labels(os.path.join(gt_dir, file_prefix + ".labels"))
        print("dense_gt_labels loaded")

        # Build voxel to label container map
        map_voxel_to_label_counter = dict()
        for sparse_point, sparse_label in zip(sparse_points, sparse_labels):
            voxel = get_voxel(sparse_point)
            if voxel not in map_voxel_to_label_counter:
                map_voxel_to_label_counter[voxel] = LabelCounter()
            map_voxel_to_label_counter[voxel].increment(sparse_label)
        print(
            "{} sparse points, {} registered voxels".format(
                len(sparse_points), len(map_voxel_to_label_counter)
            )
        )

        # Build KNN tree as fallback if voxel not found
        sparse_pcd_tree = open3d.KDTreeFlann(sparse_pcd)
        print("sparse_pcd_tree ready")

        def interpolate_label(dense_index):
            global dense_points
            global sparse_labels
            global map_voxel_to_label_counter

            dense_point = dense_points[dense_index]
            voxel = get_voxel(dense_point)
            if voxel not in map_voxel_to_label_counter:
                k, sparse_indexes, _ = sparse_pcd_tree.search_knn_vector_3d(
                    dense_point, 10
                )
                knn_sparse_labels = sparse_labels[sparse_indexes]
                dense_label = np.bincount(knn_sparse_labels).argmax()
            else:
                dense_label = map_voxel_to_label_counter[voxel].get_label()
            return dense_label

        # Assign labels
        start = time.time()
        dense_indexes = list(range(len(dense_points)))
        with multiprocessing.Pool() as pool:
            dense_labels = pool.map(interpolate_label, dense_indexes)
        print("knn match time: ", time.time() - start)

        # Eval
        cm = ConfusionMatrix(9)
        cm.increment_from_list(dense_gt_labels, dense_labels)
        cm.print_metrics()
