import argparse
import os
import numpy as np
import open3d
import time
import multiprocessing

from util.metric import ConfusionMatrix
from util.point_cloud_util import load_labels, write_labels
from dataset.semantic_dataset import map_name_to_file_prefixes


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="validation", help="train, validation, test")
    flags = parser.parse_args()

    # Directories
    sparse_dir = "result/sparse"
    dense_dir = "result/dense"
    gt_dir = "dataset/semantic_raw"

    # Parameters
    radius = 0.2
    k = 20

    # Global statistics
    cm_global = ConfusionMatrix(9)

    for file_prefix in map_name_to_file_prefixes[flags.set]:
        print("Interpolating:", file_prefix, flush=True)

        # Paths
        sparse_points_path = os.path.join(sparse_dir, file_prefix + ".pcd")
        sparse_labels_path = os.path.join(sparse_dir, file_prefix + ".labels")
        dense_points_path = os.path.join(gt_dir, file_prefix + ".pcd")
        dense_labels_path = os.path.join(dense_dir, file_prefix + ".labels")
        dense_gt_labels_path = os.path.join(gt_dir, file_prefix + ".labels")

        # Sparse points
        sparse_pcd = open3d.read_point_cloud(sparse_points_path)
        print("sparse_pcd loaded", flush=True)
        sparse_pcd_tree = open3d.KDTreeFlann(sparse_pcd)
        del sparse_pcd
        print("sparse_pcd_tree ready", flush=True)

        # Sparse labels
        sparse_labels = load_labels(sparse_labels_path)
        print("sparse_labels loaded", flush=True)

        # Dense points
        dense_pcd = open3d.read_point_cloud(dense_points_path)
        dense_points = np.asarray(dense_pcd.points)
        del dense_pcd
        print("dense_pcd loaded", flush=True)

        # Dense Ground-truth labels
        dense_gt_labels = load_labels(os.path.join(gt_dir, file_prefix + ".labels"))
        print("dense_gt_labels loaded", flush=True)

        def match_knn_label(dense_index):
            global dense_points
            global sparse_labels
            global sparse_pcd_tree
            global radius
            global k

            dense_point = dense_points[dense_index]
            result_k, sparse_indexes, _ = sparse_pcd_tree.search_hybrid_vector_3d(
                dense_point, radius, k
            )
            if result_k == 0:
                result_k, sparse_indexes, _ = sparse_pcd_tree.search_knn_vector_3d(
                    dense_point, k
                )
            knn_sparse_labels = sparse_labels[sparse_indexes]
            dense_label = np.bincount(knn_sparse_labels).argmax()

            return dense_label

        # Assign labels
        start = time.time()
        dense_indexes = list(range(len(dense_points)))
        with multiprocessing.Pool() as pool:
            dense_labels = pool.map(match_knn_label, dense_indexes)
        print("knn match time: ", time.time() - start, flush=True)

        # Write labels
        write_labels(dense_labels_path, dense_labels)
        print("Dense labels written to:", dense_labels_path, flush=True)

        # Eval
        cm = ConfusionMatrix(9)
        cm.increment_from_list(dense_gt_labels, dense_labels)
        cm.print_metrics()
        cm_global.increment_from_list(dense_gt_labels, dense_labels)

    print("Global results")
    cm_global.print_metrics()
