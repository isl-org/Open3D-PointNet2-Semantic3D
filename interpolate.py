import argparse
import os
import numpy as np
import open3d
import time
import multiprocessing
import tensorflow as tf

from util.metric import ConfusionMatrix
from util.point_cloud_util import load_labels, write_labels
from dataset.semantic_dataset import map_name_to_file_prefixes
from pprint import pprint
from tf_ops.tf_interpolate import interpolate_label_with_color


class Interpolator:
    def __init__(self):
        pl_sparse_points = tf.placeholder(tf.float32, (None, 3))
        pl_sparse_labels = tf.placeholder(tf.int32, (None,))
        pl_dense_points = tf.placeholder(tf.float32, (None, 3))
        pl_knn = tf.placeholder(tf.int32, ())
        dense_labels, dense_colors = interpolate_label_with_color(
            pl_sparse_points, pl_sparse_labels, pl_dense_points, pl_knn
        )
        self.ops = {
            "pl_sparse_points": pl_sparse_points,
            "pl_sparse_labels": pl_sparse_labels,
            "pl_dense_points": pl_dense_points,
            "pl_knn": pl_knn,
            "dense_labels": dense_labels,
            "dense_colors": dense_colors,
        }
        self.sess = tf.Session()

    def interpolate_labels(self, sparse_points, sparse_labels, dense_points, knn=3):
        return self.sess.run(
            [self.ops["dense_labels"], self.ops["dense_colors"]],
            feed_dict={
                self.ops["pl_sparse_points"]: sparse_points,
                self.ops["pl_sparse_labels"]: sparse_labels,
                self.ops["pl_dense_points"]: dense_points,
                self.ops["pl_knn"]: knn,
            },
        )


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="validation", help="train, validation, test")
    flags = parser.parse_args()

    # Directories
    sparse_dir = "result/sparse"
    dense_dir = "result/dense"
    gt_dir = "dataset/semantic_raw"
    os.makedirs(dense_dir, exist_ok=True)

    # Parameters
    radius = 0.2
    k = 20

    # Global statistics
    cm_global = ConfusionMatrix(9)
    interpolator = Interpolator()

    for file_prefix in map_name_to_file_prefixes[flags.set]:
        print("Interpolating:", file_prefix, flush=True)

        # Paths
        sparse_points_path = os.path.join(sparse_dir, file_prefix + ".pcd")
        sparse_labels_path = os.path.join(sparse_dir, file_prefix + ".labels")
        dense_points_path = os.path.join(gt_dir, file_prefix + ".pcd")
        dense_labels_path = os.path.join(dense_dir, file_prefix + ".labels")
        dense_points_colored_path = os.path.join(
            dense_dir, file_prefix + "_colored.pcd"
        )
        dense_gt_labels_path = os.path.join(gt_dir, file_prefix + ".labels")

        # Sparse points
        sparse_pcd = open3d.read_point_cloud(sparse_points_path)
        sparse_points = np.asarray(sparse_pcd.points)
        del sparse_pcd
        print("sparse_points loaded", flush=True)

        # Sparse labels
        sparse_labels = load_labels(sparse_labels_path)
        print("sparse_labels loaded", flush=True)

        # Dense points
        dense_pcd = open3d.read_point_cloud(dense_points_path)
        dense_points = np.asarray(dense_pcd.points)
        print("dense_points loaded", flush=True)

        # Dense Ground-truth labels
        try:
            dense_gt_labels = load_labels(os.path.join(gt_dir, file_prefix + ".labels"))
            print("dense_gt_labels loaded", flush=True)
        except:
            print("dense_gt_labels not found, treat as test set")
            dense_gt_labels = None

        # Assign labels
        start = time.time()
        dense_labels, dense_colors = interpolator.interpolate_labels(
            sparse_points, sparse_labels, dense_points
        )
        print("KNN interpolation time: ", time.time() - start, "seconds", flush=True)

        # Write dense labels
        write_labels(dense_labels_path, dense_labels)
        print("Dense labels written to:", dense_labels_path, flush=True)

        # Write dense point cloud with color
        dense_pcd.colors = open3d.Vector3dVector(dense_colors)
        open3d.write_point_cloud(dense_points_colored_path, dense_pcd)
        print("Dense pcd with color written to:", dense_points_colored_path, flush=True)

        # Eval
        if dense_gt_labels is not None:
            cm = ConfusionMatrix(9)
            cm.increment_from_list(dense_gt_labels, dense_labels)
            cm.print_metrics()
            cm_global.increment_from_list(dense_gt_labels, dense_labels)

    pprint("Global results")
    cm_global.print_metrics()
