import argparse
import os
import json
import numpy as np
import tensorflow as tf
import open3d
import time

import model
from dataset.kitti_dataset import KittiDataset
from util.metric import ConfusionMatrix
from predict import Predictor

if __name__ == "__main__":
    np.random.seed(0)

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="# samples, each contains num_point points",
    )
    parser.add_argument("--ckpt", default="", help="Checkpoint file")
    flags = parser.parse_args()
    hyper_params = json.loads(open("semantic.json").read())

    # Create output dir
    output_dir = os.path.join("result", "sparse")
    os.makedirs(output_dir, exist_ok=True)

    # Dataset
    dataset = KittiDataset(
        num_points_per_sample=hyper_params["num_point"],
        base_dir="/home/ylao/data/kitti",
        dates=["2011_09_26"],
        drives=["0001", "0095"],
    )

    # Model
    batch_size = 64
    predictor = Predictor(
        checkpoint_path=flags.ckpt,
        num_classes=dataset.num_classes,
        hyper_params=hyper_params,
    )

    for kitti_file_data in dataset.list_file_data:
        print("Processing {}".format(kitti_file_data.file_path_without_ext))

    #     # Predict for num_samples times
    #     points_raw_collector = []
    #     pd_labels_collector = []
    #
    #     # If flags.num_samples < batch_size, will predict one batch
    #     for batch_index in range(int(np.ceil(flags.num_samples / batch_size))):
    #         current_batch_size = min(
    #             batch_size, flags.num_samples - batch_index * batch_size
    #         )
    #
    #         # Get data
    #         points, points_raw, gt_labels, colors = semantic_file_data.sample_batch(
    #             batch_size=current_batch_size,
    #             num_points_per_sample=hyper_params["num_point"],
    #         )
    #
    #         # (bs, 8192, 3) concat (bs, 8192, 3) -> (bs, 8192, 6)
    #         if hyper_params["use_color"]:
    #             points_with_colors = np.concatenate((points, colors), axis=-1)
    #         else:
    #             points_with_colors = points
    #
    #         # Predict
    #         s = time.time()
    #         pd_labels = predictor.predict(points_with_colors)
    #         print(
    #             "Batch size: {}, time: {}".format(current_batch_size, time.time() - s)
    #         )
    #
    #         # Save to collector for file output
    #         points_raw_collector.extend(points_raw)
    #         pd_labels_collector.extend(pd_labels)
    #
    #         # Increment confusion matrix
    #         cm.increment_from_list(gt_labels.flatten(), pd_labels.flatten())
    #
    #     # Save sparse point cloud and predicted labels
    #     file_prefix = os.path.basename(semantic_file_data.file_path_without_ext)
    #
    #     points_raw_collector = np.array(points_raw_collector)
    #     pcd = open3d.PointCloud()
    #     pcd.points = open3d.Vector3dVector(points_raw_collector.reshape((-1, 3)))
    #     pcd_path = os.path.join(output_dir, file_prefix + ".pcd")
    #     open3d.write_point_cloud(pcd_path, pcd)
    #     print("Exported pcd to {}".format(pcd_path))
    #
    #     pd_labels_collector = np.array(pd_labels_collector).astype(int)
    #     pd_labels_path = os.path.join(output_dir, file_prefix + ".labels")
    #     np.savetxt(pd_labels_path, pd_labels_collector.flatten(), fmt="%d")
    #     print("Exported labels to {}".format(pd_labels_path))
    #
    # cm.print_metrics()
