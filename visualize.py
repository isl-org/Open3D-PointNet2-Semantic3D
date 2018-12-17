import open3d
import numpy as np
import argparse
import os
from util.point_cloud_util import load_labels, colorize_point_cloud


if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd_path", default="", type=str)
    parser.add_argument("--labels_path", default="", type=str)
    flags = parser.parse_args()

    # Load point cloud
    if not os.path.isfile(flags.pcd_path):
        raise ValueError("pcd path not found at {}".format(flags.pcd_path))
    pcd = open3d.read_point_cloud(flags.pcd_path)
    pcd = open3d.crop_point_cloud(pcd, [-30, -10, -2], [30, 10, 5])

    # Load labels and colorize pcd, if labels available
    if flags.labels_path != "":
        if not os.path.isfile(flags.pcd_path):
            raise ValueError("labels path not found at {}".format(flags.labels_path))
        labels = load_labels(flags.labels_path)
        colorize_point_cloud(pcd, labels)

    open3d.draw_geometries([pcd])
