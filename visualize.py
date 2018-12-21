import open3d
import argparse
import os
from util.point_cloud_util import load_labels, colorize_point_cloud
import numpy as np
from util.provider import rotate_point_cloud, rotate_feature_point_cloud


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

    # pcd = open3d.crop_point_cloud(pcd, [-30, -10, -2], [30, 10, 100])
    # batched_points = np.expand_dims(np.asarray(pcd.points), axis=0)
    # batched_points = rotate_point_cloud(batched_points, rotation_axis="y")
    # pcd.points = open3d.Vector3dVector(batched_points[0])

    # Load labels and colorize pcd, if labels available
    if flags.labels_path != "":
        if not os.path.isfile(flags.pcd_path):
            raise ValueError("labels path not found at {}".format(flags.labels_path))
        labels = load_labels(flags.labels_path)
        colorize_point_cloud(pcd, labels)

    # points = np.asarray(pcd.points)
    # colors = np.asarray(pcd.colors)
    # points_with_colors = np.concatenate((points, colors), axis=1)
    # points_with_colors = rotate_feature_point_cloud(points_with_colors)
    # points = points_with_colors[:, :3]
    # colors = points_with_colors[:, 3:]
    # pcd.points = open3d.Vector3dVector(points)
    # pcd.colors = open3d.Vector3dVector(colors)

    open3d.draw_geometries([pcd])
