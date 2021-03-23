import os
import subprocess
import open3d
import pandas as pd
import numpy as np
from dataset.semantic_dataset import all_file_prefixes


def point_cloud_txt_to_pcd(raw_dir, file_prefix):
    # File names
    txt_file = os.path.join(raw_dir, file_prefix + ".txt")
    pcd_file = os.path.join(raw_dir, file_prefix + ".pcd")

    # Skip if already done
    if os.path.isfile(pcd_file):
        print("pcd {} exists, skipped".format(pcd_file))
        return

    # .txt -> .pcd
    print("[txt->pcd]")
    print("txt: {}".format(txt_file))
    print("pcd: {}".format(pcd_file))
    pcd = open3d.geometry.PointCloud()
    point_cloud = ((pd.read_csv(txt_file, index_col=False, header=None, sep=" ")).dropna(axis=1)).values
    pcd.points = open3d.utility.Vector3dVector(point_cloud[:, :3])
    pcd.colors = open3d.utility.Vector3dVector(point_cloud[:, 4:].astype(np.uint8))
    open3d.io.write_point_cloud(pcd_file, pcd)
    print(pcd_file + " DONE!")

    return


if __name__ == "__main__":
    # By default
    # raw data: "dataset/semantic_raw"
    current_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_dir = os.path.join(current_dir, "dataset")
    raw_dir = os.path.join(dataset_dir, "semantic_raw")

    for file_prefix in all_file_prefixes:
        point_cloud_txt_to_pcd(raw_dir, file_prefix)
    print("ALL DONE!")
