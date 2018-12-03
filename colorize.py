import open3d
import os
from util.point_cloud_util import load_labels, colorize_point_cloud
import time
import glob


def colorize(input_pcd_path, input_labels_path, output_pcd_path):
    pcd = open3d.read_point_cloud(input_pcd_path)
    print("Point cloud loaded from", input_pcd_path)

    labels = load_labels(input_labels_path)
    print("Labels loaded from", input_labels_path)

    s = time.time()
    colorize_point_cloud(pcd, labels)
    print("time colorize_point_cloud pd", time.time() - s, flush=True)

    open3d.write_point_cloud(output_pcd_path, pcd)
    print("Output written to", output_pcd_path)


if __name__ == "__main__":

    # Dense folders
    # gt_dir = "dataset/semantic_raw"
    # pd_dir = "result/dense"
    # colorized_pd_dir = "result/dense_colorized"
    # colorized_gt_dir = "result/dense_colorized_gt"

    # Sparse folders
    gt_dir = "result/sparse"
    pd_dir = "result/sparse"
    colorized_pd_dir = "result/sparse_colorized"
    os.makedirs(colorized_pd_dir, exist_ok=True)

    pd_labels_paths = glob.glob(os.path.join(pd_dir, "*.labels"))
    for pd_labels_path in pd_labels_paths:
        print("Processing", pd_labels_path)
        file_prefix = os.path.basename(os.path.splitext(pd_labels_path)[0])

        # Input pcd
        input_pcd_path = os.path.join(gt_dir, file_prefix + ".pcd")

        # Colorize by predicted labels
        input_labels_path = os.path.join(pd_dir, file_prefix + ".labels")
        if os.path.isfile(input_labels_path):
            output_pcd_path = os.path.join(colorized_pd_dir, file_prefix + ".pcd")
            colorize(input_pcd_path, input_labels_path, output_pcd_path)
