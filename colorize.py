import open3d
import os
from util.point_cloud_util import load_labels, colorize_point_cloud
import time
import glob

if __name__ == "__main__":
    gt_dir = "dataset/semantic_raw"
    pd_dir = "result/dense"
    colorized_pd_dir = "result/dense_colorized"
    colorized_gt_dir = "result/dense_colorized_gt"

    pd_labels_paths = glob.glob(os.path.join(pd_dir, "*.labels"))
    for pd_labels_path in pd_labels_paths:
        print("Processing", pd_labels_path)
        file_prefix = os.path.basename(os.path.splitext(pd_labels_path)[0])

        # Input pcd
        pcd_path = os.path.join(gt_dir, file_prefix + ".pcd")
        pcd = open3d.read_point_cloud(pcd_path)

        # Colorize by gt labels
        gt_labels_path = os.path.join(gt_dir, file_prefix + ".labels")
        colorized_gt_path = os.path.join(colorized_gt_dir, file_prefix + ".pcd")
        try:
            gt_labels = load_labels(gt_labels_path)
        except:
            gt_labels = None
        if gt_labels is not None and not os.path.isfile(colorized_gt_path):
            s = time.time()
            colorize_point_cloud(pcd, gt_labels)
            print("time colorize_point_cloud gt", time.time() - s, flush=True)
            open3d.write_point_cloud(colorized_gt_path, pcd)

        # Colorize by predicted labels
        colorized_pd_path = os.path.join(colorized_pd_dir, file_prefix + ".pcd")
        pd_labels = load_labels(pd_labels_path)
        s = time.time()
        colorize_point_cloud(pcd, pd_labels)
        print("time colorize_point_cloud pd", time.time() - s, flush=True)
        open3d.write_point_cloud(colorized_pd_path, pcd)
