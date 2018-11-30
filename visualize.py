import open3d
import os
from util.point_cloud_util import load_labels, colorize_point_cloud
from util.metric import ConfusionMatrix
import time


if __name__ == '__main__':
    gt_dir = "dataset/semantic_raw"
    pd_dir = "result/dense" # Dense prediction

    file_prefix = "sg27_station4_intensity_rgb"
    gt_labels_path = os.path.join(gt_dir, file_prefix + ".labels")
    pd_labels_path = os.path.join(pd_dir, file_prefix + ".labels")
    pcd_path = os.path.join(gt_dir, file_prefix + ".pcd")

    # Load point cloud and labels

    s = time.time()
    pcd = open3d.read_point_cloud(pcd_path)
    print("time read_point_cloud", time.time() - s, flush=True)

    s = time.time()
    gt_labels = load_labels(gt_labels_path)
    print("time load_labels gt", time.time() - s, flush=True)

    s = time.time()
    pd_labels = load_labels(pd_labels_path)
    print("time load_labels pd", time.time() - s, flush=True)

    # Eval
    s = time.time()
    cm = ConfusionMatrix(9)
    cm.increment_from_list(gt_labels, pd_labels)
    cm.print_metrics()
    print("time eval", time.time() - s, flush=True)

    # Colorize with predicted label
    s = time.time()
    colorize_point_cloud(pcd, pd_labels)
    print("time colorize_point_cloud", time.time() - s, flush=True)
    open3d.draw_geometries([pcd])

    # Colorize with ground-truth label
    s = time.time()
    colorize_point_cloud(pcd, gt_labels)
    print("time colorize_point_cloud", time.time() - s, flush=True)
    open3d.draw_geometries([pcd])
