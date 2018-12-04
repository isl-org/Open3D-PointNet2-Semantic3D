import open3d
import time
import sys


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise ValueError("Usage: python visualize.py point_cloud_path.pcd")

    pcd_path = sys.argv[1]
    s = time.time()
    pcd = open3d.read_point_cloud(pcd_path)
    print("time read_point_cloud", time.time() - s, flush=True)
    open3d.draw_geometries([pcd])
