import open3d
import os
import time


if __name__ == "__main__":
    pcd_dir = "result/dense_colorized"
    file_prefix = "sg27_station3_intensity_rgb"
    pcd_path = os.path.join(pcd_dir, file_prefix + ".pcd")

    s = time.time()
    pcd = open3d.read_point_cloud(pcd_path)
    print("time read_point_cloud", time.time() - s, flush=True)

    open3d.draw_geometries([pcd])
