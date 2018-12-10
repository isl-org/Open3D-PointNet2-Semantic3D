import pykitti
import open3d
import time

basedir = "/home/ylao/data/kitti"
date = "2011_09_26"
drive = "0001"

pcd = open3d.PointCloud()
vis = open3d.Visualizer()
vis.create_window()
render_option = vis.get_render_option()
render_option.point_size = 0.01

data = pykitti.raw(basedir, date, drive)
for points_with_intensity in data.velo:
    points = points_with_intensity[:, :3]
    pcd.points = open3d.Vector3dVector(points)
    vis.add_geometry(pcd)
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.01)

vis.destroy_window()
