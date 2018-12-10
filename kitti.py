import pykitti
import open3d
import time

basedir = "/home/ylao/data/kitti"
date = "2011_09_26"
drive = "0001"

data = pykitti.raw(basedir, date, drive)

pcd = open3d.PointCloud()
pcd.points = open3d.Vector3dVector(data.get_velo(0)[:, :3])

vis = open3d.Visualizer()
vis.create_window()
vis.add_geometry(pcd)

render_option = vis.get_render_option()
render_option.point_size = 0.01

for points_with_intensity in data.velo:
    points = points_with_intensity[:, :3]
    pcd.points = open3d.Vector3dVector(points)
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.01)

vis.destroy_window()
