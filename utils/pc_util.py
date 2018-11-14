""" Utility functions for processing point clouds.

Author: Charles R. Qi, Hao Su
Date: November 2016
"""

import numpy as np
from .plyfile import PlyData, PlyElement


def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata["vertex"].data
    pc_array = np.array([[x, y, z] for x, y, z in pc])
    return pc_array


def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [
        (points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])
    ]
    vertex = np.array(points, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    el = PlyElement.describe(vertex, "vertex", comments=["vertices"])
    PlyData([el], text=text).write(filename)


def write_ply_color(points, labels, out_filename, num_classes=None):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    # RGB-A color map
    color_map = {
        0: [255, 0, 0, 255],
        1: [255, 165, 0, 255],
        2: [179, 255, 0, 255],
        3: [7, 255, 0, 255],
        4: [0, 255, 157, 255],
        5: [0, 181, 255, 255],
        6: [0, 15, 255, 255],
        7: [155, 0, 255, 255],
        8: [255, 0, 189, 255],
    }

    labels = labels.astype(int)
    N = points.shape[0]
    fout = open(out_filename, "w")
    for i in range(N):
        c = color_map[labels[i]]
        fout.write(
            "v %f %f %f %d %d %d\n"
            % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2])
        )
    fout.close()


def write_ply_true_color(points, colors, out_filename):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as OBJ file """
    colors = colors.astype(int)
    N = points.shape[0]
    fout = open(out_filename, "w")
    for i in range(N):
        fout.write(
            "v %f %f %f %d %d %d\n"
            % (
                points[i, 0],
                points[i, 1],
                points[i, 2],
                colors[i, 0],
                colors[i, 1],
                colors[i, 2],
            )
        )
    fout.close()
