import os
import open3d
import numpy as np
import util.provider as provider
from dataset.semantic_dataset import SemanticFileData, SemanticDataset
import pykitti


class KittiFileData(SemanticFileData):
    def __init__(self, points, box_size):
        self.box_size = box_size
        self.points = points
        self.pcd = open3d.PointCloud()
        self.pcd.points = open3d.Vector3dVector(self.points)

        # Load label. In pure test set, fill with zeros.
        self.labels = np.zeros(len(self.points)).astype(bool)

        # Load colors. If not use_color, fill with zeros.
        self.colors = np.zeros_like(self.points)

        # Sort according to x to speed up computation of boxes and z-boxes
        sort_idx = np.argsort(self.points[:, 0])
        self.points = self.points[sort_idx]
        self.labels = self.labels[sort_idx]
        self.colors = self.colors[sort_idx]

    def get_batch_of_z_boxes_from_origin(
        self, min_x_box, max_x_box, min_y_box, max_y_box, min_z, max_z
    ):
        """
        Returns a batch of (max_x_box - min_x_box) * (max_y_box - min_y_box) samples,
        where each sample contains num_points_per_sample points.

        min_x_box: lower bound of box index of the x-axis
        max_x_box: upper bound of box index of the x-axis
        min_y_box: lower bound of box index of the y-axis
        max_y_box: upper bound of box index of the y-axis
        min_z: lower bound of z-axis value
        max_z: upper bound of z-axis value

        E.g. box_size = 10, min_x = -3, max_x = 3, min_y = -1, max_y = 1, then
        get_batch_z_boxes_from_origin will return 12 samples, with total coverage
        x: -30 to 30; y: -10 to 10; z: -inf to +inf.
        """
        if (
            not isinstance(min_x_box, int)
            or not isinstance(max_x_box, int)
            or not isinstance(min_y_box, int)
            or not isinstance(max_y_box, int)
        ):
            raise ValueError("Box index bounds must be integers")

        min_x = min_x_box * self.box_size
        max_x = max_x_box * self.box_size
        min_y = min_y_box * self.box_size
        max_y = max_y_box * self.box_size

        region_pcd = open3d.crop_point_cloud(
            self.pcd, [min_x, min_y, min_z], [max_x, max_y, max_z]
        )

        pass


class KittiDataset(SemanticDataset):
    def __init__(self, num_points_per_sample, base_dir, dates, drives, box_size):
        """Create a dataset holder
        num_points_per_sample (int): Defaults to 8192. The number of point per sample
        split (str): Defaults to 'train'. The selected part of the data (train, test,
                     reduced...)
        color (bool): Defaults to True. Whether to use colors or not
        box_size (int): Defaults to 10. The size of the extracted cube.
        path (float): Defaults to 'dataset/semantic_data/'.
        """
        # Dataset parameters
        self.num_points_per_sample = num_points_per_sample
        self.num_classes = 9
        self.labels_names = [
            "unlabeled",
            "man-made terrain",
            "natural terrain",
            "high vegetation",
            "low vegetation",
            "buildings",
            "hard scape",
            "scanning artifact",
            "cars",
        ]
        self.box_size = box_size

        # Load files
        self.list_file_data = []
        for date in dates:
            for drive in drives:
                print("Loading date: {}, drive: {}".format(date, drive))
                pykitti_data = pykitti.raw(base_dir, date, drive)
                frame_idx = 0
                for points_with_intensity in pykitti_data.velo:
                    # Get points
                    points = points_with_intensity[:, :3]
                    # Init file data
                    file_data = KittiFileData(points=points, box_size=box_size)
                    # TODO: just for compatibility reason to include the name
                    file_data.file_path_without_ext = os.path.join(
                        date, drive, "{:04d}".format(frame_idx)
                    )
                    frame_idx += 1
                    self.list_file_data.append(file_data)
