import os
import open3d
import numpy as np
from dataset.semantic_dataset import SemanticFileData, SemanticDataset
import pykitti


class KittiFileData(SemanticFileData):
    def __init__(self, points, box_size_x, box_size_y):
        self.box_size_x = box_size_x
        self.box_size_y = box_size_y

        # Crop the region of interest centered at origin
        # TODO: This is a special treatment, since we only care about the origin now
        min_z = -2
        max_z = 5
        min_x = -self.box_size_x / 2.0
        max_x = -min_x
        min_y = -self.box_size_y / 2.0
        max_y = -min_y
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(points)
        region_pcd = open3d.crop_point_cloud(
            pcd, [min_x, min_y, min_z], [max_x, max_y, max_z]
        )
        self.points = np.asarray(region_pcd.points)

        # Load label. In pure test set, fill with zeros.
        self.labels = np.zeros(len(self.points)).astype(bool)

        # Load colors. If not use_color, fill with zeros.
        self.colors = np.zeros_like(self.points)

        # Sort according to x to speed up computation of boxes and z-boxes
        sort_idx = np.argsort(self.points[:, 0])
        self.points = self.points[sort_idx]
        self.labels = self.labels[sort_idx]
        self.colors = self.colors[sort_idx]

    def get_batch_of_one_z_box_from_origin(self, num_points_per_sample):
        # This point cloud has already been cropped near the origin
        # extract_mask = self._extract_z_box(np.array([0, 0, 0]))
        # points = self.points[extract_mask]

        sample_mask = self._get_fix_sized_sample_mask(
            self.points, num_points_per_sample
        )
        points = self.points[sample_mask]

        centered_points = self._center_box(points)

        batch_points = np.expand_dims(points, 0)
        centered_batch_points = np.expand_dims(centered_points, 0)
        return centered_batch_points, batch_points


class KittiDataset(SemanticDataset):
    def __init__(
        self, num_points_per_sample, base_dir, dates, drives, box_size_x, box_size_y
    ):
        """Create a dataset holder
        num_points_per_sample (int): Defaults to 8192. The number of point per sample
        split (str): Defaults to 'train'. The selected part of the data (train, test,
                     reduced...)
        color (bool): Defaults to True. Whether to use colors or not
        box_size_x (int): Defaults to 10. The size of the extracted cube.
        box_size_y (int): Defaults to 10. The size of the extracted cube.
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
        self.box_size_x = box_size_x
        self.box_size_y = box_size_y

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
                    file_data = KittiFileData(
                        points=points, box_size_x=box_size_x, box_size_y=box_size_y
                    )
                    # TODO: just for compatibility reason to include the name
                    file_data.file_path_without_ext = os.path.join(
                        date, drive, "{:04d}".format(frame_idx)
                    )
                    frame_idx += 1
                    self.list_file_data.append(file_data)
