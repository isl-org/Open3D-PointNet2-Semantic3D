import os
import open3d
import numpy as np
import util.provider as provider
from dataset.semantic_dataset import SemanticFileData, SemanticDataset
import pykitti


class KittiFileData(SemanticFileData):
    def __init__(self, points, box_size):
        self.points = points
        self.box_size = box_size

        # Shift points to min (0, 0, 0), per-image
        # Training: use the normalized points for training
        # Testing: use the normalized points for testing. However, when writing back
        #          point clouds, the shift should be added back.
        self.points_min_raw = np.min(self.points, axis=0)
        self.points = self.points - self.points_min_raw
        self.points_min = np.min(self.points, axis=0)
        self.points_max = np.max(self.points, axis=0)

        # Load label. In pure test set, fill with zeros.
        self.labels = np.zeros(len(self.points)).astype(bool)

        # Load colors. If not use_color, fill with zeros.
        self.colors = np.zeros_like(self.points)

        # Sort according to x to speed up computation of boxes and z-boxes
        sort_idx = np.argsort(self.points[:, 0])
        self.points = self.points[sort_idx]
        self.labels = self.labels[sort_idx]
        self.colors = self.colors[sort_idx]


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
