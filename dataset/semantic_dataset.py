import os
import open3d
import numpy as np
import util.provider as provider
from util.point_cloud_util import load_labels

train_file_prefixes = [
    "bildstein_station1_xyz_intensity_rgb",
    "bildstein_station3_xyz_intensity_rgb",
    "bildstein_station5_xyz_intensity_rgb",
    "domfountain_station1_xyz_intensity_rgb",
    "domfountain_station2_xyz_intensity_rgb",
    "domfountain_station3_xyz_intensity_rgb",
    "neugasse_station1_xyz_intensity_rgb",
    "sg27_station1_intensity_rgb",
    "sg27_station2_intensity_rgb",
]

validation_file_prefixes = [
    "sg27_station4_intensity_rgb",
    "sg27_station5_intensity_rgb",
    "sg27_station9_intensity_rgb",
    "sg28_station4_intensity_rgb",
    "untermaederbrunnen_station1_xyz_intensity_rgb",
    "untermaederbrunnen_station3_xyz_intensity_rgb",
]

test_file_prefixes = [
    "birdfountain_station1_xyz_intensity_rgb",
    "castleblatten_station1_intensity_rgb",
    "castleblatten_station5_xyz_intensity_rgb",
    "marketplacefeldkirch_station1_intensity_rgb",
    "marketplacefeldkirch_station4_intensity_rgb",
    "marketplacefeldkirch_station7_intensity_rgb",
    "sg27_station10_intensity_rgb",
    "sg27_station3_intensity_rgb",
    "sg27_station6_intensity_rgb",
    "sg27_station8_intensity_rgb",
    "sg28_station2_intensity_rgb",
    "sg28_station5_xyz_intensity_rgb",
    "stgallencathedral_station1_intensity_rgb",
    "stgallencathedral_station3_intensity_rgb",
    "stgallencathedral_station6_intensity_rgb",
]

all_file_prefixes = train_file_prefixes + validation_file_prefixes + test_file_prefixes

map_name_to_file_prefixes = {
    "train": train_file_prefixes,
    "train_full": train_file_prefixes + validation_file_prefixes,
    "validation": validation_file_prefixes,
    "test": test_file_prefixes,
    "all": all_file_prefixes,
}


class FileData:
    def __init__(self, file_prefix, num_points, split, use_color, box_size, file_path):
        """
        Loads file data
        """
        self.file_prefix = file_prefix
        self.file_path = file_path

        # Load points
        pcd = open3d.read_point_cloud(file_path + ".pcd")
        self.points = np.asarray(pcd.points)

        # Shift points to min (0, 0, 0), per-image
        # Training: use the normalized points for training
        # Testing: use the normalized points for testing. However, when writing back
        #          point clouds, the shift should be added back.
        self.points_min_raw = np.min(self.points, axis=0)
        self.points = self.points - self.points_min_raw
        self.points_min = np.min(self.points, axis=0)
        self.points_max = np.max(self.points, axis=0)

        # Load label. In pure test set, fill with zero
        if split == "test":
            self.labels = np.zeros(len(self.points)).astype(bool)
        else:
            self.labels = load_labels(file_path + ".labels")

        # Load colors, regardless of whether use_color is true
        self.colors = np.asarray(pcd.colors)

        # Sort according to x to speed up computation of boxes and z-boxes
        sort_idx = np.argsort(self.points[:, 0])
        self.points = self.points[sort_idx]
        self.labels = self.labels[sort_idx]
        self.colors = self.colors[sort_idx]


class SemanticDataset:
    def __init__(self, num_points, split, use_color, box_size, path):
        """Create a dataset holder
        num_points (int): Defaults to 8192. The number of point in each input
        split (str): Defaults to 'train'. The selected part of the data (train, test,
                     reduced...)
        color (bool): Defaults to True. Whether to use colors or not
        box_size (int): Defaults to 10. The size of the extracted cube.
        path (float): Defaults to 'dataset/semantic_data/'.
        """
        # Dataset parameters
        self.num_points = num_points
        self.split = split
        self.use_color = use_color
        self.box_size = box_size
        self.num_classes = 9
        self.path = path
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

        # Load the data
        print("Loading semantic data:", self.split)

        # Get file names to load
        file_prefixes = map_name_to_file_prefixes[self.split]
        print("Loading file_prefixes:", file_prefixes)

        # self.list_file_path
        # self.list_points
        # self.list_points_min_raw
        # self.list_points_min
        # self.list_points_max
        # self.list_labels
        # self.list_colors
        # Load data to map_prefix_to_file_data
        self.list_file_data = []
        for file_prefix in file_prefixes:
            file_path = os.path.join(self.path, file_prefix)
            file_data = FileData(
                file_prefix,
                self.num_points,
                self.split,
                self.use_color,
                self.box_size,
                file_path,
            )
            self.list_file_data.append(file_data)

        # TODO: remove this
        self.list_points = [fd.points for fd in self.list_file_data]
        self.list_labels = [fd.labels for fd in self.list_file_data]
        self.list_colors = [fd.colors for fd in self.list_file_data]
        self.list_points_max = [fd.points_max for fd in self.list_file_data]
        self.list_points_min = [fd.points_min for fd in self.list_file_data]
        self.list_points_min_raw = [fd.points_min_raw for fd in self.list_file_data]

        # Pre-compute the probability of picking a point
        # in a given scene. This is useful to compute the scene index later,
        # in order to pick more seeds in bigger scenes
        self.scene_probas = []
        total = self.get_total_num_points()
        for scene_index in range(len(self)):
            proba = float(len(self.list_points[scene_index])) / float(total)
            self.scene_probas.append(proba)

        # Prepare the points weights if it is a training set
        if self.split == "train" or self.split == "train_full":
            # Compute the weights
            label_weights = np.zeros(9)

            # First, compute the histogram of each labels
            for seg in self.list_labels:
                tmp, _ = np.histogram(seg, range(10))
                label_weights += tmp

            # Then, an heuristic gives the weights
            # 1 / log(1.2 + probability of occurrence)
            label_weights = label_weights.astype(np.float32)
            label_weights = label_weights / np.sum(label_weights)
            self.label_weights = 1 / np.log(1.2 + label_weights)

    def next_batch(self, batch_size, augment=True):
        batch_data = []
        batch_label = []
        batch_weights = []

        for _ in range(batch_size):
            points, labels, colors, weights = self.next_input(is_training=True)
            if self.use_color:
                batch_data.append(np.hstack((points, colors)))
            else:
                batch_data.append(points)
            batch_label.append(labels)
            batch_weights.append(weights)

        batch_data = np.array(batch_data)
        batch_label = np.array(batch_label)
        batch_weights = np.array(batch_weights)

        if augment:
            if self.use_color:
                batch_data = provider.rotate_feature_point_cloud(batch_data, 3)
            else:
                batch_data = provider.rotate_point_cloud(batch_data)

        return batch_data, batch_label, batch_weights

    def next_input(self, is_training):
        """
        Returns points and other info within a z - cropped box.
        """
        # Pick a scene, scenes with more points are more likely to be chosen
        scene_index = self.get_random_scene_index()
        points = self.list_points[scene_index]

        # Pick a point, and crop a z-box around
        center_point = points[np.random.randint(0, len(points))]
        scene_extract_mask = self.extract_z_box(center_point, points, scene_index)
        points = points[scene_extract_mask]

        # Crop labels and colors
        labels = self.list_labels[scene_index][scene_extract_mask]
        if self.use_color:
            colors = self.list_colors[scene_index][scene_extract_mask]
        else:
            colors = None

        # TODO: change this to numpy's build-in functions
        # Shuffling or up-sampling if needed
        if len(points) - self.num_points > 0:
            true_array = np.ones(self.num_points, dtype=bool)
            false_array = np.zeros(len(points) - self.num_points, dtype=bool)
            sample_mask = np.concatenate((true_array, false_array), axis=0)
            np.random.shuffle(sample_mask)
        else:
            # Not enough points, recopy the data until there are enough points
            sample_mask = np.arange(len(points))
            while len(sample_mask) < self.num_points:
                sample_mask = np.concatenate((sample_mask, sample_mask), axis=0)
            sample_mask = sample_mask[: self.num_points]

        points = points[sample_mask]
        labels = labels[sample_mask]
        if self.use_color:
            colors = colors[sample_mask]

        # Shift the points, such that min(z) == 0, and x = 0 and y = 0 is the center
        # This canonical column is used for both training and inference
        points_centered = self.center_box(points)

        if is_training:
            weights = self.label_weights[labels]
            return points_centered, labels, colors, weights
        else:
            return (
                scene_index,
                points_centered,
                points + self.list_points_min_raw[scene_index],
                labels,
                colors,
            )

    def get_random_scene_index(self):
        # Does not take into account the scene number of points
        # return np.random.randint(0,len(self.list_points))
        return np.random.choice(
            np.arange(0, len(self.list_points)), p=self.scene_probas
        )

    def center_box(self, data):
        # Shift the box so that z = 0 is the min and x = 0 and y = 0 is the box center
        # E.g. if box_size == 10, then the new mins are (-5, -5, 0)
        box_min = np.min(data, axis=0)
        shift = np.array(
            [box_min[0] + self.box_size / 2, box_min[1] + self.box_size / 2, box_min[2]]
        )
        return data - shift

    def extract_z_box(self, center_point, points, scene_idx):
        """
        Crop along z axis (vertical) from the center_point.

        Args:
            center_point: only x and y coordinates will be used
            points: points (n * 3)
            scene_idx: scene index to get the min and max of the whole scene
        """
        # TODO TAKES LOT OF TIME !! THINK OF AN ALTERNATIVE !
        scene_max = self.list_points_max[scene_idx]
        scene_min = self.list_points_min[scene_idx]
        scene_z_size = scene_max[2] - scene_min[2]
        box_min = center_point - [self.box_size / 2, self.box_size / 2, scene_z_size]
        box_max = center_point + [self.box_size / 2, self.box_size / 2, scene_z_size]

        i_min = np.searchsorted(points[:, 0], box_min[0])
        i_max = np.searchsorted(points[:, 0], box_max[0])
        mask = (
            np.sum(
                (points[i_min:i_max, :] >= box_min)
                * (points[i_min:i_max, :] <= box_max),
                axis=1,
            )
            == 3
        )
        mask = np.hstack(
            (
                np.zeros(i_min, dtype=bool),
                mask,
                np.zeros(len(points) - i_max, dtype=bool),
            )
        )

        # mask = np.sum((points>=box_min)*(points<=box_max),axis=1) == 3
        assert np.sum(mask) != 0
        return mask

    def get_total_num_points(self):
        total = 0
        for scene_index in range(len(self)):
            total += len(self.list_points[scene_index])
        return total

    def get_num_batches(self, batch_size):
        return int(self.get_total_num_points() / (batch_size * self.num_points))

    def __len__(self):
        return len(self.list_points)

    def get_file_paths_without_extension(self):
        return [file_data.file_path for file_data in self.list_file_data]
