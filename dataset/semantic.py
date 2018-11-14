"""
Unified semantic with color support and many options
"""
import os
import sys

ROOT_DIR = os.path.abspath(os.path.pardir)
sys.path.append(ROOT_DIR)

import numpy as np
import utils.provider as provider


class Dataset:
    def __init__(self, npoints, split, use_color, box_size, path, dropout_max):
        """Create a dataset holder
        npoints (int): Defaults to 8192. The number of point in each input
        split (str): Defaults to 'train'. The selected part of the data (train, test,
                     reduced...)
        color (bool): Defaults to True. Whether to use colors or not
        box_size (int): Defaults to 10. The size of the extracted cube.
        path (float): Defaults to 'dataset/semantic_data/'.
        dropout_max (float): Defaults to 0.875. Maximum dropout to apply on the inputs.
        accept_rate (float): Minimum rate (between 0.0 and 1.0) of points in the box to
                             accept it. E.g : npoints = 100, then you need at least 50
                             points.
        """
        # Dataset parameters
        self.npoints = npoints
        self.split = split
        self.use_color = use_color
        self.box_size = box_size
        self.dropout_max = dropout_max
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
            "scanning artefacts",
            "cars",
        ]
        self.filenames_train = [
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
        self.filenames_test = [
            "sg27_station4_intensity_rgb",
            "sg27_station5_intensity_rgb",
            "sg27_station9_intensity_rgb",
            "sg28_station4_intensity_rgb",
            "untermaederbrunnen_station1_xyz_intensity_rgb",
            "untermaederbrunnen_station3_xyz_intensity_rgb",
        ]
        self.real_test_filenames = [
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

        # Load the data
        self.load_data()

        # Precompute the random scene probabilities, and zmax
        self.compute_random_scene_index_proba()
        self.set_pc_zmax_zmin()

        # Prepare the points weights if it is a training set
        if split == "train" or split == "train_short" or split == "full":
            # Compute the weights
            labelweights = np.zeros(9)
            # First, compute the histogram of each labels
            for seg in self.semantic_labels_list:
                tmp, _ = np.histogram(seg, range(10))
                labelweights += tmp

            # Then, an heuristic gives the weights : 1/log(1.2 + probability of occurrence)
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            self.labelweights = 1 / np.log(1.2 + labelweights)

        elif split == "test" or split == "test_short" or split == "test_full":
            self.labelweights = np.ones(9)

    def load_data(self):
        print("Loading semantic data...")
        if self.split == "train":
            filenames = self.filenames_train
        elif self.split == "test":
            filenames = self.filenames_test
        elif self.split == "full":
            filenames = self.filenames_train + self.filenames_test
        elif self.split == "test_full":
            filenames = self.real_test_filenames
        # train on a smaller, easier dataset to speed up computation
        elif self.split == "train_short":
            filenames = self.filenames_train[0:2]
        elif self.split == "test_short":
            filenames = self.filenames_train[2:3]

        self.data_filenames = [os.path.join(self.path, file) for file in filenames]
        self.scene_points_list = list()
        self.semantic_labels_list = list()
        if self.use_color:
            self.scene_colors_list = list()
        for filename in self.data_filenames:
            data_points = np.load(filename + "_vertices.npz")
            if self.split == "test_full":
                data_labels = np.zeros(len(data_points[data_points.files[0]])).astype(
                    bool
                )
            else:
                data_labels = np.load(filename + "_labels.npz")
            if self.use_color:
                data_colors = np.load(filename + "_colors.npz")
            # sort according to x to speed up computation of boxes and z-boxes
            data_points = data_points[data_points.files[0]]
            if self.split != "test_full":
                data_labels = data_labels[data_labels.files[0]]
            if self.use_color:
                data_colors = data_colors[data_colors.files[0]]
            sort_idx = np.argsort(data_points[:, 0])
            data_points = data_points[sort_idx]
            data_labels = data_labels[sort_idx]
            if self.use_color:
                data_colors = data_colors[sort_idx]
            self.scene_points_list.append(data_points)
            self.semantic_labels_list.append(data_labels.astype(np.int8))
            if self.use_color:
                self.scene_colors_list.append(data_colors)

        # Normalize RGB
        for i in range(len(self.scene_colors_list)):
            self.scene_colors_list[i] = (
                self.scene_colors_list[i].astype("float32") / 255
            )

        # Set min to (0,0,0)
        self.scene_max_list = list()
        self.scene_min_list = list()
        self.raw_scene_min_list = list()
        for i in range(len(self.scene_points_list)):
            self.raw_scene_min_list.append(np.min(self.scene_points_list[i], axis=0))
            self.scene_points_list[i] = self.scene_points_list[i] - np.min(
                self.scene_points_list[i], axis=0
            )
            self.scene_max_list.append(np.max(self.scene_points_list[i], axis=0))
            self.scene_min_list.append(np.min(self.scene_points_list[i], axis=0))

    def __getitem__(self, index):
        """
        input : index of a scene
        output: the whole scene of npointsx3 (xyz) points of the scene and their
                 labels, and colors if colors are used
        """
        point_set = self.scene_points_list[index]
        labels = self.semantic_labels_list[index].astype(np.int32)
        if self.use_color:
            colors = self.scene_colors_list[index]
            return point_set, labels, colors
        return point_set, labels

    def next_batch(self, batch_size, augment=True, dropout=True):
        batch_data = []
        batch_label = []
        batch_weights = []
        feature_size = 0
        for _ in range(batch_size):
            data, label, colors, weights = self.next_input(dropout)
            if self.use_color:
                feature_size = 3
                data = np.hstack((data, colors))
            batch_data.append(data)
            batch_label.append(label)
            batch_weights.append(weights)

        batch_data = np.array(batch_data)
        batch_label = np.array(batch_label)
        batch_weights = np.array(batch_weights)

        # Optional batch augmentation
        if augment and feature_size:
            batch_data = provider.rotate_feature_point_cloud(batch_data, feature_size)
        if augment and not feature_size:
            batch_data = provider.rotate_point_cloud(batch_data)

        return batch_data, batch_label, batch_weights

    def next_input(
        self, dropout=False, sample=True, verbose=False, predicting=False
    ):

        input_ok = False
        count_try = 0

        # Try to find a non-empty cloud to process
        while not input_ok:
            count_try += 1
            # Randomly choose a scene, taking account that some scenes contains more
            # points than others
            scene_index = self.get_random_scene_index()

            # Randomly choose a seed
            scene = self.scene_points_list[scene_index]  # [[x,y,z],...[x,y,z]]
            scene_labels = self.semantic_labels_list[scene_index]
            if self.use_color:
                scene_colors = self.scene_colors_list[scene_index]

            # Random (on points)
            seed_index = np.random.randint(0, len(scene))
            seed = scene[seed_index]  # [x,y,z]

            # Crop a z-box around that seed
            scene_extract_mask = self.extract_z_box(seed, scene, scene_index)
            # Verify the cloud is not empty
            if np.sum(scene_extract_mask) == 0:
                if verbose:
                    print("Warning : empty box")
                continue
            else:
                if verbose:
                    print(
                        "There are %i points in the box" % (np.sum(scene_extract_mask))
                    )
                input_ok = True

            data = scene[scene_extract_mask]
            labels = scene_labels[scene_extract_mask]
            if self.use_color:
                colors = scene_colors[scene_extract_mask]
            else:
                colors = None

        if sample:
            if len(data) - self.npoints > 0:
                trueArray = np.ones(self.npoints, dtype=bool)
                falseArray = np.zeros(len(data) - self.npoints, dtype=bool)
                sample_mask = np.concatenate((trueArray, falseArray), axis=0)
                np.random.shuffle(sample_mask)
            else:
                # Not enough points, recopy the data until there are enough points
                sample_mask = np.arange(len(data))
                while len(sample_mask) < self.npoints:
                    sample_mask = np.concatenate((sample_mask, sample_mask), axis=0)
                sample_mask = sample_mask[np.arange(self.npoints)]
            raw_data = data[sample_mask]

            # Center the box in 2D
            data = self.center_box(raw_data)

            labels = labels[sample_mask]
            if self.use_color:
                colors = colors[sample_mask]

            # Compute the weights
            weights = self.labelweights[labels]

            # Optional dropout
            if dropout:
                drop_index = self.input_dropout(data)
                weights[drop_index] *= 0

        if predicting:
            return (
                scene_index,
                data,
                raw_data + self.raw_scene_min_list[scene_index],
                labels,
                colors,
                weights,
            )
        else:
            return data, labels, colors, weights

    def set_pc_zmax_zmin(self):
        self.pc_zmin = []
        self.pc_zmax = []
        for scene_index in range(len(self)):
            self.pc_zmin.append(np.min(self.scene_points_list[scene_index], axis=0)[2])
            self.pc_zmax.append(np.max(self.scene_points_list[scene_index], axis=0)[2])

    def get_random_scene_index(self):
        # Does not take into account the scene number of points
        # return np.random.randint(0,len(self.scene_points_list))
        return np.random.choice(
            np.arange(0, len(self.scene_points_list)), p=self.scenes_proba
        )

    def compute_random_scene_index_proba(self):
        # Precompute the probability of picking a point
        # in a given scene. This is useful to compute the scene index later,
        # in order to pick more seeds in bigger scenes
        self.scenes_proba = []
        total = self.get_total_num_points()
        for scene_index in range(len(self)):
            proba = float(len(self.scene_points_list[scene_index])) / float(total)
            self.scenes_proba.append(proba)

    def center_box(self, data):
        # Shift the box so that z= 0 is the min and x=0 and y=0 is the center of the
        # box horizontally
        box_min = np.min(data, axis=0)
        shift = np.array(
            [box_min[0] + self.box_size / 2, box_min[1] + self.box_size / 2, box_min[2]]
        )
        return data - shift

    def extract_box(self, seed, scene):
        # 10 meters seems intuitively to be a good value to understand the scene, we
        # must test that

        box_min = seed - [self.box_size / 2, self.box_size / 2, self.box_size / 2]
        box_max = seed + [self.box_size / 2, self.box_size / 2, self.box_size / 2]

        i_min = np.searchsorted(scene[:, 0], box_min[0])
        i_max = np.searchsorted(scene[:, 0], box_max[0])
        mask = (
            np.sum(
                (scene[i_min[0] : i_max, :] >= box_min)
                * (scene[i_min[0] : i_max, :] <= box_max),
                axis=1,
            )
            == 3
        )
        mask = np.hstack(
            (
                np.zeros(i_min, dtype=bool),
                mask,
                np.zeros(len(scene) - i_max, dtype=bool),
            )
        )
        print(mask.shape)
        return mask

    def extract_z_box(self, seed, scene, scene_idx):
        ## TAKES LOT OF TIME !! THINK OF AN ALTERNATIVE !
        # 2D crop, takes all the z axis

        scene_max = self.scene_max_list[scene_idx]
        scene_min = self.scene_min_list[scene_idx]
        scene_z_size = scene_max[2] - scene_min[2]
        box_min = seed - [self.box_size / 2, self.box_size / 2, scene_z_size]
        box_max = seed + [self.box_size / 2, self.box_size / 2, scene_z_size]

        i_min = np.searchsorted(scene[:, 0], box_min[0])
        i_max = np.searchsorted(scene[:, 0], box_max[0])
        mask = (
            np.sum(
                (scene[i_min:i_max, :] >= box_min) * (scene[i_min:i_max, :] <= box_max),
                axis=1,
            )
            == 3
        )
        mask = np.hstack(
            (
                np.zeros(i_min, dtype=bool),
                mask,
                np.zeros(len(scene) - i_max, dtype=bool),
            )
        )

        # mask = np.sum((scene>=box_min)*(scene<=box_max),axis=1) == 3
        return mask

    def input_dropout(self, input):
        dropout_ratio = np.random.random() * self.dropout_max
        drop_index = np.where(np.random.random((input.shape[0])) <= dropout_ratio)[0]
        return drop_index

    def get_total_num_points(self):
        total = 0
        for scene_index in range(len(self)):
            total += len(self.scene_points_list[scene_index])
        return total

    def get_num_batches(self, batch_size):
        return int(self.get_total_num_points() / (batch_size * self.npoints))

    def __len__(self):
        return len(self.scene_points_list)

    def get_data_filenames(self):
        return self.data_filenames

if __name__ == "__main__":
    import multiprocessing as mp
    import time

    data = Dataset(8192, "train", True, 10, "semantic_data", 0, 0)
