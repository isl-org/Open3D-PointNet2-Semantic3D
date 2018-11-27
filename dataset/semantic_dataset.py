import os
import numpy as np
import utils.provider as provider


class SemanticDataset:
    def __init__(self, npoints, split, use_color, box_size, path):
        """Create a dataset holder
        npoints (int): Defaults to 8192. The number of point in each input
        split (str): Defaults to 'train'. The selected part of the data (train, test,
                     reduced...)
        color (bool): Defaults to True. Whether to use colors or not
        box_size (int): Defaults to 10. The size of the extracted cube.
        path (float): Defaults to 'dataset/semantic_data/'.
        """
        # Dataset parameters
        self.npoints = npoints
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
            "scanning artefacts",
            "cars",
        ]
        self.file_names_train = [
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
        self.file_names_test = [
            "sg27_station4_intensity_rgb",
            "sg27_station5_intensity_rgb",
            "sg27_station9_intensity_rgb",
            "sg28_station4_intensity_rgb",
            "untermaederbrunnen_station1_xyz_intensity_rgb",
            "untermaederbrunnen_station3_xyz_intensity_rgb",
        ]
        self.file_names_real_test = [
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

        # Pre-compute the random scene probabilities, and zmax
        self.compute_random_scene_index_proba()
        self.set_pc_zmax_zmin()

        # Prepare the points weights if it is a training set
        if self.split == "train" or self.split == "train_short" or self.split == "full":
            # Compute the weights
            label_weights = np.zeros(9)
            # First, compute the histogram of each labels
            for seg in self.list_labels:
                tmp, _ = np.histogram(seg, range(10))
                label_weights += tmp

            # Then, an heuristic gives the weights : 1/log(1.2 + probability of occurrence)
            label_weights = label_weights.astype(np.float32)
            label_weights = label_weights / np.sum(label_weights)
            self.label_weights = 1 / np.log(1.2 + label_weights)

        elif (
            self.split == "test"
            or self.split == "test_short"
            or self.split == "test_full"
        ):
            self.label_weights = np.ones(9)

    def load_data(self):
        """
        Fills:
            self.list_file_path
            self.list_points
            self.list_points_min_raw
            self.list_points_min
            self.list_points_max
            self.list_labels
            self.list_colors
        """
        print("Loading semantic data...")

        # Get file names to load
        if self.split == "train":
            file_names = self.file_names_train
        elif self.split == "test":
            file_names = self.file_names_test
        elif self.split == "full":
            file_names = self.file_names_train + self.file_names_test
        else:
            assert self.split == "test_full"
            file_names = self.file_names_real_test

        self.list_file_path = [os.path.join(self.path, file) for file in file_names]
        self.list_points = list()
        self.list_labels = list()
        self.list_colors = list()
        self.list_points_max = list()
        self.list_points_min = list()
        self.list_points_min_raw = list()

        for file_path in self.list_file_path:
            # Load points
            points = np.load(file_path + "_vertices.npz")
            points = points[points.files[0]]

            # Shift points to min (0, 0, 0)
            # Training: use the normalized points for training
            # Testing: use the normalized points for testing. However, when writing back
            #          point clouds, the shift should be added back.
            points_min_raw = np.min(points, axis=0)
            points = points - points_min_raw
            points_min = np.min(points, axis=0)
            points_max = np.max(points, axis=0)

            # Load label. In pure test set, fill with zero
            if self.split == "test_full":
                labels = np.zeros(len(points)).astype(bool)
            else:
                labels = np.load(file_path + "_labels.npz")
                labels = labels[labels.files[0]]

            # Load colors, regardless of whether use_color is true
            colors = np.load(file_path + "_colors.npz")
            colors = colors[colors.files[0]]
            colors = colors.astype(np.float32) / 255.0  # Normalize RGB to 0~1

            # Sort according to x to speed up computation of boxes and z-boxes
            sort_idx = np.argsort(points[:, 0])
            points = points[sort_idx]
            labels = labels[sort_idx]
            colors = colors[sort_idx]

            # Append to list
            self.list_points.append(points)
            self.list_points_min_raw.append(points_min_raw)
            self.list_points_min.append(points_min)
            self.list_points_max.append(points_max)
            self.list_labels.append(labels.astype(np.int8))
            self.list_colors.append(colors)

    def next_batch(self, batch_size, augment=True):
        batch_data = []
        batch_label = []
        batch_weights = []
        feature_size = 0
        for _ in range(batch_size):
            data, label, colors, weights = self.next_input()
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

    def next_input(self, sample=True, verbose=False, predicting=False):

        input_ok = False
        count_try = 0

        # Try to find a non-empty cloud to process
        while not input_ok:
            count_try += 1
            # Randomly choose a scene, taking account that some scenes contains more
            # points than others
            scene_index = self.get_random_scene_index()

            # Randomly choose a seed
            scene = self.list_points[scene_index]  # [[x,y,z],...[x,y,z]]
            scene_labels = self.list_labels[scene_index]
            if self.use_color:
                scene_colors = self.list_colors[scene_index]

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
            weights = self.label_weights[labels]

        if predicting:
            return (
                scene_index,
                data,
                raw_data + self.list_points_min_raw[scene_index],
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
            self.pc_zmin.append(np.min(self.list_points[scene_index], axis=0)[2])
            self.pc_zmax.append(np.max(self.list_points[scene_index], axis=0)[2])

    def get_random_scene_index(self):
        # Does not take into account the scene number of points
        # return np.random.randint(0,len(self.list_points))
        return np.random.choice(
            np.arange(0, len(self.list_points)), p=self.scenes_proba
        )

    def compute_random_scene_index_proba(self):
        # Precompute the probability of picking a point
        # in a given scene. This is useful to compute the scene index later,
        # in order to pick more seeds in bigger scenes
        self.scenes_proba = []
        total = self.get_total_num_points()
        for scene_index in range(len(self)):
            proba = float(len(self.list_points[scene_index])) / float(total)
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

        scene_max = self.list_points_max[scene_idx]
        scene_min = self.list_points_min[scene_idx]
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

    def get_total_num_points(self):
        total = 0
        for scene_index in range(len(self)):
            total += len(self.list_points[scene_index])
        return total

    def get_num_batches(self, batch_size):
        return int(self.get_total_num_points() / (batch_size * self.npoints))

    def __len__(self):
        return len(self.list_points)

    def get_data_filenames(self):
        return self.list_file_path
