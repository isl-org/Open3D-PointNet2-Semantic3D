"""
Unified semantic with color support and many options
"""
import os
import sys
ROOT_DIR = os.path.abspath(os.path.pardir)
sys.path.append(ROOT_DIR)

import numpy as np
import utils.provider as provider

class Dataset():

    def __init__(self, npoints, split, use_color, box_size, path, dropout_max, accept_rate):
        """Create a dataset holder
            npoints (int): Defaults to 8192. The number of point in each input
            split (str): Defaults to 'train'. The selected part of the data (train, test, reduced...)
            color (bool): Defaults to True. Whether to use colors or not
            box_size (int): Defaults to 10. The size of the extracted cube.
            path (float): Defaults to 'dataset/semantic_data/'. 
            dropout_max (float): Defaults to 0.875. Maximum dropout to apply on the inputs.
            accept_rate (float): Minimum rate (between 0.0 and 1.0) of points in the box to accept it. E.g : npoints = 100, then you need at least 50 points.
        """
        # Dataset parameters
        self.npoints = npoints
        self.split = split
        self.use_color = use_color
        self.box_size = box_size
        self.dropout_max = dropout_max
        self.num_classes = 9
        self.path = path
        self.accept_rate = accept_rate
        self.labels_names = ['unlabeled', 'man-made terrain', 'natural terrain', 'high vegetation', 'low vegetation', 'buildings', 'hard scape', 'scanning artefacts', 'cars']
        self.filenames_test = [
            "sg27_station4_intensity_rgb",
            "sg27_station5_intensity_rgb",
            "sg27_station9_intensity_rgb",
            "sg28_station4_intensity_rgb",
            "untermaederbrunnen_station1_xyz_intensity_rgb",
            "untermaederbrunnen_station3_xyz_intensity_rgb"
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
            "sg27_station2_intensity_rgb"
            ]

        # Load the data
        self.load_data()
        
        # Prepare the points weights if it is a training set
        if split=='train' or split=='train_short':
            # Compute the weights
            labelweights = np.zeros(9)
            # First, compute the histogram of each labels
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(10))
                labelweights += tmp
            
            # Then, an heuristic gives the weights : 1/log(1.2 + probability of occurrence)
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
            
        elif split=='test' or split=='test_short':
            self.labelweights = np.ones(9)

    def load_data(self):
        if self.split=='train':
            filenames = self.filenames_train
        elif self.split=='test':
            filenames = self.filenames_test
        # train on a smaller, easier dataset to speed up computation
        elif self.split=='train_short':
            filenames = self.filenames_train[0:2]
        elif self.split=='test_short':
            filenames = self.filenames_train[2:3]

        self.data_filenames = [os.path.join(self.path, file) for file in filenames]
        self.scene_points_list = list()
        self.semantic_labels_list = list()
        if self.use_color:
            self.scene_colors_list = list()
        for filename in self.data_filenames:
            data_points = np.load(filename + "_vertices.npz")
            data_labels = np.load(filename + "_labels.npz")
            if self.use_color:
                data_colors = np.load(filename + "_colors.npz")
            self.scene_points_list.append(data_points[data_points.files[0]])
            self.semantic_labels_list.append(data_labels[data_labels.files[0]])
            if self.use_color:
                self.scene_colors_list.append(data_colors[data_colors.files[0]])

    def __getitem__(self, index):
        """
        input : index of a scene
        output : the whole scene of npointsx3 (xyz) points of the scene and their labels, and colors if colors are used
        """
        point_set = self.scene_points_list[index]       
        labels = self.semantic_labels_list[index].astype(np.int32)
        if self.use_color:
            colors = self.scene_colors_list[index]
            return point_set, labels, colors
        return point_set, labels

    def next_batch(self,batch_size,augment=True,dropout=True):
        batch_data = []
        batch_label = []
        batch_weights = []

        for _ in range(batch_size):
            data, label, colors, weights = self.next_input()
            if self.use_color:
                data = np.hstack((data, colors))
            batch_data.append(data)
            batch_label.append(label)
            batch_weights.append(weights)

        batch_data = np.array(batch_data)
        batch_label = np.array(batch_label)
        batch_weights = np.array(batch_weights)

        # Optional batch augmentation
        if augment and self.use_color:
            batch_data = provider.rotate_colored_point_cloud(batch_data)
        if augment and not self.use_color:
            batch_data = provider.rotate_point_cloud(batch_data)

        return batch_data, batch_label, batch_weights

    def next_input(self,dropout=False,sample=True, verbose=False, visu=False):
        input_ok = False
        count_try = 0
        verbose = False

        # Try to find a non-empty cloud to process
        while not input_ok:     
            count_try += 1
            # Randomly choose a scene
            scene_index = np.random.randint(0,len(self.scene_points_list))
            # Randomly choose a seed
            scene = self.scene_points_list[scene_index] # [[x,y,z],...[x,y,z]]
            scene_labels = self.semantic_labels_list[scene_index]
            if self.use_color:
                scene_colors = self.scene_colors_list[scene_index]

            # Random (on points)
            #seed_index = np.random.randint(0,len(scene))
            #seed = scene[seed_index] # [x,y,z]

            # Random (space)
            scene_max = np.max(scene,axis=0)
            scene_min = np.min(scene,axis=0)
            seed = np.random.uniform(scene_min,scene_max,3)
            # Crop a z-box around that seed
            scene_extract_mask = self.extract_z_box(seed,scene)
            # Verify the cloud is not empty
            if np.sum(scene_extract_mask) == 0:
                if verbose:
                    print ("Warning : empty box")
                continue
            else:
                if verbose:
                    print ("There are %i points in the box" %(np.sum(scene_extract_mask)))
                input_ok = True
                if visu:
                    return scene_index, scene_extract_mask, np.histogram(scene_labels[scene_extract_mask], range(10))[0], seed

            data = scene[scene_extract_mask]
            labels = scene_labels[scene_extract_mask]
            if self.use_color:
                colors = scene_colors[scene_extract_mask]
            else:
                colors = None

        if sample:
            if len(data) - self.npoints > 0:
                trueArray = np.ones(self.npoints, dtype = bool)
                falseArray = np.zeros(len(data) - self.npoints, dtype = bool)
                sample_mask = np.concatenate((trueArray, falseArray), axis=0)
                np.random.shuffle(sample_mask)
            else:
                # Not enough points, recopy the data until there are enough points
                sample_mask = np.arange(len(data))
                while (len(sample_mask) < self.npoints):
                    sample_mask = np.concatenate((sample_mask, sample_mask), axis=0)
                sample_mask = sample_mask[np.arange(self.npoints)]
            data = data[sample_mask]
            labels = labels[sample_mask]
            if self.use_color:
                colors = colors[sample_mask]

            # Compute the weights
            weights = self.labelweights[labels]

            # Optional dropout
            if dropout:
                drop_index = self.input_dropout(data)
                weights[drop_index] *= 0

        return data, labels, colors, weights

    def extract_box(self,seed,scene):
        # 10 meters seems intuitivly to be a good value to understand the scene, we must test that

        box_min = seed - [self.box_size/2, self.box_size/2, self.box_size/2]
        box_max = seed + [self.box_size/2, self.box_size/2, self.box_size/2]

        mask = np.sum((scene>=box_min)*(scene<=box_max),axis=1) == 3
        return mask

    def extract_z_box(self,seed,scene):
        # 2D crop, takes all the z axis
        scene_max = np.max(scene,axis=0)
        scene_min = np.min(scene,axis=0)
        scene_z_size = scene_max[2]-scene_min[2]
        box_min = seed - [self.box_size/2, self.box_size/2, scene_z_size]
        box_max = seed + [self.box_size/2, self.box_size/2, scene_z_size]

        mask = np.sum((scene>=box_min)*(scene<=box_max),axis=1) == 3
        return mask

    def input_dropout(self,input):
        dropout_ratio = np.random.random()*self.dropout_max
        drop_index = np.where(np.random.random((input.shape[0]))<=dropout_ratio)[0]
        return drop_index
    
    def get_total_num_points(self):
        total = 0
        for scene_index in range(len(self)):
            total += len(self.scene_points_list[scene_index]) 
        return total

    def get_num_batches(self, batch_size):
        return int(self.get_total_num_points()/(batch_size*self.npoints))*4

    def __len__(self):
        return len(self.scene_points_list)
    
    def get_hist(self):
        labelweights = np.zeros(9)
            # First, compute the histogram of each labels
        for seg in self.semantic_labels_list:
            tmp,_ = np.histogram(seg,range(10))
            labelweights += tmp
        return labelweights
    
    def get_list_classes_str(self):
        return 'unlabeled, man-made terrain, natural terrain, high vegetation, low vegetation, buildings, hard scape, scanning artefacts, cars'

    def get_data_filenames(self):
        return self.data_filenames
        
    def get_scene_shape(self, scene_index):
        return self.scene_points_list[scene_index].shape
        
    def get_scene(self, scene_index):
        return self.scene_points_list[scene_index]

