"""
New version of semantic with the following goals :
- no more pre-split
- random seed, we look for a neightbor cube with adapted size
- check that there is enough points to go
- augmentation and dropout handled inside the class
"""

import os
import sys
ROOT_DIR = os.path.abspath(os.path.pardir)
sys.path.append(ROOT_DIR)

import numpy as np
import utils.pc_util as pc_util
import utils.scene_util as scene_util
import utils.provider as provider

# Dataset global parameters

NUM_CLASSES = 9

DATA_PATH = "dataset/semantic_data/"

FILENAMES_TRAIN = [
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
FILENAMES_TEST = [
            "sg27_station4_intensity_rgb",
            "sg27_station5_intensity_rgb",
            "sg27_station9_intensity_rgb",
            "sg28_station4_intensity_rgb",
            "untermaederbrunnen_station1_xyz_intensity_rgb",
            "untermaederbrunnen_station3_xyz_intensity_rgb"
            ]

LABELS_NAMES = ['unlabeled', 'man-made terrain', 'natural terrain', 'high vegetation', 'low vegetation', 'buildings', 'hard scape', 'scanning artefacts', 'cars']

class Dataset():

    def __init__(self, npoints=8192, split='train'):
        self.npoints = npoints
        self.split = split

        # load the data
        self.load_data()
        
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
            filenames = FILENAMES_TRAIN
        elif self.split=='test':
            filenames = FILENAMES_TEST
        # train on a smaller, easier dataset to speed up computation
        elif self.split=='train_short':
            filenames = FILENAMES_TRAIN[0:2]
        elif self.split=='test_short':
            filenames = FILENAMES_TRAIN[2:3]

        self.data_filenames = [os.path.join(DATA_PATH, file) for file in filenames]
        self.scene_points_list = list()
        self.semantic_labels_list = list()
        self.scene_colors_list = list()
        for filename in self.data_filenames:
            data_points = np.load(filename + "_vertices.npz")
            data_labels = np.load(filename + "_labels.npz")
            data_colors = np.load(filename + "_colors.npz")
            self.scene_points_list.append(data_points[data_points.files[0]])
            self.semantic_labels_list.append(data_labels[data_labels.files[0]])
            self.scene_colors_list.append(data_colors[data_colors.files[0]])

    def __getitem__(self, index):
        """
        input : index of a scene
        output : the whole scene of npointsx3 (xyz) points of the scene and their labels
        """
        point_set = self.scene_points_list[index]       
        labels = self.semantic_labels_list[index].astype(np.int32)
        colors = self.scene_colors_list[index]
        return point_set, labels, colors

    def next_batch(self,batch_size,augment=True,dropout=True):
        batch_data = []
        batch_label = []
        batch_weights = []

        for batch in range(batch_size):
            data, label, colors, weights = self.next_input()
            data = np.hstack((data, colors))
            batch_data.append(data)
            batch_label.append(label)
            batch_weights.append(weights)

        batch_data = np.array(batch_data)
        batch_label = np.array(batch_label)
        batch_weights = np.array(batch_weights)

        # Optional batch augmentation
        if augment:
            batch_data = provider.rotate_colored_point_cloud(batch_data)

        return batch_data, batch_label, batch_weights

    def next_input(self,dropout=False,sample=True):
        input_ok = False
        count_try = 0

        # While there is no input satisfying
        while not input_ok:     
            count_try += 1
            # Randomly choose a scene
            scene_index = np.random.randint(0,len(self.scene_points_list))
            # Randomly choose a seed
            scene = self.scene_points_list[scene_index] # [[x,y,z],...[x,y,z]]
            scene_labels = self.semantic_labels_list[scene_index]
            scene_colors = self.scene_colors_list[scene_index]
            seed_index = np.random.randint(0,len(scene))
            seed = scene[seed_index] # [x,y,z]
            # Crop a box around that seed
            scene_extract_mask = self.extract_box(seed,scene)
            # Verify there is enough points inside
            if np.sum(scene_extract_mask) < self.npoints:
                #print "Warning : not enough point in the box (%i points), try=%i." %(np.sum(scene_extract_mask),count_try)
                continue
            else:
                #print "Initially, %i points in the box" %(np.sum(scene_extract_mask))
                input_ok = True

            Input = scene[scene_extract_mask]
            labels = scene_labels[scene_extract_mask]
            colors = scene_colors[scene_extract_mask]

        if sample:
            if len(Input) - self.npoints > 0:
                trueArray = np.ones(self.npoints, dtype = bool)
                falseArray = np.zeros(len(cur_point_set) - self.npoints, dtype = bool)
                sample_mask = np.concatenate((trueArray, falseArray), axis=0)
                np.random.shuffle(sample_mask)
            else:
                # not enough points, recopy the Input until enough points
                sample_mask = np.arange(len(Input))
                while (len(sample_mask) < self.npoints):
                    sample_mask = np.concatenate((sample_mask, sample_mask), axis=0)
                sample_mask = sample_mask[np.arange(self.npoints)]
            Input = Input[sample_mask]
            labels = labels[sample_mask]
            colors = colors[sample_mask]

            # Compute the weights
            weights = self.labelweights[labels]

            # Optional dropout
            if dropout:
                drop_index = self.input_dropout(Input)
                weights[drop_index] *= 0

        return Input, labels, colors, weights

    def extract_box(self,seed,scene):
        # 5 meters seems intuitivly to be a good value to understand the scene, we must test that
        # We want to keep the terrain in most input, so we force to include it in, say, proba_terrain = 80% of the cases
        # on average
        box_size = 15
        proba_terrain = 0.8

        # Compute the min points of the scene
        scene_min = np.min(scene, axis=0)

        if(np.random.random()<proba_terrain):
            seed[2] = scene_min[2] + box_size/2

        box_min = seed - [box_size/2, box_size/2, box_size/2]
        box_max = seed + [box_size/2, box_size/2, box_size/2]

        mask = np.sum((scene>=box_min)*(scene<=box_max),axis=1) == 3
        return mask

    def input_dropout(self,input,dropout_max=0.875):
        dropout_ratio = np.random.random()*dropout_max
        drop_index = np.where(np.random.random((input.shape[0]))<=dropout_ratio)[0]
        return drop_index
    
    def get_total_num_points(self):
        total = 0
        for scene_index in range(len(self)):
            total += len(self.scene_points_list[scene_index]) 
        return total

    def __len__(self):
        return len(self.scene_points_list)

if __name__ == '__main__':
    pass
    # TODO : write tests
