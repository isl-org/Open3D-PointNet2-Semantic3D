import os
import sys
ROOT_DIR = os.path.abspath(os.path.pardir)
sys.path.append(ROOT_DIR)

import numpy as np
import utils.pc_util as pc_util
import utils.scene_util as scene_util

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

def coupageDeSceneEnN(point_set, labels, sceneIndex, verbose=False):
    """
    point_cloud is an array contaning the 3D points, labels is an array containig the labels
    we cut it in a 2D-grid (seen from above)
    """
    N = 14 # was 14
    coordmax = np.max(point_set,axis=0)
    coordmin = np.min(point_set,axis=0)
    point_cloud_scenes = list()
    label_scenes = list()
    reject = 0
    for i in range(N):
        for j in range(N):
            current_coordmin = np.array([i/float(N), j/float(N), 0])*coordmax + np.array([1-i/float(N), 1-j/float(N), 1])*coordmin
            current_coordmax = np.array([(i+1)/float(N), (j+1)/float(N), 1])*coordmax + np.array([1-(i+1)/float(N), 1-(j+1)/float(N), 0])*coordmin
            #if verbose: print(i, j, current_coordmin, current_coordmax)
            curchoice = np.sum((point_set >= (current_coordmin-0.2))*(point_set < (current_coordmax+0.2)),axis=1)==3
            
            if len(point_set[curchoice,:]) > 10000:
                #if verbose: print("ok")
                point_cloud_scenes.append(point_set[curchoice,:])
                label_scenes.append(np.reshape(labels[curchoice,:], (-1)))
            else:
                reject += 1
    #print "REJECTION : " + str(reject)
    return point_cloud_scenes, label_scenes

   
class SemanticDataset():
    """
    Intended to work as a container (array-like) for the dataset 
    scene_points_list : a list of arrays of size (nbPointsAfterVoxelisation, 3)
    semantic_labels_list : a list of arrays of shape (nbPointsAfterVoxelisation)
    """
    def __init__(self, root, npoints=8192, split='train'):
        self.npoints = npoints
        self.root = root
        self.split = split

        # load the data
        self.load_data()
        
        temp1 = list()
        temp2 = list()
        for k in range(len(self.scene_points_list)):
            scene_pts_list, lbl_list = coupageDeSceneEnN(self.scene_points_list[k], np.reshape(self.semantic_labels_list[k], (-1, 1)), k)
            temp1 += scene_pts_list
            temp2 += lbl_list
        self.scene_points_list, self.semantic_labels_list = temp1, temp2
        
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
        for filename in self.data_filenames:
            data_points = np.load(filename + "_vertices.npz")
            data_labels = np.load(filename + "_labels.npz")
            self.scene_points_list.append(data_points[data_points.files[0]])
            self.semantic_labels_list.append(data_labels[data_labels.files[0]])

    def __getitem__(self, index):
        """
        input : index of a scene
        ouput : a selection of npointsx3 (xyz) points of the scene, their labels and labelweights
        
        the scene is sampled in the vicinity of one randomly-chosen point
        the vicinity extends from the top of the point cloud to the bottom
        """
        
        CUBE_SIZE = 6 #only half the size actually. They used 0.75 in scannet
        NUM_CLASSES = 8
        point_set = self.scene_points_list[index]
        
        semantic_seg = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set,axis=0) #max sur les colonnes. vecteur ligne de taille 3
        coordmin = np.min(point_set,axis=0) #min sur les colonnes. idem.
        isvalid = False
        for i in range(NUM_CLASSES+2):
            curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],:] #selection aleatoire d'un point de la scene
            curmin = curcenter-[CUBE_SIZE, CUBE_SIZE,1.5]
            curmax = curcenter+[CUBE_SIZE,CUBE_SIZE,1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((point_set>=(curmin-0.2))*(point_set<=(curmax+0.2)),axis=1)==3
            cur_point_set = point_set[curchoice,:] #selection des points qui appartiennent a un voisinage du point central
            cur_semantic_seg = semantic_seg[curchoice] #selection de leurs labels
            if len(cur_semantic_seg)==0: 
                continue
            mask = np.sum((cur_point_set>=(curmin-0.01))*(cur_point_set<=(curmax+0.01)),axis=1)==3
            vidx = np.ceil((cur_point_set[mask,:]-curmin)/(curmax-curmin)*[31.0,31.0,62.0])
            vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])
            isvalid = np.sum(cur_semantic_seg>0)/len(cur_semantic_seg)>=0.7 and len(vidx)/31.0/31.0/62.0>=0.02
            #on requiert 70% de points annotes et autre chose que je ne comprends pas
            if isvalid:
                break
        if len(cur_point_set) - self.npoints > 0:
            trueArray = np.ones(self.npoints, dtype = bool)
            falseArray = np.zeros(len(cur_point_set) - self.npoints, dtype = bool)
            choice = np.concatenate((trueArray, falseArray), axis=0)
            np.random.shuffle(choice)

        else:
            choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
              
        point_set = cur_point_set[choice] 

        #on selectionne aleatoirement npoints (8192 par defaut) points du voisinage du pointcentral
        semantic_seg = cur_semantic_seg[choice] #on recupere leurs labels
        mask = mask[choice]
        #sample_weight = np.reshape(self.labelweights[semantic_seg], (1, -1))
        sample_weight = self.labelweights[semantic_seg]
        sample_weight *= mask # (NUM_POINTS,)
        
        # EXPERIMENTAL NORMALIZATION
        point_set = (point_set-np.mean(point_set,axis=0))/np.std(point_set,axis=0)

        return point_set, semantic_seg, sample_weight

    def __len__(self):
        return len(self.scene_points_list)

class SemanticDatasetWholeScene():
    def __init__(self, root, npoints=8192, split='train'):
        self.npoints = npoints
        self.root = root
        self.split = split

        # load the data
        self.load_data()
        
        temp1 = list()
        temp2 = list()
        for k in range(len(self.scene_points_list)):
            scene_pts_list, lbl_list = coupageDeSceneEnN(self.scene_points_list[k], np.reshape(self.semantic_labels_list[k], (-1, 1)), k)
            temp1 += scene_pts_list
            temp2 += lbl_list
        self.scene_points_list, self.semantic_labels_list = temp1, temp2
        
        if split=='train' or split=='train_short':
            labelweights = np.zeros(9)
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(10))
                labelweights += tmp
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
        for filename in self.data_filenames:
            data_points = np.load(filename + "_vertices.npz")
            data_labels = np.load(filename + "_labels.npz")
            self.scene_points_list.append(data_points[data_points.files[0]])
            self.semantic_labels_list.append(data_labels[data_labels.files[0]])      
                
    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        coordmax = np.max(point_set_ini,axis=0)
        coordmin = np.min(point_set_ini,axis=0)
        CUBE_SIZE = 6 #only half the size actually. They used 0.75 in scannet
        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/(2.0*CUBE_SIZE)).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/(2.0*CUBE_SIZE)).astype(np.int32)
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()
        #isvalid = False
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin+[i*2.0*CUBE_SIZE,j*2.0*CUBE_SIZE,0]
                curmax = coordmin+[(i+1)*(2.0*CUBE_SIZE),(j+1)*(2.0*CUBE_SIZE),coordmax[2]-coordmin[2]]
                curchoice = np.sum((point_set_ini>=(curmin-0.2))*(point_set_ini<=(curmax+0.2)),axis=1)==3
                cur_point_set = point_set_ini[curchoice,:]
                cur_semantic_seg = semantic_seg_ini[curchoice]
                if len(cur_semantic_seg)==0:
                    continue
                mask = np.sum((cur_point_set>=(curmin-0.001))*(cur_point_set<=(curmax+0.001)),axis=1)==3
                choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
                point_set = cur_point_set[choice,:] # Nx3
                semantic_seg = cur_semantic_seg[choice] # N
                mask = mask[choice]
                if sum(mask)/float(len(mask))<0.01:
                    continue
                sample_weight = self.labelweights[semantic_seg]
                sample_weight *= mask # N
                point_sets.append(np.expand_dims(point_set,0)) # 1xNx3
                semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
                sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN
        point_sets = np.concatenate(tuple(point_sets),axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs),axis=0)
        sample_weights = np.concatenate(tuple(sample_weights),axis=0)
        
        return point_sets, semantic_segs, sample_weights

    def __len__(self):
        return len(self.scene_points_list)
