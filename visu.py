import argparse
import numpy as np
import os
import sys
import importlib
import utils.pc_util as pc_util

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--set', default="train", help='train or test [default: train]')
parser.add_argument('--n', type=int, default=0, help='Number of the scene. -1 to export all scenes [default : 0]')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
parser.add_argument('--dataset', default='semantic', help='Dataset [default: semantic]')
FLAGS = parser.parse_args()

SET = FLAGS.set
N = FLAGS.n
NUM_POINT = FLAGS.num_point
DATASET_NAME = FLAGS.dataset

# Import dataset
data = importlib.import_module('dataset.' + DATASET_NAME)
NUM_CLASSES = data.NUM_CLASSES
DATA = data.Dataset(npoints=NUM_POINT, split=SET) # TRAIN_DATASET[0] ([x y z], [label (0-8)]

# Outputs

OUTPUT_DIR_GROUNDTRUTH = "visu/scenes_groundtruth_test"
if not os.path.exists(OUTPUT_DIR_GROUNDTRUTH): os.mkdir(OUTPUT_DIR_GROUNDTRUTH)

if N>=0:
    xyz, labels, _ = DATA[N]
    print "Exporting scene number " + str(N)
    pc_util.write_ply_color(xyz, labels, OUTPUT_DIR_GROUNDTRUTH+"/{}_{}.obj".format(SET, N), NUM_CLASSES)
else:
    for i in range(len(DATA)):      
        xyz, labels, _ = DATA[i]
        mean_x, mean_y, mean_z = np.mean(xyz, axis=0)
        std_x, std_y, std_z = np.std(xyz, axis=0)
        print "Exporting scene number " + str(i)
        # Print some statistics 
        print "Stats: means=%1.2f/%1.2f/%1.2f, std=%1.2f/%1.2f/%1.2f" % (mean_x, mean_y, mean_z, std_x, std_y, std_z)
        # You can open this files with meshlab for instance
        pc_util.write_ply_color(xyz, labels, OUTPUT_DIR_GROUNDTRUTH+"/{}_{}.obj".format(SET, i), NUM_CLASSES)


