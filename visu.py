import argparse
import numpy as np
import os
import sys

import utils.pc_util as pc_util
import dataset.semantic as data

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--set', default="train", help='train or test [default: train]')
parser.add_argument('--n', type=int, default=0, help='Number of the scene. -1 to export all scenes [default : 0]')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
FLAGS = parser.parse_args()

SET = FLAGS.set
N = FLAGS.n
NUM_POINT = FLAGS.num_point
NUM_CLASSES = 9

# load data
DATA = data.SemanticDataset(root="", npoints=NUM_POINT, split=SET) # TRAIN_DATASET[0] ([x y z], [label (0-8)]

if N>=0:
    xyz, labels, _ = DATA[N]
    print "Exporting scene number " + str(N)
    pc_util.write_ply_color(xyz, labels, "visu/scenes_groundtruth/{}_{}.obj".format(SET, N), NUM_CLASSES)
else:
    for i in range(len(DATA)):
        print "Exporting scene number " + str(i)
        xyz, labels, _ = DATA[i]
        pc_util.write_ply_color(xyz, labels, "visu/scenes_groundtruth/{}_{}.obj".format(SET, i), NUM_CLASSES)


