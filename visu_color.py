import argparse
import numpy as np
import os
import sys
import importlib
import utils.pc_util as pc_util

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--set', default="train", help='train or test [default: train]')
parser.add_argument('--n', type=int, default=1, help='Number of batches you want [default : 1]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size [default: 32]')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
parser.add_argument('--dataset', default='semantic_color', help='Dataset [default: semantic_color]')
FLAGS = parser.parse_args()

SET = FLAGS.set
N = FLAGS.n
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DATASET_NAME = FLAGS.dataset

DROPOUT = True
DROPOUT_RATIO = 0
AUGMENT = False
STATS = True

# Import dataset
data = importlib.import_module('dataset.' + DATASET_NAME)
NUM_CLASSES = data.NUM_CLASSES
DATA = data.Dataset(npoints=NUM_POINT, split=SET) # TRAIN_DATASET[0] ([x y z], [label (0-8)]

# Outputs

OUTPUT_DIR = "visu/color"
if not os.path.exists(OUTPUT_DIR): os.mkdir(OUTPUT_DIR)

if DROPOUT:
    print "Warning : dropout is on, with ratio %f" %(DROPOUT_RATIO)
if AUGMENT:
    print "Warning : rotation is on"

for i in range(N):
    data, label_data, _ = DATA.next_batch(BATCH_SIZE, AUGMENT, DROPOUT)

    if (N > 1):
        print "Exporting batch number " + str(N)
    for j, scene in enumerate(data):
        labels = label_data[j]
        if DROPOUT:
            # Replicate the input dropout to see the effect of the decimation
            drop_idx = np.where(np.random.random((scene.shape[0]))<=DROPOUT_RATIO)[0]
            scene = np.delete(scene,drop_idx, axis=0)
            labels = np.delete(labels,drop_idx)
        #np.savetxt(OUTPUT_DIR+"/{}_{}_{}.obj".format(SET, "trueColors", j), scene, delimiter=" ")
        pc_util.write_ply_true_color(scene[:,0:3], scene[:,3:6], OUTPUT_DIR+"/{}_{}_{}.obj".format(SET, "trueColors", j))
        pc_util.write_ply_color(scene[:,0:3], labels, OUTPUT_DIR+"/{}_{}_{}.obj".format(SET, "labelColors", j), NUM_CLASSES)
