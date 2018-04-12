"""
Predict the label
"""
import argparse
import numpy as np
import tensorflow as tf
import os
import sys
import dataset.semantic as data
import models.pointnet2_sem_seg as MODEL
import utils.pc_util as pc_util

# Parser

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--ckpt', default='', help='Checkpoint file')
parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
parser.add_argument('--set', default="train", help='train or test [default: train]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size [default: 1]') # LET DEFAULT FOR THE MOMENT!


FLAGS = parser.parse_args()

CHECKPOINT = FLAGS.ckpt
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
SET = FLAGS.set
BATCH_SIZE = FLAGS.batch_size

# Import dataset

NUM_CLASSES = 9

DATASET = data.SemanticDataset(root="", npoints=NUM_POINT, split=SET)

LABELS_TEXT = ["unlabled", "man-made terrain", "natural terrain", "high vegetation", "low vegetation", "buildings", "hard scape", "scanning artefacts", "cars"]

# Outputs

OUTPUT_DIR_GROUNDTRUTH = "visu/scenes_groundtruth_test"
OUTPUT_DIR_PREDICTION = "visu/scenes_predictions_test"

if not os.path.exists(OUTPUT_DIR_GROUNDTRUTH): os.mkdir(OUTPUT_DIR_GROUNDTRUTH)
if not os.path.exists(OUTPUT_DIR_PREDICTION): os.mkdir(OUTPUT_DIR_PREDICTION)


def predict():
    """
    Load the selected checkpoint and predict the labels
    Write in the output directories both groundtruth and prediction
    This enable to visualize side to side the prediction and the true labels,
    and helps to debug the network
    """
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl, _ = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        print tf.shape(pointclouds_pl)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, _ = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, CHECKPOINT)
    print "Model restored."

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred}

    # To add the histograms
    meta_hist_true = np.zeros(9)
    meta_hist_pred = np.zeros(9)

    for idx in range(len(DATASET)):
        xyz, true_labels, _ = DATASET[idx]
        # Ground truth
        print "Exporting scene number " + str(idx)

        pc_util.write_ply_color(xyz, true_labels, "{}/{}_{}.obj".format(OUTPUT_DIR_GROUNDTRUTH,SET, idx), NUM_CLASSES)

        # Prediction
        
        pred_labels = predict_one_input(sess, ops, idx)

        # Compute mean IoU
        iou, update_op = tf.metrics.mean_iou(tf.to_int64(true_labels), tf.to_int64(pred_labels), NUM_CLASSES)
        sess.run(tf.local_variables_initializer())
        update_op.eval(session=sess)
        print(sess.run(iou))

        hist_true, _ = np.histogram(true_labels,range(NUM_CLASSES+1))
        hist_pred, _ = np.histogram(pred_labels,range(NUM_CLASSES+1))

        # update meta histograms
        meta_hist_true += hist_true
        meta_hist_pred += hist_pred

        # print individual histograms
        print hist_true
        print hist_pred
        
        pc_util.write_ply_color(xyz, pred_labels, "{}/{}_{}.obj".format(OUTPUT_DIR_PREDICTION,SET, idx), NUM_CLASSES)
    
    meta_hist_pred = (meta_hist_pred/sum(meta_hist_pred))*100
    meta_hist_true = (meta_hist_true/sum(meta_hist_true))*100
    print(LABELS_TEXT)
    print(meta_hist_true)
    print(meta_hist_pred)


def predict_one_input(sess, ops, idx):
    is_training = False
    data, _, _ = DATASET[idx] # NUM_POINTx 3
    batch_data = np.array([data]) # 1 x NUM_POINT x 3
    feed_dict = {ops['pointclouds_pl']: batch_data,
                 ops['is_training_pl']: is_training}
    pred_val = sess.run([ops['pred']], feed_dict=feed_dict)
    pred_val = pred_val[0][0] # NUMPOINTSx9
    pred_val = np.argmax(pred_val,1)
    return pred_val


if __name__ == "__main__":
    print 'pid: %s'%(str(os.getpid()))
    with tf.Graph().as_default():
        predict()
