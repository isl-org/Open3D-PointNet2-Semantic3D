"""
This file predicts the labels with the weights calculated by train.py or train_multi.py.
You need to give as parameter the ckpt you want to use. A certain number of inputs is generated.
A prediction is made on each of these inputs, to be compared to the ground truth, also exported.
If you set --cloud=False (which is the default usage), each input is saved in a different point
cloud, and there will be n inputs. If you set --cloud=True, they will be aggregated into scenes
and there will be n inputs per scene. These aggregated point clouds is the basis upon which
interpolation is made to give predictions on the full raw point clouds and to truly evaluate
the network's performances.
"""
import argparse
import numpy as np
import tensorflow as tf
import importlib
import os
import json
import models.model as MODEL
import utils.pc_util as pc_util

# Parser

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, help="GPU to use [default: GPU 0]")
parser.add_argument(
    "--cloud",
    type=bool,
    default=False,
    help="whether you want full point clouds to be exported",
)
parser.add_argument(
    "--n", type=int, default=8, help="Number of inputs you want [default : 8]"
)
parser.add_argument("--ckpt", default="", help="Checkpoint file")
parser.add_argument(
    "--num_point", type=int, default=8192, help="Point Number [default: 8192]"
)
parser.add_argument("--set", default="test", help="train or test [default: test]")
parser.add_argument("--dataset", default="semantic", help="Dataset [default: semantic]")
parser.add_argument(
    "--config", type=str, default="config.json", metavar="N", help="config file"
)
FLAGS = parser.parse_args()

JSON_DATA_CUSTOM = open(FLAGS.config).read()
CUSTOM = json.loads(JSON_DATA_CUSTOM)
JSON_DATA = open("default.json").read()
PARAMS = json.loads(JSON_DATA)
PARAMS.update(CUSTOM)

CHECKPOINT = FLAGS.ckpt
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
SET = FLAGS.set
DATASET_NAME = FLAGS.dataset
N = FLAGS.n
print("N", N)

# Fix GPU use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)

DROPOUT = False
DROPOUT_RATIO = 0.875
AUGMENT = False
MAX_EXPORT = 20  # the maximum number of scenes to be exported

if DROPOUT:
    print("dropout is on, with ratio %f" % (DROPOUT_RATIO))
if AUGMENT:
    print("rotation is on")

# Import dataset
data = importlib.import_module("dataset." + DATASET_NAME)
DATASET = data.Dataset(
    npoints=NUM_POINT,
    split=SET,
    box_size=PARAMS["box_size"],
    use_color=PARAMS["use_color"],
    dropout_max=PARAMS["input_dropout"],
    path=PARAMS["data_path"],
    z_feature=PARAMS["use_z_feature"],
)
NUM_CLASSES = DATASET.num_classes

LABELS_TEXT = DATASET.labels_names

# Outputs
OUTPUT_DIR = os.path.join("visu", DATASET_NAME + "_" + SET)
if not os.path.exists("visu"):
    os.mkdir("visu")
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


def predict_one_input(sess, ops, data):
    is_training = False
    batch_data = np.array([data])  # 1 x NUM_POINT x 3
    feed_dict = {ops["pointclouds_pl"]: batch_data, ops["is_training_pl"]: is_training}
    pred_val = sess.run([ops["pred"]], feed_dict=feed_dict)
    pred_val = pred_val[0][0]  # NUMPOINTSx9
    pred_val = np.argmax(pred_val, 1)
    return pred_val

def predict():
    """
    Load the selected checkpoint and predict the labels
    Write in the output directories both groundtruth and prediction
    This enable to visualize side to side the prediction and the true labels,
    and helps to debug the network
    """
    with tf.device("/gpu:" + str(GPU_INDEX)):
        pointclouds_pl, labels_pl, _ = MODEL.placeholder_inputs(
            1, NUM_POINT, hyperparams=PARAMS
        )
        print(tf.shape(pointclouds_pl))
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, _ = MODEL.get_model(
            pointclouds_pl, is_training_pl, NUM_CLASSES, hyperparams=PARAMS
        )

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
    print("Model restored.")

    ops = {
        "pointclouds_pl": pointclouds_pl,
        "labels_pl": labels_pl,
        "is_training_pl": is_training_pl,
        "pred": pred,
    }

    OUTPUT_DIR_FULL_PC = os.path.join(OUTPUT_DIR, "full_scenes_predictions")
    if not os.path.exists(OUTPUT_DIR_FULL_PC):
        os.mkdir(OUTPUT_DIR_FULL_PC)
    nscenes = len(DATASET)
    p = 6 if PARAMS["use_color"] else 3
    scene_points = [np.array([]).reshape((0, p)) for i in range(nscenes)]
    ground_truth = [np.array([]) for i in range(nscenes)]
    predicted_labels = [np.array([]) for i in range(nscenes)]

    for i in range(N * nscenes):
        if i % 100 == 0 and i > 0:
            print("{} inputs generated".format(i))
        f, data, raw_data, true_labels, col, _ = DATASET.next_input(
            DROPOUT, True, False, predicting=True
        )
        if p == 6:
            raw_data = np.hstack((raw_data, col))
            data = np.hstack((data, col))
        pred_labels = predict_one_input(sess, ops, data)
        scene_points[f] = np.vstack((scene_points[f], raw_data))
        ground_truth[f] = np.hstack((ground_truth[f], true_labels))
        predicted_labels[f] = np.hstack((predicted_labels[f], pred_labels))

    file_names = DATASET.get_data_filenames()
    print("{} point clouds to export".format(len(file_names)))

    for f, filename in enumerate(file_names):
        print(
            "exporting file {} which has {} points".format(
                os.path.basename(filename), len(ground_truth[f])
            )
        )
        pc_util.write_ply_color(
            scene_points[f][:, 0:3],
            ground_truth[f],
            OUTPUT_DIR_FULL_PC
            + "/{}_groundtruth.txt".format(os.path.basename(filename)),
            NUM_CLASSES,
        )
        pc_util.write_ply_color(
            scene_points[f][:, 0:3],
            predicted_labels[f],
            OUTPUT_DIR_FULL_PC
            + "/{}_aggregated.txt".format(os.path.basename(filename)),
            NUM_CLASSES,
        )
        np.savetxt(
            OUTPUT_DIR_FULL_PC + "/{}_pred.txt".format(os.path.basename(filename)),
            predicted_labels[f].reshape((-1, 1)),
            delimiter=" ",
        )
    print("done.")



if __name__ == "__main__":
    print("pid: %s" % (str(os.getpid())))
    with tf.Graph().as_default():
        predict()
