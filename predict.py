"""
This file predicts the labels with the weights calculated by train.py.
These aggregated point clouds is the basis upon which interpolation is made to give
predictions on the full raw point clouds and to truly evaluate the network's
performances.
"""

import argparse
import os
import json
import numpy as np
import tensorflow as tf
import models.model as MODEL
import utils.pc_util as pc_util
from dataset.semantic import SemanticDataset
from utils.metric import ConfusionMatrix


# Parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--n", type=int, default=8, help="# samples, each contains num_point points"
)
parser.add_argument("--ckpt", default="", help="Checkpoint file")
parser.add_argument(
    "--num_point", type=int, default=8192, help="Point Number [default: 8192]"
)
parser.add_argument("--set", default="test", help="train or test [default: test]")

# Two global arg collections
FLAGS = parser.parse_args()
PARAMS = json.loads(open("semantic.json").read())


def predict_one_input(sess, ops, data):
    is_training = False
    batch_data = np.array([data])  # 1 x FLAGS.num_point x 3
    feed_dict = {ops["pointclouds_pl"]: batch_data, ops["is_training_pl"]: is_training}
    pd_val = sess.run([ops["pred"]], feed_dict=feed_dict)
    pd_val = pd_val[0][0]  # FLAGS.num_point x 9
    pd_label = np.argmax(pd_val, 1)
    return pd_label


if __name__ == "__main__":
    # Create output dir
    output_dir = os.path.join("results", "sparse")
    os.makedirs(output_dir, exist_ok=True)

    # Import dataset
    dataset = SemanticDataset(
        npoints=FLAGS.num_point,
        split=FLAGS.set,
        box_size=PARAMS["box_size"],
        use_color=PARAMS["use_color"],
        path=PARAMS["data_path"],
    )

    with tf.device("/gpu:0"):
        pointclouds_pl, labels_pl, _ = MODEL.placeholder_inputs(
            1, FLAGS.num_point, hyperparams=PARAMS
        )
        print(tf.shape(pointclouds_pl))
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # Simple model
        pred, _ = MODEL.get_model(
            pointclouds_pl, is_training_pl, dataset.num_classes, hyperparams=PARAMS
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
    saver.restore(sess, FLAGS.ckpt)
    print("Model restored.")

    ops = {
        "pointclouds_pl": pointclouds_pl,
        "labels_pl": labels_pl,
        "is_training_pl": is_training_pl,
        "pred": pred,
    }

    nscenes = len(dataset)
    p = 6 if PARAMS["use_color"] else 3
    scene_points = [np.array([]).reshape((0, p)) for i in range(nscenes)]
    ground_truth = [np.array([]) for i in range(nscenes)]
    predicted_labels = [np.array([]) for i in range(nscenes)]

    for i in range(FLAGS.n * nscenes):
        if i % 100 == 0 and i > 0:
            print("{} inputs generated".format(i))
        scene_index, data, raw_data, true_labels, col, _ = dataset.next_input(
            sample=True, verbose=False, predicting=True
        )
        if p == 6:
            raw_data = np.hstack((raw_data, col))
            data = np.hstack((data, col))
        pred_labels = predict_one_input(sess, ops, data)
        scene_points[scene_index] = np.vstack((scene_points[scene_index], raw_data))
        ground_truth[scene_index] = np.hstack((ground_truth[scene_index], true_labels))
        predicted_labels[scene_index] = np.hstack(
            (predicted_labels[scene_index], pred_labels)
        )

    file_names = dataset.get_data_filenames()
    print("{} point clouds to export".format(len(file_names)))
    cm = ConfusionMatrix(9)

    for scene_index, file_name in enumerate(file_names):
        file_prefix = os.path.basename(file_name)
        print(
            "exporting file {} which has {} points".format(
                file_prefix, len(ground_truth[scene_index])
            )
        )
        pc_util.write_ply_color(
            scene_points[scene_index][:, 0:3],
            ground_truth[scene_index],
            os.path.join(output_dir, file_prefix + "_groundtruth.txt"),
        )
        pc_util.write_ply_color(
            scene_points[scene_index][:, 0:3],
            predicted_labels[scene_index],
            os.path.join(output_dir, file_prefix + "_aggregated.txt"),
        )

        pd_labels_path = os.path.join(output_dir, file_prefix + "_pred.txt")
        np.savetxt(
            pd_labels_path,
            predicted_labels[scene_index].reshape((-1, 1)),
            delimiter=" ",
        )
        gt_labels_path = os.path.join(output_dir, file_prefix + "_gt.txt")
        np.savetxt(
            gt_labels_path, ground_truth[scene_index].reshape((-1, 1)), delimiter=" "
        )
        cm.increment_from_file(gt_labels_path, pd_labels_path)

    cm.print_metrics()
