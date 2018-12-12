import argparse
import os
import json
import numpy as np
import tensorflow as tf
import open3d

import model
from dataset.semantic_dataset import SemanticDataset
from util.metric import ConfusionMatrix


class Predictor:
    def __init__(self, checkpoint_path, hyper_params):
        # Get ops from graph
        with tf.device("/gpu:0"):
            # Placeholder
            pl_points, _, _ = model.get_placeholders(
                1, hyper_params["num_point"], hyperparams=hyper_params
            )
            pl_is_training = tf.placeholder(tf.bool, shape=())
            print("pl_points shape", tf.shape(pl_points))

            # Prediction
            pred, _ = model.get_model(
                pl_points, pl_is_training, dataset.num_classes, hyperparams=hyper_params
            )

            # Saver
            saver = tf.train.Saver()

        self.ops = {
            "pl_points": pl_points,
            "pl_is_training": pl_is_training,
            "pred": pred,
        }

        # Restore checkpoint to session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.sess = tf.Session(config=config)
        saver.restore(self.sess, checkpoint_path)
        print("Model restored")

    def predict(self, points):
        """
        Args:
            points: batch_size * num_point * 3

        Returns:
            pred_labels: batch_size * num_point * 1
        """
        is_training = False
        batch_data = np.array([points])  # batch_size * num_point * 3
        feed_dict = {
            self.ops["pl_points"]: batch_data,
            self.ops["pl_is_training"]: is_training,
        }
        pred_val = self.sess.run([self.ops["pred"]], feed_dict=feed_dict)
        pred_val = pred_val[0]  # batch_size * num_point * 1
        pred_labels = np.argmax(pred_val, 2)  # batch_size * num_point * 1
        return pred_labels


if __name__ == "__main__":
    np.random.seed(0)

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="# samples, each contains num_point points",
    )
    parser.add_argument("--ckpt", default="", help="Checkpoint file")
    parser.add_argument("--set", default="validation", help="train, validation, test")
    flags = parser.parse_args()
    hyper_params = json.loads(open("semantic.json").read())

    # Create output dir
    output_dir = os.path.join("result", "sparse")
    os.makedirs(output_dir, exist_ok=True)

    # Dataset
    dataset = SemanticDataset(
        num_points_per_sample=hyper_params["num_point"],
        split=flags.set,
        box_size=hyper_params["box_size"],
        use_color=hyper_params["use_color"],
        path=hyper_params["data_path"],
    )

    # Model
    predictor = Predictor(checkpoint_path=flags.ckpt, hyper_params=hyper_params)

    num_scenes = dataset.num_scenes
    p = 6 if hyper_params["use_color"] else 3
    scene_points = [np.array([]).reshape((0, p)) for i in range(num_scenes)]
    ground_truth = [np.array([]) for i in range(num_scenes)]
    predicted_labels = [np.array([]) for i in range(num_scenes)]

    for batch_index in range(flags.num_samples * num_scenes):
        scene_index, data, raw_data, true_labels, colors = dataset.next_sample(
            is_training=False
        )
        if p == 6:
            raw_data = np.hstack((raw_data, colors))
            data = np.hstack((data, colors))
        pred_labels = predictor.predict(data)
        pred_labels = np.squeeze(pred_labels)
        scene_points[scene_index] = np.vstack((scene_points[scene_index], raw_data))
        ground_truth[scene_index] = np.hstack((ground_truth[scene_index], true_labels))
        predicted_labels[scene_index] = np.hstack(
            (predicted_labels[scene_index], pred_labels)
        )

        if batch_index % 100 == 0:
            print(
                "Batch {} predicted, num points in scenes {}".format(
                    batch_index, [len(labels) for labels in predicted_labels]
                )
            )

    file_paths_without_ext = dataset.get_file_paths_without_ext()
    print("{} point clouds to export".format(len(file_paths_without_ext)))
    cm = ConfusionMatrix(9)

    for scene_index in range(num_scenes):
        file_prefix = os.path.basename(file_paths_without_ext[scene_index])

        # Save sparse point cloud
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(scene_points[scene_index][:, 0:3])
        open3d.write_point_cloud(os.path.join(output_dir, file_prefix + ".pcd"), pcd)

        # Save predicted labels of the sparse point cloud
        pd_labels = predicted_labels[scene_index].astype(int)
        gt_labels = ground_truth[scene_index].astype(int)
        pd_labels_path = os.path.join(output_dir, file_prefix + ".labels")
        np.savetxt(pd_labels_path, pd_labels, fmt="%d")

        # Increment confusion matrix
        cm.increment_from_list(gt_labels, pd_labels)

        # Print
        print("Exported: {} with {} points".format(file_prefix, len(pd_labels)))

    cm.print_metrics()

    # Process each file
    semantic_file_data = dataset.list_file_data[0]
    points, points_raw, gt_labels, colors = semantic_file_data.next_sample(
        hyper_params["num_point"]
    )
    points_with_colors = np.hstack((points, colors))
    pd_labels = predictor.predict(points_with_colors)[0]
    cm = ConfusionMatrix(9)
    import ipdb; ipdb.set_trace()
    cm.increment_from_list(gt_labels, pd_labels)
    cm.print_metrics()
