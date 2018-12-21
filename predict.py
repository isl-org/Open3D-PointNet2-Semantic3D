import argparse
import os
import json
import numpy as np
import tensorflow as tf
import open3d
import time

import model
from dataset.semantic_dataset import SemanticDataset
from util.metric import ConfusionMatrix
from tf_ops.tf_interpolate import three_nn, interpolate_label


def interpolate_dense_labels_simple(sparse_points, sparse_labels, dense_points, k=3):
    sparse_pcd = open3d.PointCloud()
    sparse_pcd.points = open3d.Vector3dVector(sparse_points)
    sparse_pcd_tree = open3d.KDTreeFlann(sparse_pcd)

    dense_labels = []
    counter = 0
    for dense_point in dense_points:
        result_k, sparse_indexes, _ = sparse_pcd_tree.search_knn_vector_3d(
            dense_point, k
        )
        knn_sparse_labels = sparse_labels[sparse_indexes]
        if len(np.unique(knn_sparse_labels)) == len(knn_sparse_labels):
            # Fix skew, since bincount+argmax output smallest value if all unique
            dense_label = knn_sparse_labels[0]
        else:
            dense_label = np.bincount(knn_sparse_labels).argmax()
        dense_labels.append(dense_label)
        counter += 1
    return dense_labels


class PredictInterpolator:
    def __init__(self, checkpoint_path, num_classes, hyper_params):
        # Get ops from graph
        with tf.device("/gpu:0"):
            # Placeholders
            pl_sparse_points_centered_batched, _, _ = model.get_placeholders(
                hyper_params["num_point"], hyperparams=hyper_params
            )
            pl_is_training = tf.placeholder(tf.bool, shape=())

            # Prediction
            pred, _ = model.get_model(
                pl_sparse_points_centered_batched,
                pl_is_training,
                num_classes,
                hyperparams=hyper_params,
            )
            sparse_labels_batched = tf.argmax(pred, axis=2)
            # (1, num_sparse_points) -> (num_sparse_points,)
            sparse_labels = tf.reshape(sparse_labels_batched, [-1])
            sparse_labels = tf.cast(sparse_labels, tf.int32)

            # Saver
            saver = tf.train.Saver()

            # Graph for interpolating labels
            # Assuming batch_size == 1 for simplicity
            pl_sparse_points_batched = tf.placeholder(tf.float32, (None, None, 3))
            sparse_points = tf.reshape(pl_sparse_points_batched, [-1, 3])
            pl_dense_points = tf.placeholder(tf.float32, (None, 3))
            pl_knn = tf.placeholder(tf.int32, ())
            dense_labels = interpolate_label(
                sparse_points, sparse_labels, pl_dense_points, pl_knn
            )

        self.ops = {
            "pl_sparse_points_centered_batched": pl_sparse_points_centered_batched,
            "pl_sparse_points_batched": pl_sparse_points_batched,
            "pl_dense_points": pl_dense_points,
            "pl_is_training": pl_is_training,
            "pl_knn": pl_knn,
            "dense_labels": dense_labels,
        }

        # Restore checkpoint to session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.sess = tf.Session(config=config)
        saver.restore(self.sess, checkpoint_path)
        print("Model restored")

    def predict_and_interpolate(
        self,
        sparse_points_centered_batched,
        sparse_points_batched,
        dense_points,
        run_metadata=None,
        run_options=None,
    ):
        s = time.time()
        if run_metadata is None:
            run_metadata = tf.RunMetadata()
        if run_options is None:
            run_options = tf.RunOptions()
        dense_labels_val = self.sess.run(
            self.ops["dense_labels"],
            feed_dict={
                self.ops[
                    "pl_sparse_points_centered_batched"
                ]: sparse_points_centered_batched,
                self.ops["pl_sparse_points_batched"]: sparse_points_batched,
                self.ops["pl_dense_points"]: dense_points,
                self.ops["pl_knn"]: 3,
                self.ops["pl_is_training"]: False,
            },
            options=run_options,
            run_metadata=run_metadata,
        )
        print("sess.run interpolate_labels time", time.time() - s)
        return dense_labels_val


class Predictor:
    def __init__(self, checkpoint_path, num_classes, hyper_params):
        # Get ops from graph
        with tf.device("/gpu:0"):
            # Placeholder
            pl_points, _, _ = model.get_placeholders(
                hyper_params["num_point"], hyperparams=hyper_params
            )
            pl_is_training = tf.placeholder(tf.bool, shape=())
            print("pl_points shape", tf.shape(pl_points))

            # Prediction
            pred, _ = model.get_model(
                pl_points, pl_is_training, num_classes, hyperparams=hyper_params
            )

            # Saver
            saver = tf.train.Saver()

            # Graph for interpolating labels
            # Assuming batch_size == 1 for simplicity
            pl_sparse_points = tf.placeholder(tf.float32, (None, 3))
            pl_sparse_labels = tf.placeholder(tf.int32, (None,))
            pl_dense_points = tf.placeholder(tf.float32, (None, 3))
            pl_knn = tf.placeholder(tf.int32, ())
            sparse_indices = interpolate_label(
                pl_sparse_points, pl_sparse_labels, pl_dense_points, pl_knn
            )

        self.ops = {
            "pl_points": pl_points,
            "pl_is_training": pl_is_training,
            "pred": pred,
            "pl_sparse_points": pl_sparse_points,
            "pl_sparse_labels": pl_sparse_labels,
            "pl_dense_points": pl_dense_points,
            "pl_knn": pl_knn,
            "sparse_indices": sparse_indices,
        }

        # Restore checkpoint to session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.sess = tf.Session(config=config)
        saver.restore(self.sess, checkpoint_path)
        print("Model restored")

    def predict(self, batch_data, run_metadata=None, run_options=None):
        """
        Args:
            batch_data: batch_size * num_point * 6(3)

        Returns:
            pred_labels: batch_size * num_point * 1
        """
        is_training = False
        feed_dict = {
            self.ops["pl_points"]: batch_data,
            self.ops["pl_is_training"]: is_training,
        }
        if run_metadata is None:
            run_metadata = tf.RunMetadata()
        if run_options is None:
            run_options = tf.RunOptions()

        pred_val = self.sess.run(
            [self.ops["pred"]],
            options=run_options,
            run_metadata=run_metadata,
            feed_dict=feed_dict,
        )
        pred_val = pred_val[0]  # batch_size * num_point * 1
        pred_labels = np.argmax(pred_val, 2)  # batch_size * num_point * 1
        return pred_labels

    def interpolate_labels(self, sparse_points, sparse_labels, dense_points):
        s = time.time()
        dense_labels = self.sess.run(
            self.ops["sparse_indices"],
            feed_dict={
                self.ops["pl_sparse_points"]: sparse_points,
                self.ops["pl_sparse_labels"]: sparse_labels,
                self.ops["pl_dense_points"]: dense_points,
                self.ops["pl_knn"]: 3,
            },
        )
        print("sess.run interpolate_labels time", time.time() - s)
        return dense_labels


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
        box_size_x=hyper_params["box_size_x"],
        box_size_y=hyper_params["box_size_y"],
        use_color=hyper_params["use_color"],
        path=hyper_params["data_path"],
    )

    # Model
    batch_size = 64
    predictor = Predictor(
        checkpoint_path=flags.ckpt,
        num_classes=dataset.num_classes,
        hyper_params=hyper_params,
    )

    # Process each file
    cm = ConfusionMatrix(9)

    for semantic_file_data in dataset.list_file_data[:1]:
        print("Processing {}".format(semantic_file_data))

        # Predict for num_samples times
        points_raw_collector = []
        pd_labels_collector = []

        # If flags.num_samples < batch_size, will predict one batch
        for batch_index in range(int(np.ceil(flags.num_samples / batch_size))):
            current_batch_size = min(
                batch_size, flags.num_samples - batch_index * batch_size
            )

            # Get data
            points, points_raw, gt_labels, colors = semantic_file_data.sample_batch(
                batch_size=current_batch_size,
                num_points_per_sample=hyper_params["num_point"],
            )

            # (bs, 8192, 3) concat (bs, 8192, 3) -> (bs, 8192, 6)
            if hyper_params["use_color"]:
                points_with_colors = np.concatenate((points, colors), axis=-1)
            else:
                points_with_colors = points

            # Predict
            s = time.time()
            pd_labels = predictor.predict(points_with_colors)
            print(
                "Batch size: {}, time: {}".format(current_batch_size, time.time() - s)
            )

            # Save to collector for file output
            points_raw_collector.extend(points_raw)
            pd_labels_collector.extend(pd_labels)

            # Increment confusion matrix
            cm.increment_from_list(gt_labels.flatten(), pd_labels.flatten())

        # Save sparse point cloud and predicted labels
        file_prefix = os.path.basename(semantic_file_data.file_path_without_ext)

        points_raw_collector = np.array(points_raw_collector)
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(points_raw_collector.reshape((-1, 3)))
        pcd_path = os.path.join(output_dir, file_prefix + ".pcd")
        open3d.write_point_cloud(pcd_path, pcd)
        print("Exported pcd to {}".format(pcd_path))

        pd_labels_collector = np.array(pd_labels_collector).astype(int)
        pd_labels_path = os.path.join(output_dir, file_prefix + ".labels")
        np.savetxt(pd_labels_path, pd_labels_collector.flatten(), fmt="%d")
        print("Exported labels to {}".format(pd_labels_path))

    cm.print_metrics()
