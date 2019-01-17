import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
interpolate_module = tf.load_op_library(
    os.path.join(BASE_DIR, "build", "libtf_interpolate.so")
)


def three_nn(xyz1, xyz2):
    """
    Input:
        xyz1: (b,n,3) float32 array, unknown points
        xyz2: (b,m,3) float32 array, known points
    Output:
        dist: (b,n,3) float32 array, distances to known points
        idx: (b,n,3) int32 array, indices to known points
    """
    return interpolate_module.three_nn(xyz1, xyz2)


ops.NoGradient("ThreeNN")


def interpolate_label_with_color(sparse_points, sparse_labels, dense_points, knn):
    """
    Input:
        sparse_points: (num_sparse_points, 3) float32 array, points
                      with known labels
        sparse_labels: (num_sparse_points, 3) float32 array, labels of
                      sparse_points
        dense_points: (num_dense_points, 3) float32 array, points
                      with unknown labels
        knn: int, use k-NN for label interpolation
    Output:
        dense_labels:  (num_dense_points,) int32 array, indices
        dense_colors:  (num_dense_points, 3) uint8 array, colors for dense_labels
    """
    return interpolate_module.interpolate_label_with_color(
        sparse_points, sparse_labels, dense_points, knn
    )


ops.NoGradient("InterpolateLabelWithColor")


def three_interpolate(points, idx, weight):
    """
    Input:
        points: (b,m,c) float32 array, known points
        idx: (b,n,3) int32 array, indices to known points
        weight: (b,n,3) float32 array, weights on known points
    Output:
        out: (b,n,c) float32 array, interpolated point values
    """
    return interpolate_module.three_interpolate(points, idx, weight)


@tf.RegisterGradient("ThreeInterpolate")
def _three_interpolate_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    weight = op.inputs[2]
    return [
        interpolate_module.three_interpolate_grad(points, idx, weight, grad_out),
        None,
        None,
    ]
