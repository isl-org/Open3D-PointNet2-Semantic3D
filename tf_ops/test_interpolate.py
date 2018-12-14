import numpy as np
import tensorflow as tf
from tf_interpolate import three_nn, three_interpolate

if __name__ == "__main__":
    np.random.seed(100)
    pts = np.random.random((32, 128, 64)).astype("float32")
    tmp1 = np.random.random((32, 512, 3)).astype("float32")
    tmp2 = np.random.random((32, 128, 3)).astype("float32")

    with tf.device("/cpu:0"):
        points = tf.constant(pts)
        xyz1 = tf.constant(tmp1)
        xyz2 = tf.constant(tmp2)
        dist, idx = three_nn(xyz1, xyz2)

    with tf.Session("") as sess:
        dist, idx = sess.run(three_nn(xyz1, xyz2))
        print(dist.shape, dist.dtype)
        print(dist[:3, :3, :1])
        print(idx.shape, idx.dtype)
        print(idx[:3, :3, :1])

    # (32, 512, 3) float32
    # [[[0.03415794]
    # [0.01955329]
    # [0.00292144]]
    #
    # [[0.0038327 ]
    # [0.01290642]
    # [0.01430019]]
    #
    # [[0.04590415]
    # [0.01247532]
    # [0.00384166]]]
    # (32, 512, 3) int32
    # [[[ 62]
    # [ 83]
    # [ 53]]
    #
    # [[ 71]
    # [ 82]
    # [112]]
    #
    # [[ 77]
    # [126]
    # [ 42]]]
