import numpy as np
import tensorflow as tf
import time
from tf_interpolate import three_nn, three_interpolate

if __name__ == "__main__":
    np.random.seed(100)

    target_points = np.random.random((64, 8192, 3)).astype("float32")
    reference_points = np.random.random((64, 1024, 3)).astype("float32")

    with tf.device("/cpu:0"):
        xyz1 = tf.constant(target_points)
        xyz2 = tf.constant(reference_points)
        dist, idx = three_nn(xyz1, xyz2)

    with tf.Session("") as sess:
        # Warm up
        dist, idx = sess.run(three_nn(xyz1, xyz2))

        # Run
        s = time.time()
        dist, idx = sess.run(three_nn(xyz1, xyz2))
        print("Time: {}".format(time.time() - s))
        print(idx.shape, idx.dtype)
        print(dist.shape, dist.dtype)
        print(dist[:3, :3, :1].flatten())
        print(idx[:3, :3, :1].flatten())

        # Expected output
        # (64, 8192, 3) int32
        # (64, 8192, 3) float32
        # [0.00175864 0.00671887 0.0034472  0.00337327 0.00191902 0.00075543
        #  0.00169418 0.00473733 0.00381071]
        # [137 856 116  76 915 199 117 659 786]
