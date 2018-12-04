import tensorflow as tf
import numpy as np
import time
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
from tf_sampling import prob_sample, farthest_point_sample, gather_point


class TestGrouping(tf.test.TestCase):
    def test(self):
        knn = True
        np.random.seed(100)
        pts = np.random.random((32, 512, 64)).astype("float32")
        tmp1 = np.random.random((32, 512, 3)).astype("float32")
        tmp2 = np.random.random((32, 128, 3)).astype("float32")
        with tf.device("/gpu:0"):
            points = tf.constant(pts)
            xyz1 = tf.constant(tmp1)
            xyz2 = tf.constant(tmp2)
            radius = 0.1
            nsample = 64
            if knn:
                _, idx = knn_point(nsample, xyz1, xyz2)
                grouped_points = group_point(points, idx)
            else:
                idx, _ = query_ball_point(radius, nsample, xyz1, xyz2)
                grouped_points = group_point(points, idx)
                # grouped_points_grad = tf.ones_like(grouped_points)
                # points_grad = tf.gradients(grouped_points, points, grouped_points_grad)
        with tf.Session("") as sess:
            now = time.time()
            for _ in range(100):
                ret = sess.run(grouped_points)
            print(time.time() - now)
            print(ret.shape, ret.dtype)
            print(ret)

    def test_grad(self):
        with tf.device("/gpu:0"):
            points = tf.constant(np.random.random((1, 128, 16)).astype("float32"))
            print(points)
            xyz1 = tf.constant(np.random.random((1, 128, 3)).astype("float32"))
            xyz2 = tf.constant(np.random.random((1, 8, 3)).astype("float32"))
            radius = 0.3
            nsample = 32
            idx, pts_cnt = query_ball_point(radius, nsample, xyz1, xyz2)
            grouped_points = group_point(points, idx)
            print(grouped_points)

        with self.test_session():
            print("---- Going to compute gradient error")
            err = tf.test.compute_gradient_error(
                points, (1, 128, 16), grouped_points, (1, 8, 32, 16)
            )
            print(err)
            self.assertLess(err, 1e-4)


class TestInterpolate(tf.test.TestCase):
    def test(self):
        np.random.seed(100)
        pts = np.random.random((32, 128, 64)).astype("float32")
        tmp1 = np.random.random((32, 512, 3)).astype("float32")
        tmp2 = np.random.random((32, 128, 3)).astype("float32")
        with tf.device("/cpu:0"):
            points = tf.constant(pts)
            xyz1 = tf.constant(tmp1)
            xyz2 = tf.constant(tmp2)
            dist, idx = three_nn(xyz1, xyz2)
            weight = tf.ones_like(dist) / 3.0
            interpolated_points = three_interpolate(points, idx, weight)
        with tf.Session("") as sess:
            now = time.time()
            for _ in range(100):
                ret = sess.run(interpolated_points)
            print(time.time() - now)
            print(ret.shape, ret.dtype)
            # print ret

    def test_grad(self):
        with self.test_session():
            points = tf.constant(np.random.random((1, 8, 16)).astype("float32"))
            print(points)
            xyz1 = tf.constant(np.random.random((1, 128, 3)).astype("float32"))
            xyz2 = tf.constant(np.random.random((1, 8, 3)).astype("float32"))
            dist, idx = three_nn(xyz1, xyz2)
            weight = tf.ones_like(dist) / 3.0
            interpolated_points = three_interpolate(points, idx, weight)
            print(interpolated_points)
            err = tf.test.compute_gradient_error(
                points, (1, 8, 16), interpolated_points, (1, 128, 16)
            )
            print(err)
            self.assertLess(err, 1e-4)


class TestSampling(tf.test.TestCase):
    def test(self):
        np.random.seed(100)
        triangles = np.random.rand(1, 5, 3, 3).astype("float32")
        with tf.device("/gpu:0"):
            inp = tf.constant(triangles)
            tria = inp[:, :, 0, :]
            trib = inp[:, :, 1, :]
            tric = inp[:, :, 2, :]
            areas = tf.sqrt(
                tf.reduce_sum(tf.cross(trib - tria, tric - tria) ** 2, 2) + 1e-9
            )
            randomnumbers = tf.random_uniform((1, 8192))
            triids = prob_sample(areas, randomnumbers)
            tria_sample = gather_point(tria, triids)
            trib_sample = gather_point(trib, triids)
            tric_sample = gather_point(tric, triids)
            us = tf.random_uniform((1, 8192))
            vs = tf.random_uniform((1, 8192))
            uplusv = 1 - tf.abs(us + vs - 1)
            uminusv = us - vs
            us = (uplusv + uminusv) * 0.5
            vs = (uplusv - uminusv) * 0.5
            pt_sample = (
                tria_sample
                + (trib_sample - tria_sample) * tf.expand_dims(us, -1)
                + (tric_sample - tria_sample) * tf.expand_dims(vs, -1)
            )
            print("pt_sample: ", pt_sample)
            reduced_sample = gather_point(
                pt_sample, farthest_point_sample(1024, pt_sample)
            )
            print(reduced_sample)
        with tf.Session("") as sess:
            ret = sess.run(reduced_sample)
        print(ret.shape, ret.dtype)
        # pickle.dump(ret, open("1.pkl", "wb"), -1)


if __name__ == "__main__":
    tf.test.main()
