#include <cstdio>
#include <ctime>
#include <cstring>      // memset
#include <cstdlib>      // rand, RAND_MAX
#include <cmath>        // sqrtf
#include <Core/Core.h>  // open3d
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
using namespace tensorflow;

REGISTER_OP("ThreeNN")
    .Input("xyz1: float32")
    .Input("xyz2: float32")
    .Output("dist: float32")
    .Output("idx: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(0));
        return Status::OK();
    });
REGISTER_OP("ThreeInterpolate")
    .Input("points: float32")
    .Input("idx: int32")
    .Input("weight: float32")
    .Output("out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        ::tensorflow::shape_inference::ShapeHandle dims1;  // (b,m,c)
        c->WithRank(c->input(0), 3, &dims1);
        ::tensorflow::shape_inference::ShapeHandle dims2;  // (b,n,3)
        c->WithRank(c->input(1), 3, &dims2);
        // (b,n,c)
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape(
            {c->Dim(dims1, 0), c->Dim(dims2, 1), c->Dim(dims1, 2)});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("ThreeInterpolateGrad")
    .Input("points: float32")
    .Input("idx: int32")
    .Input("weight: float32")
    .Input("grad_out: float32")
    .Output("grad_points: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

float randomf() { return (rand() + 0.5) / (RAND_MAX + 1.0); }
static double get_time() {
    timespec tp;
    clock_gettime(CLOCK_MONOTONIC, &tp);
    return tp.tv_sec + tp.tv_nsec * 1e-9;
}

std::vector<Eigen::Vector3d> buffer_to_eigen_vector(const float *buffer,
                                                    size_t num_elements) {
    std::vector<Eigen::Vector3d> eigen_vectors;
    size_t vector_size = num_elements / 3;
    for (size_t i = 0; i < vector_size; ++i) {
        double x = buffer[i * 3 + 0];
        double y = buffer[i * 3 + 1];
        double z = buffer[i * 3 + 2];
        Eigen::Vector3d v;
        v << x, y, z;
        eigen_vectors.push_back(v);
    }
    return eigen_vectors;
}

// Find three nearest neighbors with square distance
// input: xyz1 (b,n,3), xyz2(b,m,3)
// output: dist (b,n,3), idx (b,n,3)
// E.g.
// - target_points (b, n, 3): e.g. (64, 8192, 3), the "3" here is x, y, z
// - reference_points (b, m, 3): e.g. (64, 1024, 3), the "3" here is x, y, z
// - dist (b, n, 3): (64, 8192, 3), for each input point in target_points, find
//                   3 nearest neighbors in base_points and return the
//                   distances squared, the "3" means "3" nearest neighbors
// - idx (b, n, 3): (64, 8192, 3), for each input point in target_points, find 3
//                   nearest neighbors in base_points and return the indexes in
//                   base_points, the "3" means "3" nearest neighbors
void threenn_cpu(int b, int n, int m, const float *xyz1, const float *xyz2,
                 float *dist, int *idx) {
    for (int i = 0; i < b; ++i) {
        open3d::PointCloud target_pcd;
        target_pcd.points_ = buffer_to_eigen_vector(xyz1, n * 3);

        open3d::PointCloud reference_pcd;
        reference_pcd.points_ = buffer_to_eigen_vector(xyz2, m * 3);

        open3d::KDTreeFlann reference_kd_tree(reference_pcd);
        for (size_t j = 0; j < n; ++j) {
            std::vector<int> indices;
            std::vector<double> distance2;
            reference_kd_tree.SearchKNN(target_pcd.points_[j], 3, indices,
                                        distance2);
            idx[j * 3 + 0] = indices[0];
            idx[j * 3 + 1] = indices[1];
            idx[j * 3 + 2] = indices[2];
            dist[j * 3 + 0] = distance2[0];
            dist[j * 3 + 1] = distance2[1];
            dist[j * 3 + 2] = distance2[2];
        }

        xyz1 += n * 3;
        xyz2 += m * 3;
        dist += n * 3;
        idx += n * 3;
    }
}

// input: points (b,m,c), idx (b,n,3), weight (b,n,3)
// output: out (b,n,c)
void threeinterpolate_cpu(int b, int m, int c, int n, const float *points,
                          const int *idx, const float *weight, float *out) {
    float w1, w2, w3;
    int i1, i2, i3;
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < n; ++j) {
            w1 = weight[j * 3];
            w2 = weight[j * 3 + 1];
            w3 = weight[j * 3 + 2];
            i1 = idx[j * 3];
            i2 = idx[j * 3 + 1];
            i3 = idx[j * 3 + 2];
            for (int l = 0; l < c; ++l) {
                out[j * c + l] = points[i1 * c + l] * w1 +
                                 points[i2 * c + l] * w2 +
                                 points[i3 * c + l] * w3;
            }
        }
        points += m * c;
        idx += n * 3;
        weight += n * 3;
        out += n * c;
    }
}

// input: grad_out (b,n,c), idx (b,n,3), weight (b,n,3)
// output: grad_points (b,m,c)
void threeinterpolate_grad_cpu(int b, int n, int c, int m,
                               const float *grad_out, const int *idx,
                               const float *weight, float *grad_points) {
    float w1, w2, w3;
    int i1, i2, i3;
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < n; ++j) {
            w1 = weight[j * 3];
            w2 = weight[j * 3 + 1];
            w3 = weight[j * 3 + 2];
            i1 = idx[j * 3];
            i2 = idx[j * 3 + 1];
            i3 = idx[j * 3 + 2];
            for (int l = 0; l < c; ++l) {
                grad_points[i1 * c + l] += grad_out[j * c + l] * w1;
                grad_points[i2 * c + l] += grad_out[j * c + l] * w2;
                grad_points[i3 * c + l] += grad_out[j * c + l] * w3;
            }
        }
        grad_out += n * c;
        idx += n * 3;
        weight += n * 3;
        grad_points += m * c;
    }
}

class ThreeNNOp : public OpKernel {
   public:
    explicit ThreeNNOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {
        const Tensor &xyz1_tensor = context->input(0);
        OP_REQUIRES(
            context,
            xyz1_tensor.dims() == 3 && xyz1_tensor.shape().dim_size(2) == 3,
            errors::InvalidArgument("ThreeNN expects (b,n,3) xyz1 shape."));
        int b = xyz1_tensor.shape().dim_size(0);
        int n = xyz1_tensor.shape().dim_size(1);

        const Tensor &xyz2_tensor = context->input(1);
        OP_REQUIRES(
            context,
            xyz2_tensor.dims() == 3 && xyz2_tensor.shape().dim_size(2) == 3,
            errors::InvalidArgument("ThreeNN expects (b,m,3) xyz2 shape."));
        int m = xyz2_tensor.shape().dim_size(1);

        Tensor *dist_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
                                    0, TensorShape{b, n, 3}, &dist_tensor));
        Tensor *idx_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
                                    1, TensorShape{b, n, 3}, &idx_tensor));

        auto xyz1_flat = xyz1_tensor.flat<float>();
        const float *xyz1 = &(xyz1_flat(0));
        auto xyz2_flat = xyz2_tensor.flat<float>();
        const float *xyz2 = &(xyz2_flat(0));
        auto dist_flat = dist_tensor->flat<float>();
        float *dist = &(dist_flat(0));
        auto idx_flat = idx_tensor->flat<int>();
        int *idx = &(idx_flat(0));
        threenn_cpu(b, n, m, xyz1, xyz2, dist, idx);
    }
};
REGISTER_KERNEL_BUILDER(Name("ThreeNN").Device(DEVICE_CPU), ThreeNNOp);

class ThreeInterpolateOp : public OpKernel {
   public:
    explicit ThreeInterpolateOp(OpKernelConstruction *context)
        : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {
        const Tensor &points_tensor = context->input(0);
        OP_REQUIRES(context, points_tensor.dims() == 3,
                    errors::InvalidArgument(
                        "ThreeInterpolate expects (b,m,c) points shape"));
        int b = points_tensor.shape().dim_size(0);
        int m = points_tensor.shape().dim_size(1);
        int c = points_tensor.shape().dim_size(2);

        const Tensor &idx_tensor = context->input(1);
        OP_REQUIRES(context,
                    idx_tensor.dims() == 3 &&
                        idx_tensor.shape().dim_size(0) == b &&
                        idx_tensor.shape().dim_size(2) == 3,
                    errors::InvalidArgument(
                        "ThreeInterpolate expects (b,n,3) idx shape"));
        int n = idx_tensor.shape().dim_size(1);
        const Tensor &weight_tensor = context->input(2);
        OP_REQUIRES(context,
                    weight_tensor.dims() == 3 &&
                        weight_tensor.shape().dim_size(0) == b &&
                        weight_tensor.shape().dim_size(1) == n &&
                        weight_tensor.shape().dim_size(2) == 3,
                    errors::InvalidArgument(
                        "ThreeInterpolate expects (b,n,3) weight shape"));

        Tensor *out_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
                                    0, TensorShape{b, n, c}, &out_tensor));

        auto points_flat = points_tensor.flat<float>();
        const float *points = &(points_flat(0));
        auto idx_flat = idx_tensor.flat<int>();
        const int *idx = &(idx_flat(0));
        auto weight_flat = weight_tensor.flat<float>();
        const float *weight = &(weight_flat(0));
        auto out_flat = out_tensor->flat<float>();
        float *out = &(out_flat(0));
        threeinterpolate_cpu(b, m, c, n, points, idx, weight, out);
    }
};
REGISTER_KERNEL_BUILDER(Name("ThreeInterpolate").Device(DEVICE_CPU),
                        ThreeInterpolateOp);

class ThreeInterpolateGradOp : public OpKernel {
   public:
    explicit ThreeInterpolateGradOp(OpKernelConstruction *context)
        : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {
        const Tensor &points_tensor = context->input(0);
        OP_REQUIRES(context, points_tensor.dims() == 3,
                    errors::InvalidArgument(
                        "ThreeInterpolateGrad expects (b,m,c) points shape"));
        int b = points_tensor.shape().dim_size(0);
        int m = points_tensor.shape().dim_size(1);
        int c = points_tensor.shape().dim_size(2);

        const Tensor &idx_tensor = context->input(1);
        OP_REQUIRES(
            context,
            idx_tensor.dims() == 3 && idx_tensor.shape().dim_size(0) == b,
            errors::InvalidArgument(
                "ThreeInterpolateGrad expects (b,n,3) idx shape"));
        int n = idx_tensor.shape().dim_size(1);
        const Tensor &weight_tensor = context->input(2);
        OP_REQUIRES(context,
                    weight_tensor.dims() == 3 &&
                        weight_tensor.shape().dim_size(0) == b &&
                        weight_tensor.shape().dim_size(1) == n &&
                        weight_tensor.shape().dim_size(2) == 3,
                    errors::InvalidArgument(
                        "ThreeInterpolateGrad expects (b,n,3) weight shape"));

        const Tensor &grad_out_tensor = context->input(3);
        OP_REQUIRES(context,
                    grad_out_tensor.dims() == 3 &&
                        grad_out_tensor.shape().dim_size(0) == b &&
                        grad_out_tensor.shape().dim_size(1) == n &&
                        grad_out_tensor.shape().dim_size(2) == c,
                    errors::InvalidArgument(
                        "ThreeInterpolateGrad expects (b,n,c) grad_out shape"));

        Tensor *grad_points_tensor = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, TensorShape{b, m, c},
                                                &grad_points_tensor));

        auto points_flat = points_tensor.flat<float>();
        const float *points = &(points_flat(0));
        auto idx_flat = idx_tensor.flat<int>();
        const int *idx = &(idx_flat(0));
        auto weight_flat = weight_tensor.flat<float>();
        const float *weight = &(weight_flat(0));
        auto grad_out_flat = grad_out_tensor.flat<float>();
        const float *grad_out = &(grad_out_flat(0));
        auto grad_points_flat = grad_points_tensor->flat<float>();
        float *grad_points = &(grad_points_flat(0));
        memset(grad_points, 0, sizeof(float) * b * m * c);
        threeinterpolate_grad_cpu(b, n, c, m, grad_out, idx, weight,
                                  grad_points);
    }
};
REGISTER_KERNEL_BUILDER(Name("ThreeInterpolateGrad").Device(DEVICE_CPU),
                        ThreeInterpolateGradOp);
