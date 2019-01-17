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
    for (size_t i = 0; i < num_elements; i += 3) {
        eigen_vectors.emplace_back(buffer[i], buffer[i + 1], buffer[i + 2]);
    }
    return eigen_vectors;
}

inline int get_most_frequent_element(const std::vector<int> &labels) {
    int max_count = 0;
    int most_frequent_label = -1;
    std::unordered_map<int, int> map_label_to_count;
    for (const int &label : labels) {
        map_label_to_count[label]++;
        int count = map_label_to_count[label];
        if (count > max_count) {
            most_frequent_label = label;
            max_count = count;
        }
    }
    return most_frequent_label;
}

static const std::vector<std::vector<uint8_t>> map_label_to_color{
    {255, 255, 255}, {0, 0, 255},   {128, 0, 0}, {255, 0, 255}, {0, 128, 0},
    {255, 0, 0},     {128, 0, 128}, {0, 0, 128}, {128, 128, 0}};

///////////////////////////////////////////////////////////////////////////////
// InterpolateLabelWithColor
///////////////////////////////////////////////////////////////////////////////
REGISTER_OP("InterpolateLabelWithColor")
    .Input("sparse_points: float32")  // (num_sparse_points, 3)
    .Input("sparse_labels: int32")    // (num_sparse_points, 3)
    .Input("dense_points: float32")   // (num_dense_points, 3)
    .Input("knn: int32")              // (num_dense_points, 3)
    .Output("dense_labels: int32")    // (num_dense_points,)
    .Output("dense_colors: uint8")    // (num_dense_points, 3)
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        // (num_dense_points, 3)
        ::tensorflow::shape_inference::ShapeHandle dense_points_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &dense_points_shape));
        // (num_dense_points,)
        ::tensorflow::shape_inference::ShapeHandle dense_labels_shape =
            c->MakeShape({c->Dim(dense_points_shape, 0)});
        c->set_output(0, dense_labels_shape);
        c->set_output(1, dense_points_shape);
        return Status::OK();
    });

void interpolate_label_with_color_cpu(
    int num_sparse_points, int num_dense_points, const float *sparse_points,
    const int *sparse_labels, const float *dense_points, int *dense_labels,
    uint8_t *dense_colors, int knn) {
    open3d::PointCloud reference_pcd;
    reference_pcd.points_ =
        buffer_to_eigen_vector(sparse_points, num_sparse_points * 3);
    open3d::KDTreeFlann reference_kd_tree(reference_pcd);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t j = 0; j < num_dense_points; ++j) {
        // Move vectors inside if using omp, outside if omp disabled
        std::vector<int> candidate_indices(knn);
        std::vector<double> candidate_dists(knn);
        Eigen::Vector3d target_point;
        size_t target_point_idx = j * 3;
        target_point(0) = dense_points[target_point_idx];
        target_point(1) = dense_points[target_point_idx + 1];
        target_point(2) = dense_points[target_point_idx + 2];
        int knn_found = reference_kd_tree.SearchKNN(
            target_point, knn, candidate_indices, candidate_dists);

        // Find most frequent label
        int max_count = 0;
        int most_frequent_label = -1;
        std::unordered_map<int, int> map_label_to_count;
        int label;
        for (size_t k = 0; k < knn_found; ++k) {
            label = sparse_labels[candidate_indices[k]];
            map_label_to_count[label]++;
            if (map_label_to_count[label] > max_count) {
                most_frequent_label = label;
                max_count = map_label_to_count[label];
            }
        }
        dense_labels[j] = most_frequent_label;

        // Assign colors
        dense_colors[j * 3] = map_label_to_color[most_frequent_label][0];
        dense_colors[j * 3 + 1] = map_label_to_color[most_frequent_label][1];
        dense_colors[j * 3 + 2] = map_label_to_color[most_frequent_label][2];
    }
}

class InterpolateLabelWithColorOp : public OpKernel {
   public:
    explicit InterpolateLabelWithColorOp(OpKernelConstruction *context)
        : OpKernel(context) {}

    void Compute(OpKernelContext *context) override {
        // Input: sparse_points
        const Tensor &sparse_points_tensor = context->input(0);
        OP_REQUIRES(context,
                    sparse_points_tensor.dims() == 2 &&
                        sparse_points_tensor.shape().dim_size(1) == 3,
                    errors::InvalidArgument(
                        "sparse_points must be: (num_sparse_points, 3)"));
        int num_sparse_points = sparse_points_tensor.shape().dim_size(0);
        auto sparse_points_flat = sparse_points_tensor.flat<float>();
        const float *sparse_points = &(sparse_points_flat(0));

        // Input: sparse_labels
        const Tensor &sparse_labels_tensor = context->input(1);
        OP_REQUIRES(
            context,
            sparse_labels_tensor.dims() == 1 &&
                sparse_labels_tensor.shape().dim_size(0) == num_sparse_points,
            errors::InvalidArgument(
                "sparse_labels must be: (num_sparse_points, 3)"));
        auto sparse_labels_flat = sparse_labels_tensor.flat<int>();
        const int *sparse_labels = &(sparse_labels_flat(0));

        // Input: dense_points
        const Tensor &dense_points_tensor = context->input(2);
        OP_REQUIRES(context,
                    dense_points_tensor.dims() == 2 &&
                        dense_points_tensor.shape().dim_size(1) == 3,
                    errors::InvalidArgument(
                        "dense_points must be: (num_dense_points, 3)"));
        int num_dense_points = dense_points_tensor.shape().dim_size(0);
        auto dense_points_flat = dense_points_tensor.flat<float>();
        const float *dense_points = &(dense_points_flat(0));

        // Input: knn
        const Tensor &knn_tensor = context->input(3);
        OP_REQUIRES(context, knn_tensor.dims() == 0,
                    errors::InvalidArgument("knn must be an int scalar"));
        auto knn_flat = knn_tensor.flat<int>();
        const int knn = knn_flat(0);

        // Output: dense_labels
        Tensor *dense_labels_tensor = nullptr;
        OP_REQUIRES_OK(
            context, context->allocate_output(0, TensorShape{num_dense_points},
                                              &dense_labels_tensor));
        auto dense_labels_flat = dense_labels_tensor->flat<int>();
        int *dense_labels = &(dense_labels_flat(0));

        // Output: dense_colors
        Tensor *dense_colors_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(
                                    1, TensorShape{num_dense_points, 3},
                                    &dense_colors_tensor));
        auto dense_colors_flat = dense_colors_tensor->flat<uint8_t>();
        uint8_t *dense_colors = &(dense_colors_flat(0));

        interpolate_label_with_color_cpu(
            num_sparse_points, num_dense_points, sparse_points, sparse_labels,
            dense_points, dense_labels, dense_colors, knn);
    }
};
REGISTER_KERNEL_BUILDER(Name("InterpolateLabelWithColor").Device(DEVICE_CPU),
                        InterpolateLabelWithColorOp);

///////////////////////////////////////////////////////////////////////////////
// ThreeNN
///////////////////////////////////////////////////////////////////////////////
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

// Find three nearest neighbors with square distance
// input: xyz1 (b,n,3), xyz2(b,m,3)
// output: dists (b,n,3), indices (b,n,3)
// E.g.
// - target_points (b, n, 3): e.g. (64, 8192, 3), the "3" here is x, y, z
// - reference_points (b, m, 3): e.g. (64, 1024, 3), the "3" here is x, y, z
// - dists (b, n, 3): (64, 8192, 3), for each input point in target_points,
//   find 3 nearest neighbors in base_points and return the distances squared,
//   the "3" means "3" nearest neighbors
// - indices (b, n, 3): (64, 8192, 3), for each input point in target_points,
//   find 3 nearest neighbors in base_points and return the indexes in
//   base_points, the "3" means "3" nearest neighbors
void threenn_cpu(int b, int n, int m, const float *xyz1, const float *xyz2,
                 float *dists, int *indices) {
    // OPENMP only sees benefits if b is large, e.g. b == 64
    // #ifdef _OPENMP
    // #pragma omp parallel for schedule(static)
    // #endif
    for (int batch_index = 0; batch_index < b; ++batch_index) {
        std::vector<int> three_indices;
        std::vector<double> three_dists;
        open3d::PointCloud target_pcd;
        open3d::PointCloud reference_pcd;

        target_pcd.points_ =
            buffer_to_eigen_vector(xyz1 + batch_index * n * 3, n * 3);
        reference_pcd.points_ =
            buffer_to_eigen_vector(xyz2 + batch_index * m * 3, m * 3);
        open3d::KDTreeFlann reference_kd_tree(reference_pcd);

        for (size_t j = 0; j < n; ++j) {
            reference_kd_tree.SearchKNN(target_pcd.points_[j], 3, three_indices,
                                        three_dists);
            size_t start_idx = batch_index * n * 3 + j * 3;
            indices[start_idx + 0] = three_indices[0];
            indices[start_idx + 1] = three_indices[1];
            indices[start_idx + 2] = three_indices[2];
            dists[start_idx + 0] = three_dists[0];
            dists[start_idx + 1] = three_dists[1];
            dists[start_idx + 2] = three_dists[2];
        }
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

///////////////////////////////////////////////////////////////////////////////
// ThreeInterpolate
///////////////////////////////////////////////////////////////////////////////
REGISTER_OP("ThreeInterpolate")
    .Input("points: float32")
    .Input("idx: int32")
    .Input("weight: float32")
    .Output("out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        ::tensorflow::shape_inference::ShapeHandle dims1;  // (b,m,c)
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &dims1));
        ::tensorflow::shape_inference::ShapeHandle dims2;  // (b,n,3)
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &dims2));
        // (b,n,c)
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape(
            {c->Dim(dims1, 0), c->Dim(dims2, 1), c->Dim(dims1, 2)});
        c->set_output(0, output);
        return Status::OK();
    });

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

///////////////////////////////////////////////////////////////////////////////
// ThreeInterpolateGrad
///////////////////////////////////////////////////////////////////////////////
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
