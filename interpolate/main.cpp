// code inspired by  https://github.com/aboulch/snapnet
// The IoUs (per class and average) and the accuracy are computed for each scene
// that is in the folder passed as argument, and then the global IoUs and global
// accuracy are computed and saved. More information below.
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <Eigen/Dense>
#include <algorithm>

#include "Core/Core.h"
#include "IO/IO.h"

static std::vector<std::string> possible_file_prefixes{
    // "bildstein_station1_xyz_intensity_rgb",
    // "bildstein_station3_xyz_intensity_rgb",
    // "bildstein_station5_xyz_intensity_rgb",
    // "domfountain_station1_xyz_intensity_rgb",
    // "domfountain_station2_xyz_intensity_rgb",
    // "domfountain_station3_xyz_intensity_rgb",
    // "neugasse_station1_xyz_intensity_rgb",
    // "sg27_station1_intensity_rgb",
    // "sg27_station2_intensity_rgb",

    // "sg27_station4_intensity_rgb",
    // "sg27_station5_intensity_rgb",
    // "sg27_station9_intensity_rgb",
    // "sg28_station4_intensity_rgb",
    "untermaederbrunnen_station1_xyz_intensity_rgb",
    // "untermaederbrunnen_station3_xyz_intensity_rgb",

    // "birdfountain_station1_xyz_intensity_rgb",
    // "castleblatten_station1_intensity_rgb",
    // "castleblatten_station5_xyz_intensity_rgb",
    // "marketplacefeldkirch_station1_intensity_rgb",
    // "marketplacefeldkirch_station4_intensity_rgb",
    // "marketplacefeldkirch_station7_intensity_rgb",
    // "sg27_station10_intensity_rgb",
    // "sg27_station3_intensity_rgb",
    // "sg27_station6_intensity_rgb",
    // "sg27_station8_intensity_rgb",
    // "sg28_station2_intensity_rgb",
    // "sg28_station5_xyz_intensity_rgb",
    // "stgallencathedral_station1_intensity_rgb",
    // "stgallencathedral_station3_intensity_rgb",
    // "stgallencathedral_station6_intensity_rgb"
};

class LabelCounter {
   public:
    void increment(int label) {
        if (finalized_) {
            throw std::runtime_error("Counter finalized");
        }
        label_counters_[label]++;
    }
    void finalize_label() {
        auto max_element_it =
            std::max_element(label_counters_.begin(), label_counters_.end());
        label_ = std::distance(label_counters_.begin(), max_element_it);
        finalized_ = true;
    }
    int get_label() {
        if (!finalized_) {
            throw std::runtime_error("Counter not finalized");
        }
        return label_;
    }

   private:
    std::vector<int> label_counters_ = std::vector<int>(9, 0);
    int label_ = 0;
    bool finalized_ = false;
};

// comparator for map_voxel_to_label_counter
struct Vector3iComp {
    bool operator()(const Eigen::Vector3i& v1,
                    const Eigen::Vector3i& v2) const {
        if (v1[0] < v2[0]) {
            return true;
        } else if (v1[0] == v2[0]) {
            if (v1[1] < v2[1]) {
                return true;
            } else if (v1[1] == v2[1] && v1[2] < v2[2]) {
                return true;
            }
        }
        return false;
    }
};

std::vector<int> read_labels(const std::string& file_path) {
    std::vector<int> labels;
    std::ifstream infile(file_path);
    int label;
    if (infile.fail()) {
        std::cerr << file_path << " not found at read_labels" << std::endl;
    } else {
        while (infile >> label) {
            labels.push_back(label);
        }
    }
    infile.close();
    return labels;
}

void write_labels(const std::vector<int> labels, const std::string& file_path) {
    std::ofstream out_file(file_path.c_str());
    if (out_file.fail()) {
        std::cerr << "Output file cannot be created: " << file_path
                  << " Consider creating the directory first" << std::endl;

    } else {
        std::cout << "Writting dense labels" << std::endl;
        for (const int& label : labels) {
            out_file << label << std::endl;
        }
        std::cout << "Output written to: " << file_path << std::endl;
    }
    out_file.close();
}

Eigen::Vector3i get_voxel(double x, double y, double z, double voxel_size) {
    int x_index = std::floor(x / voxel_size) + 0.5;
    int y_index = std::floor(y / voxel_size) + 0.5;
    int z_index = std::floor(z / voxel_size) + 0.5;
    return Eigen::Vector3i(x_index, y_index, z_index);
}

Eigen::Vector3i get_voxel(const Eigen::Vector3d& point, double voxel_size) {
    return get_voxel(point(0), point(1), point(2), voxel_size);
}

// The pointnet2 network only takes up to a few thousand points at a time,
// so we do not have the real results yet But we can get results on a
// sparser point cloud (after decimation, and after we dynamically sample
// inputs on the decimated point clouds. The job of this function is to take
// a very sparse point cloud (a few hundred thousand points) with
// predictions by the network and to interpolate the results to the much
// denser raw point clouds. This is achieved by a division of the space into
// a voxel grid, implemented as a map called map_voxel_to_label_counter.
// First the sparse point cloud is iterated and the map is constructed. We store
// for each voxel and each label the nb of points from the sparse cloud
// and with the right label was in the voxel. Then we assign to each
// voxel the label which got the most points. And finally we can iterate
// the dense point cloud and dynamically assign labels according to the
// map_voxel_to_label_counter. IoU per class and accuracy are calculated at
// the end.
void interpolate_labels_one_point_cloud(const std::string& input_dense_dir,
                                        const std::string& input_sparse_dir,
                                        const std::string& output_dir,
                                        const std::string& file_prefix,
                                        double voxel_size) {
    std::cout << "[Interpolating] " + file_prefix << std::endl;

    // Paths
    std::string dense_points_path =
        input_dense_dir + "/" + file_prefix + ".pcd";
    std::string dense_labels_path = output_dir + "/" + file_prefix + ".labels";
    std::string sparse_points_path =
        input_sparse_dir + "/" + file_prefix + ".pcd";
    std::string sparse_labels_path =
        input_sparse_dir + "/" + file_prefix + ".labels";

    // Read sparse points
    open3d::PointCloud sparse_pcd;
    open3d::ReadPointCloud(sparse_points_path, sparse_pcd);
    std::cout << sparse_pcd.points_.size() << " sparse points" << std::endl;

    // Read sparse labels
    std::vector<int> sparse_labels = read_labels(sparse_labels_path);
    std::cout << sparse_labels.size() << " sparse labels" << std::endl;

    // Build voxel to label container map. This is the main data structure.
    // First we build and finalize counter with sparse point could, then look up
    // the map to interpolate the large point cloud.
    std::map<Eigen::Vector3i, LabelCounter, Vector3iComp>
        map_voxel_to_label_counter;

    for (size_t i = 0; i < sparse_labels.size(); ++i) {
        Eigen::Vector3i voxel = get_voxel(sparse_pcd.points_[i], voxel_size);
        if (map_voxel_to_label_counter.count(voxel) == 0) {
            LabelCounter ilc;
            map_voxel_to_label_counter[voxel] = ilc;
        }
        map_voxel_to_label_counter[voxel].increment(sparse_labels[i]);
    }
    for (auto it = map_voxel_to_label_counter.begin();
         it != map_voxel_to_label_counter.end(); it++) {
        it->second.finalize_label();
    }
    std::cout << "Number of registered voxels: "
              << map_voxel_to_label_counter.size() << std::endl;

    // Read dense points
    open3d::PointCloud dense_pcd;
    open3d::ReadPointCloud(dense_points_path, dense_pcd);
    std::cout << dense_pcd.points_.size() << " dense points" << std::endl;

    // Interpolate to dense point cloud
    std::vector<int> dense_labels;
    size_t num_processed_points = 0;
    size_t num_miss = 0;
    for (Eigen::Vector3d& point : dense_pcd.points_) {
        Eigen::Vector3i voxel = get_voxel(point, voxel_size);
        int label;
        if (map_voxel_to_label_counter.count(voxel) == 0) {
            // TODO: change to nearest neighbor search
            num_miss++;
            label = 0;
        } else {
            label = map_voxel_to_label_counter[voxel].get_label();
        }
        dense_labels.push_back(label);

        num_processed_points++;
        if (num_processed_points % 1000000 == 0) {
            size_t num_hit = num_processed_points - num_miss;
            double hit_rate = (double)num_hit / num_processed_points * 100;
            std::cout << num_processed_points << " processed, " << num_hit
                      << " (" << hit_rate << "%) hit" << std::endl;
        }
    }
    std::cout << dense_labels.size() << " dense labels generated" << std::endl;

    // Write label
    write_labels(dense_labels, dense_labels_path);
}

int main(int argc, char** argv) {
    // Parse arguments
    if (argc < 5) {
        std::cerr << "USAGE: " << argv[0] << " input_dense_dir"
                  << " input_sparse_dir"
                  << " output_dir " << std::endl;
        exit(1);
    }
    std::string input_dense_dir = argv[1];
    std::string input_sparse_dir = argv[2];
    std::string output_dir = argv[3];
    double voxel_size = strtof(argv[4], NULL);
    std::cout << "Using voxel size: " << voxel_size << std::endl;

    // Collect all existing files
    std::vector<std::string> file_prefixes;
    for (unsigned int i = 0; i < possible_file_prefixes.size(); i++) {
        std::string sparse_labels_path = std::string(input_sparse_dir) + "/" +
                                         possible_file_prefixes[i] + ".labels";
        std::ifstream sparse_points_file(sparse_labels_path.c_str());
        if (!sparse_points_file.fail()) {
            file_prefixes.push_back(possible_file_prefixes[i]);
            std::cout << "Found: " + possible_file_prefixes[i] << std::endl;
        }
        sparse_points_file.close();
    }

    // Interpolate
    for (const std::string& file_prefix : file_prefixes) {
        interpolate_labels_one_point_cloud(input_dense_dir, input_sparse_dir,
                                           output_dir, file_prefix, voxel_size);
    }

    return 0;
}
