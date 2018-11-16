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

Eigen::Vector3i get_voxel(float x, float y, float z, float voxel_size) {
    int x_index = std::floor(x / voxel_size) + 0.5;
    int y_index = std::floor(y / voxel_size) + 0.5;
    int z_index = std::floor(z / voxel_size) + 0.5;
    return Eigen::Vector3i(x_index, y_index, z_index);
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
                                        float voxel_size) {
    std::cout << "[Interpolating] " + file_prefix << std::endl;

    // Load files
    std::string sparse_points_path =
        input_sparse_dir + "/" + file_prefix + "_aggregated.txt";
    std::string sparse_labels_path =
        input_sparse_dir + "/" + file_prefix + "_pred.txt";
    std::string dense_points_path =
        input_dense_dir + "/" + file_prefix + ".txt";
    std::string out_labels_path = output_dir + "/" + file_prefix + ".labels";

    std::ifstream sparse_points_file(sparse_points_path.c_str());
    std::ifstream sparse_labels_file(sparse_labels_path.c_str());
    std::ifstream dense_points_file(dense_points_path.c_str());
    std::ofstream out_labels_file(out_labels_path.c_str());

    if (sparse_points_file.fail()) {
        std::cerr << sparse_points_path << " not found" << std::endl;
    }
    if (sparse_labels_file.fail()) {
        std::cerr << sparse_labels_path << " not found" << std::endl;
    }
    if (dense_points_file.fail()) {
        std::cerr << dense_points_path << " not found" << std::endl;
    }
    if (out_labels_file.fail()) {
        std::cerr << "Output file cannot be created" << std::endl;
    }

    // Read sparse points and labels, build voxel to label container map
    std::string line_point;
    std::string line_label;
    std::map<Eigen::Vector3i, LabelCounter, Vector3iComp>
        map_voxel_to_label_counter;

    size_t num_sparse_points = 0;
    while (getline(sparse_points_file, line_point) &&
           getline(sparse_labels_file, line_label)) {
        std::stringstream sstr_label(line_label);
        int label;
        sstr_label >> label;

        std::stringstream sstr(line_point);
        float x, y, z;
        int r, g, b;
        std::string v;
        sstr >> v >> x >> y >> z >> r >> g >> b;
        Eigen::Vector3i voxel = get_voxel(x, y, z, voxel_size);

        if (map_voxel_to_label_counter.count(voxel) == 0) {
            LabelCounter ilc;
            map_voxel_to_label_counter[voxel] = ilc;
        }
        map_voxel_to_label_counter[voxel].increment(label);
        num_sparse_points++;
    }
    std::cout << "Number of sparse points: " << num_sparse_points << std::endl;

    for (auto it = map_voxel_to_label_counter.begin();
         it != map_voxel_to_label_counter.end(); it++) {
        it->second.finalize_label();
    }
    std::cout << "Number of registered voxels: "
              << map_voxel_to_label_counter.size() << std::endl;

    // Interpolate to dense point cloud
    // TODO: change to nearest neighbor search
    size_t num_processed_points = 0;
    size_t num_miss = 0;
    while (getline(dense_points_file, line_point)) {
        std::stringstream sstr(line_point);
        float x, y, z;
        int intensity, r, g, b;
        sstr >> x >> y >> z >> intensity >> r >> g >> b;

        int label;
        Eigen::Vector3i voxel = get_voxel(x, y, z, voxel_size);
        if (map_voxel_to_label_counter.count(voxel) == 0) {
            num_miss++;
            label = 0;
        } else {
            label = map_voxel_to_label_counter[voxel].get_label();
        }
        out_labels_file << label << std::endl;

        num_processed_points++;
        if (num_processed_points % 1000000 == 0) {
            size_t num_hit = num_processed_points - num_miss;
            float hit_rate = (float)num_hit / num_processed_points * 100;
            std::cout << num_processed_points << " processed, " << num_hit
                      << " (" << hit_rate << "%) hit" << std::endl;
        }
    }
    std::cout << "Label output: " << out_labels_path << std::endl;

    sparse_points_file.close();
    sparse_labels_file.close();
    dense_points_file.close();
    out_labels_file.close();
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
    float voxel_size = strtof(argv[4], NULL);
    std::cout << "Using voxel size: " << voxel_size << std::endl;

    // Collect all existing files
    std::vector<std::string> file_prefixes;
    for (unsigned int i = 0; i < possible_file_prefixes.size(); i++) {
        std::string sparse_labels_path = std::string(input_sparse_dir) + "/" +
                                         possible_file_prefixes[i] +
                                         "_pred.txt";
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
