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

class InterpolationLabelsContainer {
    std::vector<int> label_count;
    int label;

   public:
    InterpolationLabelsContainer() {
        label_count = std::vector<int>(9, 0);
        label = 0;
    }
    void add_label(int label) { label_count[label]++; }
    void calculate_label() {
        label = max_element(label_count.begin(), label_count.end()) -
                label_count.begin();
    }
    int get_label() { return label; }
    void set_label(int l) { label = l; }
};

// comparator for voxels
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

// The pointnet2 network only takes up to a few thousand points at a time,
// so we do not have the real results yet But we can get results on a
// sparser point cloud (after decimation, and after we dynamically sample
// inputs on the decimated point clouds. The job of this function is to take
// a very sparse point cloud (a few hundred thousand points) with
// predictions by the network and to interpolate the results to the much
// denser raw point clouds. This is achieved by a division of the space into
// a voxel grid, implemented as a map called voxels. First the sparse point
// cloud is iterated and the map is constructed. We store for each voxel and
// each label the nb of points from the sparse cloud and with the right
// label was in the voxel. Then we assign to each voxel the label which got
// the most points. And finally we can iterate the dense point cloud and
// dynamically assign labels according to the voxels. IoU per class and
// accuracy are calculated at the end.
void interpolate_labels_one_point_cloud(const std::string& input_dense_dir,
                                        const std::string& input_sparse_dir,
                                        const std::string& output_dir,
                                        const std::string& file_prefix,
                                        const float& voxel_size,
                                        const bool& export_labels) {
    // File names
    std::string sparse_points_path =
        input_sparse_dir + "/" + file_prefix + "_aggregated.txt";
    std::string sparse_labels_path =
        input_sparse_dir + "/" + file_prefix + "_pred.txt";
    std::string dense_points_path =
        input_dense_dir + "/" + file_prefix + ".txt";

    std::ifstream sparse_points_file(sparse_points_path.c_str());
    std::ifstream sparse_labels_file(sparse_labels_path.c_str());
    if (sparse_points_file.fail()) {
        std::cerr << sparse_points_path << " not found" << std::endl;
    }
    if (sparse_labels_file.fail()) {
        std::cerr << sparse_labels_path << " not found" << std::endl;
    }

    std::string line;
    std::string line_labels;
    std::map<Eigen::Vector3i, InterpolationLabelsContainer, Vector3iComp>
        voxels;
    while (getline(sparse_points_file, line)) {
        getline(sparse_labels_file, line_labels);
        std::stringstream sstr_label(line_labels);
        int label;
        sstr_label >> label;

        std::stringstream sstr(line);
        float x, y, z;
        int r, g, b;
        std::string v;
        sstr >> v >> x >> y >> z >> r >> g >> b;
        int x_id = std::floor(x / voxel_size) +
                   0.5;  // + 0.5, centre du voxel (k1*res, k2*res)
        int y_id = std::floor(y / voxel_size) + 0.5;
        int z_id = std::floor(z / voxel_size) + 0.5;
        Eigen::Vector3i vox(x_id, y_id, z_id);

        if (voxels.count(vox) == 0) {
            InterpolationLabelsContainer ilc;
            voxels[vox] = ilc;
        }
        voxels[vox].add_label(label);
    }

    int j = 0;
    for (auto it = voxels.begin(); it != voxels.end(); it++, j++) {
        it->second.calculate_label();
    }
    std::cout << "number of registered voxels : " << j << std::endl;

    // now we move on to the dense cloud
    // don't know how to open only when necessary
    std::string out_label_filename = output_dir + "/" + file_prefix + ".labels";
    std::ofstream out_label(out_label_filename.c_str());
    std::ifstream ifs2(dense_points_path.c_str());
    if (ifs2.fail())
        std::cerr << dense_points_path << " not found" << std::endl;

    std::cout << "labeling raw point cloud";
    if (export_labels) std::cout << " and exporting labels";
    std::cout << std::endl;

    size_t pt_id = 0;
    int nb_labeled_pts = 0;
    std::vector<int> successes(9, 0);
    std::vector<int> unions(9, 0);
    int holes_nb = 0;
    while (getline(ifs2, line)) {
        pt_id++;
        if ((pt_id + 1) % 1000000 == 0) {
            std::cout << (pt_id + 1) / 1000000 << " M. " << holes_nb
                      << " holes encountered so far " << std::endl;
        }
        std::stringstream sstr(line);
        float x, y, z;
        int intensity, r, g, b;
        sstr >> x >> y >> z >> intensity >> r >> g >> b;

        int x_id = std::floor(x / voxel_size) +
                   0.5;  // + 0.5, centre du voxel (k1*res, k2*res)
        int y_id = std::floor(y / voxel_size) + 0.5;
        int z_id = std::floor(z / voxel_size) + 0.5;

        int label;
        Eigen::Vector3i vox(x_id, y_id, z_id);
        if (voxels.count(vox) == 0) {
            holes_nb++;
            // std::cout << "voxel unlabeled. fetching closest voxel" <<
            // std::endl;
            // here no point in the voxel was in the aggregated point cloud
            // we assign it the label 0 for now (TODO : improve by nearest
            // neighbor search using octree ?)
            label = 0;
        } else {
            label = voxels[vox].get_label();
        }

        if (export_labels) out_label << label << std::endl;
    }
    out_label.close();
}

int main(int argc, char** argv) {
    // Parse arguments
    if (argc < 6) {
        std::cerr << "USAGE: " << argv[0] << " input_dense_dir"
                  << " input_sparse_dir"
                  << " output_dir "
                  << " export_labels" << std::endl;
        exit(1);
    }
    std::string input_dense_dir = argv[1];
    std::string input_sparse_dir = argv[2];
    std::string output_dir = argv[3];
    float voxel_size = strtof(argv[4], NULL);
    bool export_labels = std::string(argv[5]) == "1";

    // Collect all existing files
    std::vector<std::string> file_prefixes;
    for (unsigned int i = 0; i < possible_file_prefixes.size(); i++) {
        std::string sparse_labels_path = std::string(input_sparse_dir) + "/" +
                                         possible_file_prefixes[i] +
                                         "_pred.txt";
        std::ifstream sparse_points_file(sparse_labels_path.c_str());
        if (!sparse_points_file.fail()) {
            file_prefixes.push_back(possible_file_prefixes[i]);
            std::cout << "Found " + possible_file_prefixes[i] << std::endl;
        }
        sparse_points_file.close();
    }

    for (unsigned int i = 0; i < file_prefixes.size(); i++) {
        std::cout << "interpolation for " + file_prefixes[i] << std::endl;
        interpolate_labels_one_point_cloud(input_dense_dir, input_sparse_dir,
                                           output_dir, file_prefixes[i],
                                           voxel_size, export_labels);
    }

    return 0;
}
