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
    "bildstein_station1_xyz_intensity_rgb",
    "bildstein_station3_xyz_intensity_rgb",
    "bildstein_station5_xyz_intensity_rgb",
    "domfountain_station1_xyz_intensity_rgb",
    "domfountain_station2_xyz_intensity_rgb",
    "domfountain_station3_xyz_intensity_rgb",
    "neugasse_station1_xyz_intensity_rgb",
    "sg27_station1_intensity_rgb",
    "sg27_station2_intensity_rgb",
    "sg27_station4_intensity_rgb",
    "sg27_station5_intensity_rgb",
    "sg27_station9_intensity_rgb",
    "sg28_station4_intensity_rgb",
    "untermaederbrunnen_station1_xyz_intensity_rgb",
    "untermaederbrunnen_station3_xyz_intensity_rgb",
    "birdfountain_station1_xyz_intensity_rgb",
    "castleblatten_station1_intensity_rgb",
    "castleblatten_station5_xyz_intensity_rgb",
    "marketplacefeldkirch_station1_intensity_rgb",
    "marketplacefeldkirch_station4_intensity_rgb",
    "marketplacefeldkirch_station7_intensity_rgb",
    "sg27_station10_intensity_rgb",
    "sg27_station3_intensity_rgb",
    "sg27_station6_intensity_rgb",
    "sg27_station8_intensity_rgb",
    "sg28_station2_intensity_rgb",
    "sg28_station5_xyz_intensity_rgb",
    "stgallencathedral_station1_intensity_rgb",
    "stgallencathedral_station3_intensity_rgb",
    "stgallencathedral_station6_intensity_rgb"};

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
std::pair<int, std::pair<std::vector<int>, std::vector<int>>>
interpolate_labels_one_point_cloud(const std::string& input_dense_dir,
                                   const std::string& input_sparse_dir,
                                   const std::string& output_dir,
                                   const std::string& file_prefix,
                                   const float& voxel_size,
                                   const bool& export_labels) {
    std::string filename_sparse =
        input_sparse_dir + "/" + file_prefix + "_aggregated.txt";
    std::string filename_labels_sparse =
        input_sparse_dir + "/" + file_prefix + "_pred.txt";
    std::string filename_dense = input_dense_dir + "/" + file_prefix + ".txt";
    std::string filename_labels_dense =
        input_dense_dir + "/" + file_prefix + ".labels";
    std::ifstream ifs(filename_sparse.c_str());
    if (ifs.fail()) std::cerr << filename_sparse << " not found" << std::endl;
    std::ifstream ifs_labels(filename_labels_sparse.c_str());
    if (ifs_labels.fail())
        std::cerr << filename_labels_sparse << " not found" << std::endl;
    std::string line;
    std::string line_labels;
    int pt_id = 0;

    std::map<Eigen::Vector3i, InterpolationLabelsContainer, Vector3iComp>
        voxels;
    while (getline(ifs, line)) {
        pt_id++;
        if ((pt_id + 1) % 1000000 == 0) {
            std::cout << (pt_id + 1) / 1000000 << " M" << std::endl;
        }
        getline(ifs_labels, line_labels);
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
    std::ifstream ifs2(filename_dense.c_str());
    if (ifs2.fail()) std::cerr << filename_dense << " not found" << std::endl;
    std::ifstream ifs_labels2(filename_labels_dense.c_str());
    bool compute_perfs = (!ifs_labels2.fail());
    std::cout << "labeling raw point cloud";
    if (export_labels) std::cout << " and exporting labels";
    if (compute_perfs) std::cout << " and computing performances";
    std::cout << std::endl;
    if (!compute_perfs)
        std::cout << filename_labels_dense
                  << " not found, assuming this is testing dataset"
                  << std::endl;
    pt_id = 0;
    int nb_labeled_pts = 0;
    std::vector<int> successes(9, 0);
    std::vector<int> unions(9, 0);
    int holes_nb = 0;
    while (getline(ifs2, line)) {
        pt_id++;
        if ((pt_id + 1) % 1000000 == 0) {
            std::cout << (pt_id + 1) / 1000000 << " M. " << holes_nb
                      << " holes encoutered so far " << std::endl;
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

        if (compute_perfs) {
            getline(ifs_labels2, line_labels);
            std::stringstream sstr_label(line_labels);
            int ground_truth;
            sstr_label >> ground_truth;
            // continue if point is unlabeled
            if (ground_truth == 0) continue;
            unions[ground_truth]++;
            nb_labeled_pts++;
            if (label == ground_truth) {
                successes[label]++;
            } else {
                unions[label]++;
            }
        }
    }
    out_label.close();

    std::string perf_filename = output_dir + "/" + file_prefix + "_perf.txt";
    std::ofstream output(perf_filename.c_str());
    if (compute_perfs) output << "Performances of " + file_prefix << std::endl;
    std::string classes[9] = {
        "unlabeled",       "man-made terrain",   "natural terrain",
        "high vegetation", "low vegetation",     "buildings",
        "hard scape",      "scanning artefacts", "cars"};
    int nb_of_successes = 0;
    float sum_IoUs = 0;
    std::vector<float> IoUs(9, 0);
    if (compute_perfs) {
        for (int i = 0; i < 9; i++) {
            IoUs[i] = successes[i] / float(unions[i]);
            sum_IoUs += IoUs[i];
            std::cout << IoUs[i] << " ";
            output << "IoU of " << classes[i] << IoUs[i] << std::endl;
            nb_of_successes += successes[i];
        }
        output << "global accuracy : "
               << nb_of_successes / float(nb_labeled_pts) << std::endl;
        output << "IoU averaged on 8 classes : " << sum_IoUs / 8. << std::endl;
        std::cout << std::endl << nb_labeled_pts << std::endl;
    }
    return std::pair<int, std::pair<std::vector<int>, std::vector<int>>>(
        nb_labeled_pts,
        std::pair<std::vector<int>, std::vector<int>>(successes, unions));
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
        std::string filename_labels_sparse = std::string(input_sparse_dir) +
                                             "/" + possible_file_prefixes[i] +
                                             "_pred.txt";
        std::ifstream ifs(filename_labels_sparse.c_str());
        if (!ifs.fail()) {
            file_prefixes.push_back(possible_file_prefixes[i]);
            std::cout << "Found " + possible_file_prefixes[i] << std::endl;
        }
        ifs.close();
    }

    int total_nb_labeled_pts = 0;
    for (unsigned int i = 0; i < file_prefixes.size(); i++) {
        std::cout << "interpolation for " + file_prefixes[i] << std::endl;
        std::pair<int, std::pair<std::vector<int>, std::vector<int>>>
            scene_perfs = interpolate_labels_one_point_cloud(
                input_dense_dir, input_sparse_dir, output_dir, file_prefixes[i],
                voxel_size, export_labels);
    }

    return 0;
}
