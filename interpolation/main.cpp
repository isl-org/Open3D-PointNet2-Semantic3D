/*
 *code inspired by  https://github.com/aboulch/snapnet
 */#include  <iostream>
#include  <fstream>
#include  <string>
#include  <sstream>
#include  <vector>
#include  <map>
#include  <Eigen/Dense>
#include  <algorithm>

class Interpolation_labels_container{
    std::vector<int> label_count;
    long int label;
public:
    Interpolation_labels_container(){label_count = std::vector<int>(9,0); label = 0;}
    void add_label(int label){label_count[label]++;}
    void calculate_label(){label = max_element(label_count.begin(), label_count.end()) - label_count.begin();}
    long int get_label(){return label;}
};

// comparator for voxels
struct Vector3icomp {
    bool operator() (const Eigen::Vector3i& v1, const Eigen::Vector3i& v2) const{
        if(v1[0] < v2[0]){
            return true;
        }else if(v1[0] == v2[0]){
            if(v1[1] < v2[1]){
                return true;
            }else if(v1[1] == v2[1] && v1[2] < v2[2]){
                return true;
            }
        }
        return false;
    }
};


std::pair<int, std::pair<std::vector<int>, std::vector<int>>> interpolate_labels_one_point_cloud(const std::string& input_dense_dir,
                                                                                                 const std::string& input_sparse_dir,
                                                                                                 const std::string& filename,
                                                                                                 const float& voxel_size) {
    std::string filename_sparse =input_sparse_dir + filename + "_agregated.txt";
    std::string filename_labels_sparse =input_sparse_dir + filename + "_pred.txt";
    std::string filename_dense =input_dense_dir + filename + ".txt";
    std::string filename_labels_dense =input_sparse_dir + filename + ".labels";
    std::ifstream ifs(filename_sparse.c_str());
    if (ifs.fail()) std::cout << "filename for agregated point cloud not found" << std::endl;
    std::ifstream ifs_labels(filename_labels_sparse.c_str());
    if (ifs_labels.fail()) std::cout << "filename for agregated point cloud labels not found" << std::endl;
    std::string line;
    std::string line_labels;
    int pt_id = 0;

    std::map<Eigen::Vector3i, Interpolation_labels_container, Vector3icomp> voxels;
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
        if ((pt_id + 1) % 1000000 == 0) {
            std::cout << v << " " << x << " " << y << " " << z << " " << r << " " << g << " " << b << std::endl;
        }
        int x_id = std::floor(x / voxel_size) + 0.5; // + 0.5, centre du voxel (k1*res, k2*res)
        int y_id = std::floor(y / voxel_size) + 0.5;
        int z_id = std::floor(z / voxel_size) + 0.5;

        Eigen::Vector3i vox(x_id, y_id, z_id);

        if (voxels.count(vox) == 0) {
            Interpolation_labels_container ilc;
            voxels[vox] = ilc;
        }
        voxels[vox].add_label(label);
    }
    int j = 0;
    for (std::map<Eigen::Vector3i, Interpolation_labels_container>::iterator it = voxels.begin();
         it != voxels.end(); it++, j++) {
        it->second.calculate_label();
    }
    std::cout << "nombre de voxels enregistres : " << j << std::endl;


    // now we move on to the dense cloud
    std::cout << "processing point cloud" << std::endl;
    std::ifstream ifs2 (filename_dense.c_str());
    if (ifs2.fail()) std::cout << "filename for dense point cloud not found" << std::endl;
    std::ifstream ifs_labels2 (filename_labels_dense.c_str());
    if (ifs_labels2.fail()) std::cout << "filename for point cloud labels not found" << std::endl;
    pt_id = 0;
    int nb_labeled_pts = 0;
    std::vector<int> successes(9, 0);
    std::vector<int> unions(9, 0);
    while(getline(ifs2,line)){
        pt_id++;
        if((pt_id+1)%1000000==0){
            std::cout << (pt_id+1)/1000000 << " M" << std::endl;
        }
        getline(ifs_labels2,line_labels);
        std::stringstream sstr_label(line_labels);
        int ground_truth;
        int label;
        sstr_label >> ground_truth;
        // continue if point is unlabeled
        if(ground_truth == 0)
            continue;
        unions[ground_truth]++;
        nb_labeled_pts++;
        std::stringstream sstr(line);
        float x,y,z;
        sstr >> x >> y >> z;

        int x_id = std::floor(x/voxel_size) + 0.5; // + 0.5, centre du voxel (k1*res, k2*res)
        int y_id = std::floor(y/voxel_size) + 0.5;
        int z_id = std::floor(z/voxel_size) + 0.5;

        Eigen::Vector3i vox(x_id, y_id, z_id);
        if (voxels.count(vox)==0) {
            label = 0; // pour l'instant
        }
        else{
            label = voxels[vox].get_label();
        }

        if (label == ground_truth){
            successes[label]++;
            unions[label]++;
        }
        else{
            unions[label]++;
            unions[ground_truth]++;
        }

    }
    std::string perf_filename  = input_sparse_dir + filename + "_perf.txt";
    std::ofstream output(perf_filename.c_str());
    output << "Performances of " + filename << std::endl;
    std::string classes [9] = {"unlabeled", "man-made terrain", "natural terrain", "high vegetation", "low vegetation", "buildings", "hard scape", "scanning artefacts", "cars"};
    int nb_of_successes = 0;
    float sum_IoUs = 0;
    std::vector<float> IoUs(9);
    for (int i=0; i<9; i++){
        IoUs[i] = successes[i]/float(unions[i]);
        sum_IoUs+=IoUs[i];
        std::cout << IoUs[i] << " ";
        output << "IoU of " << classes[i] << IoUs[i] << std::endl;
        nb_of_successes += successes[i];
    }
    output << "global accuracy : " << nb_of_successes/float(nb_labeled_pts) << std::endl;
    output << "IoU averaged on 8 classes : " << sum_IoUs/8. << std::endl;
    std::cout << std::endl << nb_labeled_pts << std::endl;
    return std::pair< int, std::pair< std::vector<int>, std::vector<int> > > (nb_labeled_pts, std::pair<std::vector<int>, std::vector<int> > (successes, unions) );
}

int main (int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "USAGE : " << argv[0] << " path/to/raw/point/clouds  path/to/agregated/point/clouds" << std::endl;
        exit(1);
    }
    float voxel_size = 0.07;
    std::vector<std::string> filenames(15);
    filenames[0] = "bildstein_station1_xyz_intensity_rgb";
    filenames[1] = "bildstein_station3_xyz_intensity_rgb";
    filenames[2] = "bildstein_station5_xyz_intensity_rgb";
    filenames[3] = "domfountain_station1_xyz_intensity_rgb";
    filenames[4] = "domfountain_station2_xyz_intensity_rgb";
    filenames[5] = "domfountain_station3_xyz_intensity_rgb";
    filenames[6] = "neugasse_station1_xyz_intensity_rgb";
    filenames[7] = "sg27_station1_intensity_rgb";
    filenames[8] = "sg27_station2_intensity_rgb";
    filenames[9] = "sg27_station4_intensity_rgb";
    filenames[10] = "sg27_station5_intensity_rgb";
    filenames[11] = "sg27_station9_intensity_rgb";
    filenames[12] = "sg28_station4_intensity_rgb";
    filenames[13] = "untermaederbrunnen_station1_xyz_intensity_rgb";
    filenames[14] = "untermaederbrunnen_station3_xyz_intensity_rgb";
    std::vector<int> unions (9,0);
    std::vector<int> successes (9,0);
    int total_nb_labeled_pts = 0;
    for (int i = /*9*/ 13;i < 15; i++) {
        std::cout << "interpolation for " + filenames[i] << std::endl;
        std::pair< int, std::pair< std::vector<int>, std::vector<int> > > scene_perfs = interpolate_labels_one_point_cloud(argv[1], argv[2], filenames[i], voxel_size);
        for (int j=0; j<9; j++){
            successes[j]+= scene_perfs.second.first[j];
            unions[j]+= scene_perfs.second.second[j];
        }
        total_nb_labeled_pts += scene_perfs.first;
    }

    // now we agregate this data on the point clouds that were processed and we write it into a file

    std::string perf_filename  = std::string(argv[2]) + "global_perf.txt";
    std::ofstream output(perf_filename.c_str());
    std::string classes [9] = {"unlabeled", "man-made terrain", "natural terrain", "high vegetation", "low vegetation", "buildings", "hard scape", "scanning artefacts", "cars"};
    int total_nb_of_successes = 0;
    float sum_IoUs = 0;
    std::vector<float> IoUs(9);
    for (int i=0; i<9; i++){
        IoUs[i] = successes[i]/float(unions[i]);
        sum_IoUs+=IoUs[i];
        std::cout << IoUs[i] << " ";
        output << "IoU of " << classes[i] << IoUs[i] << std::endl;
        total_nb_of_successes += successes[i];
    }
    output << "global accuracy : " << total_nb_of_successes/float(total_nb_labeled_pts) << std::endl;
    output << "IoU averaged on 8 classes : " << sum_IoUs/8. << std::endl;
    std::cout << std::endl << total_nb_labeled_pts << std::endl;
}
