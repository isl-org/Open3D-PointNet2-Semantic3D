/*
 * code inspired by  https://github.com/aboulch/snapnet
 * The IoUs (per class and average) and the accuracy are computed for each scene that is in the folder passed as argument,
 * and then the global IoUs and global accuracy are computed and saved. More information below. 
 */
#include  <iostream>
#include  <fstream>
#include  <string>
#include  <sstream>
#include  <vector>
#include  <map>
#include  <Eigen/Dense>
#include  <algorithm>

class Interpolation_labels_container{
    std::vector<int> label_count;
    int label;
public:
    Interpolation_labels_container(){label_count = std::vector<int>(9,0); label = 0;}
    void add_label(int label){label_count[label]++;}
    void calculate_label(){label = max_element(label_count.begin(), label_count.end()) - label_count.begin();}
    int get_label(){return label;}
    void set_label(int l){label = l;}
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


std::pair<int, std::pair<std::vector<int>, std::vector<int> > > interpolate_labels_one_point_cloud(const std::string& input_dense_dir,
                                                                                                 const std::string& input_sparse_dir,
                                                                                                 const std::string& output_dir,
                                                                                                 const std::string& filename,
                                                                                                 const float& voxel_size,
                                                                                                 const bool& export_labels)
{
    /*
     * The pointnet2 network only takes up to a few thousand points at a time, so we do not have the real results yet
     * But we can get results on a sparser point cloud (after decimation, and after we dynamically sample inputs on
     * the decimated point clouds.
     * The job of this function is to take a very sparse point cloud (a few hundred thousand points) with predictions
     * by the network and to interpolate the results to the much denser raw point clouds.
     * This is achieved by a division of the space into a voxel grid, implemented as a map called voxels.
     * First the sparse point cloud is iterated and the map is constructed. We store for each voxel and each label
     * the nb of points from the sparse cloud and with the right label was in the voxel.
     * Then we assign to each voxel the label which got the most points.
     * And finally we can iterate the dense point cloud and dynamically assign labels according to the voxels.
     * IoU per class and accuracy are calculated at the end.
     */
    std::string filename_sparse =input_sparse_dir + "/" + filename + "_aggregated.txt";
    std::string filename_labels_sparse =input_sparse_dir + "/" + filename + "_pred.txt";
    std::string filename_dense =input_dense_dir + "/" + filename + ".txt";
    std::string filename_labels_dense =input_dense_dir + "/" + filename + ".labels";
    std::ifstream ifs(filename_sparse.c_str());
    if (ifs.fail()) std::cerr << filename_sparse <<" not found" << std::endl;
    std::ifstream ifs_labels(filename_labels_sparse.c_str());
    if (ifs_labels.fail()) std::cerr << filename_labels_sparse << " not found" << std::endl;
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
    // don't know how to open only when necessary
    std::string out_label_filename  = output_dir + "/" + filename + ".labels";
    std::ofstream out_label (out_label_filename.c_str());
    std::ifstream ifs2 (filename_dense.c_str());
    if (ifs2.fail()) std::cerr << filename_dense << " not found" << std::endl;
    std::ifstream ifs_labels2 (filename_labels_dense.c_str());
    bool compute_perfs = (!ifs_labels2.fail());
    std::cout << "labeling raw point cloud";
    if (export_labels) std::cout << " and exporting labels";
    if (compute_perfs) std::cout << " and computing performances";
    std::cout << std::endl;
    if (!compute_perfs) std::cout << filename_labels_dense << " not found, assuming this is testing dataset" << std::endl;
    pt_id = 0;
    int nb_labeled_pts = 0;
    std::vector<int> successes(9, 0);
    std::vector<int> unions(9, 0);
    int holes_nb = 0;
    while(getline(ifs2,line)){
        pt_id++;
        if((pt_id+1)%1000000==0){
            std::cout << (pt_id+1)/1000000 << " M. " << holes_nb << " holes encoutered so far " << std::endl;
        }
        std::stringstream sstr(line);
        float x,y,z;
        int intensity, r, g, b;
        sstr >> x >> y >> z >> intensity >> r >> g >> b;

        int x_id = std::floor(x/voxel_size) + 0.5; // + 0.5, centre du voxel (k1*res, k2*res)
        int y_id = std::floor(y/voxel_size) + 0.5;
        int z_id = std::floor(z/voxel_size) + 0.5;

        int label;
        Eigen::Vector3i vox(x_id, y_id, z_id);
        if (voxels.count(vox)==0) {
            holes_nb++;
            //std::cout << "voxel unlabeled. fetching closest voxel" << std::endl;
            // here no point in the voxel was in the aggregated point cloud
            // we assign it the label 0 for now (TODO : improve by nearest neighbor search using octree ?)
            label = 0;
        }
        else{
            label = voxels[vox].get_label();
        }

        if (export_labels) out_label << label << std::endl;

        if (compute_perfs){
            getline(ifs_labels2,line_labels);
            std::stringstream sstr_label(line_labels);
            int ground_truth;
            sstr_label >> ground_truth;
        // continue if point is unlabeled
            if(ground_truth == 0)
                continue;
            unions[ground_truth]++;
            nb_labeled_pts++;
            if (label == ground_truth){
                successes[label]++;
            }
            else{
                unions[label]++;
            }
        }

    }
    out_label.close();

    std::string perf_filename  = output_dir + "/" + filename + "_perf.txt";
    std::ofstream output(perf_filename.c_str());
    if (compute_perfs) output << "Performances of " + filename << std::endl;
    std::string classes [9] = {"unlabeled", "man-made terrain", "natural terrain", "high vegetation", "low vegetation", "buildings", "hard scape", "scanning artefacts", "cars"};
    int nb_of_successes = 0;
    float sum_IoUs = 0;
    std::vector<float> IoUs(9,0);
    if (compute_perfs){
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
    }
    return std::pair< int, std::pair< std::vector<int>, std::vector<int> > > (nb_labeled_pts, std::pair<std::vector<int>, std::vector<int> > (successes, unions) );
}

int main (int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "USAGE : " << argv[0] << " path/to/raw/point/clouds/  path/to/agregated/point/clouds/  path/for/results/   export_labels" << std::endl;
        exit(1);
    }
    float voxel_size = strtof(argv[4], NULL);
    std::vector<std::string> PossibleFileNames(30);
    PossibleFileNames[0] = "bildstein_station1_xyz_intensity_rgb";
    PossibleFileNames[1] = "bildstein_station3_xyz_intensity_rgb";
    PossibleFileNames[2] = "bildstein_station5_xyz_intensity_rgb";
    PossibleFileNames[3] = "domfountain_station1_xyz_intensity_rgb";
    PossibleFileNames[4] = "domfountain_station2_xyz_intensity_rgb";
    PossibleFileNames[5] = "domfountain_station3_xyz_intensity_rgb";
    PossibleFileNames[6] = "neugasse_station1_xyz_intensity_rgb";
    PossibleFileNames[7] = "sg27_station1_intensity_rgb";
    PossibleFileNames[8] = "sg27_station2_intensity_rgb";
    PossibleFileNames[9] = "sg27_station4_intensity_rgb";
    PossibleFileNames[10] = "sg27_station5_intensity_rgb";
    PossibleFileNames[11] = "sg27_station9_intensity_rgb";
    PossibleFileNames[12] = "sg28_station4_intensity_rgb";
    PossibleFileNames[13] = "untermaederbrunnen_station1_xyz_intensity_rgb";
    PossibleFileNames[14] = "untermaederbrunnen_station3_xyz_intensity_rgb";
    PossibleFileNames[15] = "birdfountain_station1_xyz_intensity_rgb";
    PossibleFileNames[16] = "castleblatten_station1_intensity_rgb";
    PossibleFileNames[17] = "castleblatten_station5_xyz_intensity_rgb";
    PossibleFileNames[18] = "marketplacefeldkirch_station1_intensity_rgb";
    PossibleFileNames[19] = "marketplacefeldkirch_station4_intensity_rgb";
    PossibleFileNames[20] = "marketplacefeldkirch_station7_intensity_rgb";
    PossibleFileNames[21] = "sg27_station10_intensity_rgb";
    PossibleFileNames[22] = "sg27_station3_intensity_rgb";
    PossibleFileNames[23] = "sg27_station6_intensity_rgb";
    PossibleFileNames[24] = "sg27_station8_intensity_rgb";
    PossibleFileNames[25] = "sg28_station2_intensity_rgb";
    PossibleFileNames[26] = "sg28_station5_xyz_intensity_rgb";
    PossibleFileNames[27] = "stgallencathedral_station1_intensity_rgb";
    PossibleFileNames[28] = "stgallencathedral_station3_intensity_rgb";
    PossibleFileNames[29] = "stgallencathedral_station6_intensity_rgb";
    // we try to open the files one by one in order to know which ones are present in the folder
    std::vector<std::string> fileNames;
    for (unsigned int i=0;i<PossibleFileNames.size(); i++) {
        std::string filename_labels_sparse =std::string(argv[2]) + "/" + PossibleFileNames[i] + "_pred.txt";
        std::ifstream ifs(filename_labels_sparse.c_str());
        if (!ifs.fail()) {
            fileNames.push_back(PossibleFileNames[i]);
            std::cout << "Found " + PossibleFileNames[i] << std::endl;
        }
        ifs.close();
    }
    std::vector<int> unions (9,0);
    std::vector<int> successes (9,0);
    int total_nb_labeled_pts = 0;
    for (unsigned int i=0;i < fileNames.size(); i++) {
        std::cout << "interpolation for " + fileNames[i] << std::endl;
        std::pair< int, std::pair< std::vector<int>, std::vector<int> > > scene_perfs
                = interpolate_labels_one_point_cloud(argv[1], argv[2], argv[3], fileNames[i], voxel_size, (std::string(argv[5])=="1"));
        for (int j=0; j<9; j++){
            successes[j]+= scene_perfs.second.first[j];
            unions[j]+= scene_perfs.second.second[j];
        }
        total_nb_labeled_pts += scene_perfs.first;
    }

    // now we agregate this data on the point clouds that were processed and we write it into a file

    if (fileNames.size()==0){
        std::cout << "no file found" << std::endl;
        return 0;
    }
    if (total_nb_labeled_pts !=0){
        std::string perf_filename  = std::string(argv[3]) + "/global_perf.txt";
        std::ofstream output((perf_filename).c_str());
        output << "Global performances on files ";
        for (unsigned int i=0; i<fileNames.size();i++)
            output << fileNames[i] << " ";
        output << std::endl;
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
}
