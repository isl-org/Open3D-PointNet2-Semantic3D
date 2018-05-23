/*
 *code closely inspired by  https://github.com/aboulch/snapnet
 */
 #include  <iostream>
 #include  <fstream>
 #include  <string>
 #include  <sstream>
 #include  <vector>
 #include  <map>
 #include  <Eigen/Dense>
 #include  <algorithm>
 #include  <pcl/point_types.h>
 #include  <pcl/point_cloud.h>
 #include  <pcl/common/pca.h>

typedef pcl::PointXYZRGBNormal Point;
typedef pcl::PointCloud<Point> PointCloud;
typedef PointCloud::Ptr PointCloudPtr;
typedef unsigned char Byte;

// class for voxels
class Voxel_center{
public:
    float x,y,z;
    int r,g,b;
    int label;
};

// class for sample of n points in voxel
// We don't stay with n points all the way through :
// if the surface is flat, we drop all but one of them
class Sample_points_container{
public:
    unsigned int n, i;
    std::vector<Voxel_center> points;
    Sample_points_container(){
        n                 = 10;
        i                 = 0;
        points            = std::vector<Voxel_center>(n);
    }
    void insert_if_room(Voxel_center vc);
    void resize();
    int size(){return points.size();}
    std::vector<Voxel_center>::iterator begin(){return points.begin();}
    std::vector<Voxel_center>::iterator end (){return points.end();}
};

// We fill the container with (n=10) at most and we don't accept points that are too close together
void Sample_points_container::insert_if_room(Voxel_center vc) {
    if (i < n){
        float dmin = 1e7;
        for (int j=0;j<i;j++){
            float d = (vc.x - points[j].x)*(vc.x - points[j].x)
                      + (vc.y - points[j].y)*(vc.y - points[j].y)
                      + (vc.z - points[j].z)*(vc.z - points[j].z);
            if (d < dmin) dmin = d;
        }
        if (dmin > 0.001){
            points[i] = vc;
            i++;
        }
    }
    if (i==n){
        resize(); // resizing in order to reduce memory allocation
        i++; // in order to go quickly though this function all the next times
    }
}

// Flatness is calculated with pca on the points in the container.
void Sample_points_container::resize() {
    if (i < 3){
        points.resize(i);
        return;
    }
    PointCloudPtr smallpc(new PointCloud);
    smallpc->width    = n;
    smallpc->height   = 1;
    smallpc->is_dense = false;
    smallpc->points.resize (smallpc->width * smallpc->height);
    pcl::PCA<Point> pca(*smallpc);
    Eigen::Vector3f eigenvalues = pca.getEigenValues();
    if (eigenvalues(2) > 0.00001)
        points.resize(4);
    else
        points.resize(1);
}

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


void adaptive_sampling(const std::string& raw_dir, const std::string& out_dir, std::basic_string<char> & filename, float voxel_size){
    std::cout << "Processing " << filename << std::endl;
    std::string data_filename   = raw_dir + filename + ".txt";
    std::string labels_filename = raw_dir + filename + ".labels";
    std::string output_filename = out_dir + filename + "_all.txt";
    std::ifstream ifs(data_filename.c_str());
    if (ifs.fail()) {
        std::cout << "filename for raw point cloud data not found" << std::endl;
        return;
    }
    std::ifstream ifs_labels(labels_filename.c_str());
    bool no_labels = ifs_labels.fail();
    if (no_labels) {
        std::cout << "filename for raw point cloud labels not found; assuming this is part of the testing set" << std::endl;
    }
    std::string line;
    std::string line_labels;
    int pt_id =0;

    std::map<Eigen::Vector3i, Sample_points_container, Vector3icomp> voxels;
    while(getline(ifs,line)){
        pt_id++;
        if((pt_id+1)%1000000==0){
            std::cout << (pt_id+1)/1000000 << " M" << std::endl;
        }

        int label = 0;
        if (!no_labels) {
            getline(ifs_labels, line_labels);
            std::stringstream sstr_label(line_labels);
            sstr_label >> label;

            // continue if points is unlabeled
            if (label == 0)
                continue;
        }

        std::stringstream sstr(line);
        float x,y,z;
        int intensity;
        int r, g, b;
        sstr >> x >> y >> z >> intensity >> r >> g >> b;

        int x_id = std::floor(x/voxel_size) + 0.5; // + 0.5, centre du voxel (k1*res, k2*res)
        int y_id = std::floor(y/voxel_size) + 0.5;
        int z_id = std::floor(z/voxel_size) + 0.5;

        Eigen::Vector3i vox(x_id, y_id, z_id);

        if(voxels.count(vox)>0){
            Voxel_center vc;
            vc.x = std::floor(x/voxel_size)*voxel_size;
            vc.y = std::floor(y/voxel_size)*voxel_size;
            vc.z = std::floor(z/voxel_size)*voxel_size;
            vc.r = r;
            vc.g = g;
            vc.b = b;
            vc.label = label;
            voxels[vox].insert_if_room(vc);

        }else{
            Sample_points_container container;
            Voxel_center vc;
            vc.x = std::floor(x/voxel_size)*voxel_size;
            vc.y = std::floor(y/voxel_size)*voxel_size;
            vc.z = std::floor(z/voxel_size)*voxel_size;
            vc.r = r;
            vc.g = g;
            vc.b = b;
            vc.label = label;
            container.insert_if_room(vc);
            voxels[vox] = container;
        }
    }

    // resizing point containers
    pt_id = 0;
    for(std::map<Eigen::Vector3i, Sample_points_container>::iterator it=voxels.begin(); it != voxels.end(); it++){
        it->second.resize();
        pt_id++;
    }
    std::cout << "exporting result of decimation" << std::endl;

    std::vector<Eigen::Vector3i> cols;
    cols.push_back(Eigen::Vector3i(0,0,0));
    cols.push_back(Eigen::Vector3i(192,192,192));
    cols.push_back(Eigen::Vector3i(0,255,0));
    cols.push_back(Eigen::Vector3i(38,214,64));
    cols.push_back(Eigen::Vector3i(247,247,0));
    cols.push_back(Eigen::Vector3i(255,3,0));
    cols.push_back(Eigen::Vector3i(122,0,255));
    cols.push_back(Eigen::Vector3i(0,255,255));
    cols.push_back(Eigen::Vector3i(255,110,206));
    std::ofstream output(output_filename.c_str());
    for(std::map<Eigen::Vector3i, Sample_points_container>::iterator it=voxels.begin(); it != voxels.end(); it++){
        Sample_points_container spc = it->second;
        for (std::vector<Voxel_center>::iterator it2 = spc.begin(); it2 != spc.end(); it2++) {
            output << it2->x << " " << it2->y << " " << it2->z << " " //
                   << it2->r << " " << it2->g << " " << it2->b;
            if (!no_labels) {
                output << " " << cols[it2->label][0] << " " << cols[it2->label][1] << " " << cols[it2->label][2] << " " //
                                   << it2->label;
            }
            output << std::endl;
        }
    }
    ifs.close();
}



int main (int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "USAGE : " << argv[0] << " path/to/raw/point/clouds  where/to/write/sampled/point/clouds  voxel_size" << std::endl;
        exit(1);
    }
    float voxel_size = strtof(argv[3], NULL);
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
        std::string filename_labels_sparse =std::string(argv[1]) + "/" + PossibleFileNames[i] + ".txt";
        std::ifstream ifs(filename_labels_sparse.c_str());
        if (!ifs.fail()) {
            fileNames.push_back(PossibleFileNames[i]);
            std::cout << "Found " + PossibleFileNames[i] << std::endl;
        }
        ifs.close();
    }
    for (int i = 0; i < fileNames.size(); i++) {
        std::cout << "adaptive sampling for " + fileNames[i] << std::endl;
        adaptive_sampling(argv[1], argv[2], fileNames[i], voxel_size);
    }
    if (fileNames.size()==0) std::cout << "not a single file was found in folder " + std::string(argv[1]) + "/" << std::endl;
}
