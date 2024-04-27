#include <3DHoPD/3DHoPD.h>
#include<string>
#include <iostream>
#include <filesystem>
#include <vector>
#include<omp.h>
#include <pcl/gpu/features/features.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/containers/kernel_containers.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/kinfu_large_scale/impl/raycaster.h>
#include <pcl/gpu/kinfu_large_scale/impl/screenshot_manager.hpp>
#include <pcl/gpu/kinfu_large_scale/kinfu.h>
#include <pcl/gpu/kinfu_large_scale/raycaster.h>
#include <pcl/gpu/kinfu_large_scale/tsdf_volume.h>
#include <pcl/gpu/kinfu_large_scale/color_volume.h>
#include <pcl/gpu/kinfu_large_scale/marching_cubes.h>

using namespace std;
namespace fs = boost::filesystem;

vector<vector<double>> splitString(string str) {

    std::stringstream ss(str);

    vector<string> words;
    string word;

    // Read each word separated by spaces and store them in the vector
    while (ss >> word) {
        string temp = word;
        words.push_back(temp);
    }
    vector<vector<double>> res(4, vector<double>(4, 0));

    int i = 0;

    for (int a = 0; a < 4; a++) {
        for (int b = 0; b < 4; b++) {
            double temp = stod(words[i++]);
            res[a][b] = temp;
        }
    }
    return res;
}

double frobeniusNorm(vector<vector<double>> matrix) {
    double norm = 0.0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            norm += matrix[i][j] * matrix[i][j];
        }
    }
    return sqrt(norm);
}

double compareMatrices(vector<vector<double>> matrix1, vector<vector<double>> matrix2) {
    // Calculate the Frobenius norm of the difference between the matrices
    vector<vector<double>> difference(4, vector<double>(4, 0));

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            difference[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }

    double norm = frobeniusNorm(difference);
    return norm;
}

#define MAXBUFSIZE  1000000



Eigen::Matrix4f readMatrixFromFile(const std::string& file_path) {
    Eigen::Matrix4f matrix;
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << file_path << std::endl;
        return matrix;
    }

    std::string line;
    if (!std::getline(file, line)) {
        std::cerr << "Error: Unexpected end of file." << std::endl;
        return matrix;
    }

    std::istringstream iss(line);
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            float value;
            if (!(iss >> value)) {
                std::cerr << "Error: Invalid data in line." << std::endl;
                return matrix;
            }
            matrix(row, col) = value;
        }
    }

    // Check if there are more values in the line
    std::string extra;
    if (iss >> extra) {
        std::cerr << "Error: Extra data in line." << std::endl;
        return matrix;
    }

    file.close();
    return matrix;
}





class Challenge {
public:
    string name = "Challenge";
    vector<string> scenes;
    vector<string> objects;
    vector<string> check;

    /*void populateScenes(const string& folderPath) {
        for (const auto& entry : fs::recursive_directory_iterator(folderPath)) {
            if (fs::is_regular_file(entry) && entry.path().extension() == ".pcd") {
                scenes.push_back(entry.path().string());
            }
        }
    }
    void populateCheck(const string& folderPath) {
        for (const auto& entry : fs::recursive_directory_iterator(folderPath)) {
            if (fs::is_regular_file(entry) && entry.path().extension() == ".txt") {
                check.push_back(entry.path().string());
            }
        }
    }

    void populateObjects(const std::string& folderPath) {
        for (const auto& entry : fs::directory_iterator(folderPath)) {
            if (fs::is_regular_file(entry) && entry.path().extension() == ".pcd") {
                objects.push_back(entry.path().string());
            }
        }
    }*/

    void populateScenes(const string& folderPath) {
        try {
            for (const auto& entry : fs::recursive_directory_iterator(folderPath)) {
                if (fs::is_regular_file(entry) && entry.path().extension() == ".pcd") {
                    scenes.push_back(entry.path().string());
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error while populating scenes: " << e.what() << std::endl;
        }
    }

    void populateCheck(const string& folderPath) {
        try {
            for (const auto& entry : fs::recursive_directory_iterator(folderPath)) {
                if (fs::is_regular_file(entry) && entry.path().extension() == ".txt") {
                    check.push_back(entry.path().string());
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error while populating check: " << e.what() << std::endl;
        }
    }

    void populateObjects(const std::string& folderPath) {
        try {
            for (const auto& entry : fs::directory_iterator(folderPath)) {
                if (fs::is_regular_file(entry) && entry.path().extension() == ".pcd") {
                    objects.push_back(entry.path().string());
                }
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error while populating objects: " << e.what() << std::endl;
        }
    }
};


void extractKeypointsAndDescriptorsGPU(pcl::PointCloud<pcl::PointXYZ>& cloud, double ds) {
    pcl::gpu::FPFHEstimationGPU<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_gpu;
    pcl::gpu::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne_gpu;

    // Estimate normals on the device
    pcl::gpu::DeviceArray<pcl::PointXYZ> input_gpu;
    input_gpu.upload(cloud.points);
    ne_gpu.setInputCloud(input_gpu);
    pcl::gpu::DeviceArray<pcl::Normal> normals_gpu;
    ne_gpu.compute(normals_gpu);

    // FPFH estimation
    fpfh_gpu.setInputCloud(input_gpu);
    fpfh_gpu.setInputNormals(normals_gpu);
    fpfh_gpu.setRadiusSearch(ds); // Set the radius for the search sphere
    pcl::gpu::DeviceArray<pcl::FPFHSignature33> features_gpu;
    fpfh_gpu.compute(features_gpu);
}



int main(int argc, char* argv[]) {
 

    //cout << "C:\\BTP!\\3DHoPD_willow\\3DHoPD\\willow_and_challenge\\willow_ground_truth\\" + std::string(argv[2]) + "_willow_dataset" << endl;
    //cout << "C:\\BTP!\\3DHoPD_willow\\3DHoPD\\willow_and_challenge\\willow_scenes\\" + std::string(argv[2]) + "_willow_dataset" << endl;
    cout << "C:\\BTP!\\try_data1\\" + std::string(argv[1]) + "\\check" << endl;
    Challenge obj;
    obj.populateCheck("C:\\BTP!\\try_data1\\"+ std::string(argv[1]) +"\\check");
    obj.populateObjects("C:\\BTP!\\try_data1\\"+std::string(argv[1])+"\\models");
    obj.populateScenes("C:\\BTP!\\try_data1\\"+std::string(argv[1])+"\\scenes");


    int scenesCount = obj.scenes.size();
    int objectCount = obj.objects.size();
    int checkCount = obj.check.size();

    cout << "\n\nObject = " << obj.name << endl;
    cout << "Total scenes = " << obj.scenes.size() << endl;
    cout << "Total objects = " << obj.objects.size() << endl;
    cout << "Total check = " << obj.check.size() << endl;

    /*  vector<string> filter_objects;


      for (int suffix : suffixNumbers) {

          for (const string& filePath : obj.objects) {

              size_t found = filePath.find_last_of("/\\");
              string fileName = filePath.substr(found + 1);
              size_t pos = fileName.find_last_of(".");
              int extractedSuffix = stoi(fileName.substr(7, pos - 7));
              if (extractedSuffix == suffix) {
                  cout << filePath << endl;
                  filter_objects.push_back(filePath);
              }
          }
      }
      */

 



    //int main
    ofstream ToFile;
    ToFile.open("C:\\BTP!\\3DHoPD_willow\\3DHoPD\\results\\test_final_uncluttered.txt", ios::out | ios::app);


    /*****************************************/


    cout << "hello" << endl;

    Eigen::Matrix4f T;
    vector<float> mario_score;
    int coin = 0;

    for (double ds = 0.6; ds > 0.59; ds -= 0.01) {

        double accSum = 0;
        double timeSum = 0;

        for (int j = 0; j < obj.objects.size(); j++) {
            //cout << "filtered_object " << filter_objects[j] << endl;

            int match = 0;
            double tempTime = 0;

            for (int i = 0; i < scenesCount; i++) {
                //cout << "Scene " << obj.scenes[i] << endl;

                for (int l = 0; l < obj.check.size(); l++) {

                    T = readMatrixFromFile(obj.check[l].c_str());
                    //cout << "Truth " << obj.check[l] << endl;

                    pcl::PointCloud<pcl::PointXYZ> cloud2, cloud1;
                    pcl::io::loadPCDFile<pcl::PointXYZ>(obj.scenes[i], cloud2);
                    pcl::io::loadPCDFile<pcl::PointXYZ>(obj.objects[j], cloud1);

                    threeDHoPD RP1, RP2;


                    // Using Simple Uniform Keypoint Detection, instead ISS keypoints can also be used!

                    RP1.cloud = cloud1; // Model
                    RP1.detect_uniform_keypoints_on_cloud(0.01);

                    RP2.cloud = cloud2; // Scene
                    RP2.detect_uniform_keypoints_on_cloud(0.02);
                    clock_t start1, end1;
                    double cpu_time_used1;
                    start1 = clock();


                    // setup
                    RP1.kdtree.setInputCloud(RP1.cloud.makeShared());// 
                    RP2.kdtree.setInputCloud(RP2.cloud.makeShared());// 

                    //RP1.JUST_REFERENCE_FRAME_descriptors(ds); // this is where descriptors are being built
                    //RP2.JUST_REFERENCE_FRAME_descriptors(ds);
                    
                    
                    //GPU
                    extractKeypointsAndDescriptorsGPU(cloud1, ds);
                    extractKeypointsAndDescriptorsGPU(cloud2, ds);

                    end1 = clock();
                    cpu_time_used1 = ((double)(end1 - start1)) / CLOCKS_PER_SEC;
                    tempTime += (double)cpu_time_used1;

                    pcl::Correspondences corrs;

                    clock_t start_shot2, end_shot2;
                    double cpu_time_used_shot2;
                    start_shot2 = clock();


                    // KD Tree of scene
                    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_LRF;
                    pcl::PointCloud<pcl::PointXYZ> pcd_LRF;
                    for (int i = 0; i < RP2.cloud_LRF_descriptors.size(); i++)
                    {
                        pcl::PointXYZ point;
                        point.x = RP2.cloud_LRF_descriptors[i].vector[0];
                        point.y = RP2.cloud_LRF_descriptors[i].vector[1];
                        point.z = RP2.cloud_LRF_descriptors[i].vector[2];

                        pcd_LRF.push_back(point);
                    }




                    // KD Tree of Model
                    kdtree_LRF.setInputCloud(pcd_LRF.makeShared());

                    for (int i = 0; i < RP1.cloud_LRF_descriptors.size(); i++)
                    {
                        pcl::PointXYZ searchPoint;
                        searchPoint.x = RP1.cloud_LRF_descriptors[i].vector[0];
                        searchPoint.y = RP1.cloud_LRF_descriptors[i].vector[1];
                        searchPoint.z = RP1.cloud_LRF_descriptors[i].vector[2];

                        std::vector<int> nn_indices;
                        std::vector<float> nn_sqr_distances;

                        std::vector<double> angles_vector;

                        float threshold_to_remove_false_positives = 0.0075; // Changed, init = 0.0075
                        try
                        {


                            if (kdtree_LRF.radiusSearch(searchPoint, threshold_to_remove_false_positives, nn_indices, nn_sqr_distances) > 0)// Based on this threshold, false positives are removed!
                            {


                                for (int j = 0; j < nn_indices.size(); j++)
                                {
                                    try
                                    {
                                        //cout << "e1" << endl;
                                        Eigen::VectorXf vec1, vec2;
                                        vec1.resize(24); vec2.resize(24); // desc size = 50 new


                                        // This fop loop is in error condition

                                        for (int k = 0; k < 24; k++)// then nearest neighbour based matching with HoPD
                                        {
                                            //  cout << "e2" << endl;
                                            vec1[k] = RP1.cloud_distance_histogram_descriptors[i].vector[k];
                                            vec2[k] = RP2.cloud_distance_histogram_descriptors[nn_indices[j]].vector[k];
                                            //   cout<<vec1[k]<<endl;

                                        }
                                        //  cout << "e3" << endl;
                                          //cout<<"aaaaa"<<endl;

                                        double dist = (vec1 - vec2).norm();
                                        angles_vector.push_back(dist);
                                        //  cout << "e4" << endl;
                                    }
                                    catch (const std::exception& e) {
                                        std::cerr << "Exception occurred: " << e.what() << std::endl;
                                        continue;
                                    }


                                }
                                // cout << "e6"<< endl;


                                std::vector<double>::iterator result;
                                result = std::min_element(angles_vector.begin(), angles_vector.end());

                                int min_element_index = std::distance(angles_vector.begin(), result);

                                pcl::Correspondence corr;
                                corr.index_query = RP1.patch_descriptor_indices[i];
                                corr.index_match = RP2.patch_descriptor_indices[nn_indices[min_element_index]];

                                corrs.push_back(corr);

                            }
                        }
                        catch (const std::exception& e) {
                            std::cerr << "Exception occurred: " << e.what() << std::endl;
                            continue;
                        }
                    }


                    end_shot2 = clock();
                    cpu_time_used_shot2 = ((double)(end_shot2 - start_shot2)) / CLOCKS_PER_SEC;
                    //  cout << "e7"<<endl;

                      // RANSAC based false matches removal
                    pcl::CorrespondencesConstPtr corrs_const_ptr = boost::make_shared< pcl::Correspondences >(corrs);

                    pcl::Correspondences corr_shot;
                    pcl::registration::CorrespondenceRejectorSampleConsensus< pcl::PointXYZ > Ransac_based_Rejection_shot;
                    Ransac_based_Rejection_shot.setInputSource(RP1.cloud_keypoints.makeShared());
                    Ransac_based_Rejection_shot.setInputTarget(RP2.cloud_keypoints.makeShared());
                    Ransac_based_Rejection_shot.setInlierThreshold(0.02); // Changed, init = 0.02
                    Ransac_based_Rejection_shot.setInputCorrespondences(corrs_const_ptr);
                    Ransac_based_Rejection_shot.getCorrespondences(corr_shot);
                    // cout << "e6" << endl;

                     // changes
                    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
                    kdtree.setInputCloud(RP2.cloud_keypoints.makeShared());
                    pcl::PointXYZ searchPoint;
                    int actual_keypoints = 0;

                    for (int i = 0; i < RP1.cloud_keypoints.size(); i++)
                    {

                        Eigen::Vector4f e_point1(RP1.cloud_keypoints[i].getVector4fMap());
                        Eigen::Vector4f transformed_point(T * e_point1);
                        searchPoint.x = transformed_point[0];
                        searchPoint.y = transformed_point[1];
                        searchPoint.z = transformed_point[2];

                        int K = 1;
                        std::vector<int> pointIdxNKNSearch(K);
                        std::vector<float> pointNKNSquaredDistance(K);

                        if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
                        {
                            float threshold = 0.7;
                            if (threshold > (float)sqrt(pointNKNSquaredDistance[0]))
                            {
                                actual_keypoints++;
                            }
                        }


                    }

                    // cout << "e7" << endl;

                     // To count the number of RANSAC Matches that are inline with the groundtruth correspondences
                    int cnt = 0;

                    for (int i = 0; i < (int)corr_shot.size(); i++)
                    {
                        pcl::PointXYZ point1 = RP1.cloud_keypoints[corr_shot[i].index_query];
                        pcl::PointXYZ point2 = RP2.cloud_keypoints[corr_shot[i].index_match];

                        Eigen::Vector4f e_point1(point1.getVector4fMap());
                        Eigen::Vector4f e_point2(point2.getVector4fMap());

                        Eigen::Vector4f transformed_point(T * e_point1);
                        Eigen::Vector4f diff(e_point2 - transformed_point);


                        if (diff.norm() < 1.4)

                            cnt++;
                    }
                    //cout << "matched count: " << cnt << endl;
                    if (actual_keypoints != 0) {
                        cout << "RRR of 3DLRF * : " << ((float)cnt / (float)actual_keypoints) * 100 << endl;
                        cout << coin << endl;
                        coin++;


                        mario_score.push_back(((float)cnt / (float)actual_keypoints) * 100);

                    }



                }


            }


        }



    }
    float mario_sum = accumulate(mario_score.begin(), mario_score.end(), 0);
    float marioavg = mario_sum / mario_score.size();
    cout << "Final Average: " << marioavg << endl;
    //ToFile << "Folder: " << argv[2];

    ToFile << "\t" << marioavg << "\n \n";
    cout << "done" << endl;




    // Additional processing...

    return 0;
}
