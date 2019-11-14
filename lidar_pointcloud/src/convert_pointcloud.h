/*
    Created by Kavan Sikand 11/19
*/
#include <string>

#ifndef SRC_CONVERT_POINTCLOUD_H_
#define SRC_CONVERT_POINTCLOUD_H_

void ExtractDataFromBagFile(const char* bag_path, const char* lidar_topic, const char* localization_topic, std::string dataset_name);
#endif