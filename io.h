#ifndef IO_H
#define IO_H

#include <opencv2/core.hpp>
#include <map>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "Geometries.h"

namespace io {
    /// Read geometry from xml file
    GeometryModel* readGeometry(std::string url);

    /// Write geometry to xml file 
    bool writeGeometry(std::string url, GeometryModel& geometry);

    ///  Read 3D object points with ID seperated by space from text file with the type "x y z id"
    std::map<int, std::vector<double>> readObjectPoints(const std::string& path);

    ///  Write 3D object points with ID 
    bool writeObjectPoints(const std::string& path, std::map<int, std::vector<double>>& points);

} // end namespace io

#endif // IO_H

