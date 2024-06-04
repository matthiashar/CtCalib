#ifndef GEOMETRIES_H
#define GEOMETRIES_H

#include "GeometryRTK.h"
#include "GeometryDetector.h"
#include "GeometryTilt.h"

namespace Geometries {

    std::vector<GeometryModel*> listGeometries();

    GeometryModel* geometryByName(std::string geometryName, double SDD = 0,
        double SRD = 0,	cv::Size2i imageSize = cv::Size2i(0, 0), double pixelSize = 1.0);

} // end namespace 

#endif // GEOMETRIES_H

