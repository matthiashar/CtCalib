#include "Geometries.h"

std::vector<GeometryModel*> Geometries::listGeometries()
{
	return std::vector<GeometryModel*> {
		new GeometryRTK(),
			new GeometryDetector(),
			new GeometryTilt()};
}

GeometryModel* Geometries::geometryByName(std::string geometryName, double SDD,
	double SRD, cv::Size2i imageSize, double pixelSize)
{
	if (geometryName == GeometryRTK().getName())
		return new GeometryRTK(SDD,SRD,imageSize,pixelSize);

	if (geometryName == GeometryDetector().getName())
		return new GeometryDetector(SDD,SRD,imageSize,pixelSize);

	if (geometryName == GeometryTilt().getName())
		return new GeometryTilt(SDD,SRD,imageSize,pixelSize);

	return nullptr;
}
