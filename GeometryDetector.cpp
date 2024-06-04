#include "GeometryDetector.h"
#include <iostream>


GeometryDetector::GeometryDetector(double SDD, double SRD, cv::Size2i imageSize, double pixelSize)
{
	m_image_size_px = imageSize;
	m_pixel_size_mm = pixelSize;
	m_transformation = Eigen::Matrix<double, 6, 1>::Zero();
	m_transformation_adjust = CONST_PARAM;

	m_parameter = {
	GeometryParameter("SDD", SDD, AdjustEnumOptions::ADJUST_PARAM),
		GeometryParameter("SRD", SRD, AdjustEnumOptions::CONST_PARAM),
		GeometryParameter("OutOfPlaneAngle", 0, AdjustEnumOptions::ADJUST_PARAM),
		GeometryParameter("InPlaneAngle", 0, AdjustEnumOptions::ADJUST_PARAM),
		GeometryParameter("ProjectionOffsetX", 0, AdjustEnumOptions::ADJUST_PARAM),
		GeometryParameter("ProjectionOffsetY", 0, AdjustEnumOptions::ADJUST_PARAM),
		GeometryParameter("RotationOffsetX", 0, AdjustEnumOptions::ADJUST_PARAM),
		GeometryParameter("GantryAngleStep", 0, AdjustEnumOptions::ADJUST_PARAM)
	};
}


Eigen::Matrix<double, 3, 4> GeometryDetector::getProjectionMatrix(double gantry_angle)
{
	// Create map of parameter
	std::map<std::string, GeometryParameter*> _plist = getParameterMap();

	Eigen::Matrix<double, 3, 4> projMatrix = projectionMatrix<double>(
		&gantry_angle,
		&_plist["OutOfPlaneAngle"]->m_value,
		&_plist["InPlaneAngle"]->m_value,
		&_plist["ProjectionOffsetX"]->m_value,
		&_plist["ProjectionOffsetY"]->m_value,
		&_plist["RotationOffsetX"]->m_value,
		&_plist["SDD"]->m_value,
		&_plist["SRD"]->m_value);
	return projMatrix;
}


ceres::ResidualBlockId GeometryDetector::addResidualBlock(
	ceres::Problem& problem,
	ceres::LossFunction* loss_function,
	std::vector<double>& object_point,
	double& gantry,
	double& x,
	double& y)
{
	// Create map of parameter
	std::map<std::string, GeometryParameter*> _plist = getParameterMap();

	// Create cost function
	ceres::CostFunction* cost_function = reprojectionError::Create(x, y);

	// Add residual block
	return problem.AddResidualBlock(cost_function,
		loss_function,
		&object_point[0],
		&m_transformation[0],
		&gantry, // index
		&_plist["GantryAngleStep"]->m_value,
		&_plist["OutOfPlaneAngle"]->m_value,
		&_plist["InPlaneAngle"]->m_value,
		&_plist["ProjectionOffsetX"]->m_value,
		&_plist["ProjectionOffsetY"]->m_value,
		&_plist["RotationOffsetX"]->m_value,
		&_plist["SDD"]->m_value,
		&_plist["SRD"]->m_value);
}
