#include "GeometryTilt.h"
#include <iostream>


GeometryTilt::GeometryTilt(double SDD, double SRD, cv::Size2i imageSize, double pixelSize)
{
	m_image_size_px = imageSize;
	m_pixel_size_mm = pixelSize;
	m_transformation = Eigen::Matrix<double, 6, 1>::Zero();
	m_transformation_adjust = CONST_PARAM;

	m_parameter = {
	GeometryParameter("SDD", SDD, AdjustEnumOptions::ADJUST_PARAM),
		GeometryParameter("SRD", SRD, AdjustEnumOptions::CONST_PARAM),
		GeometryParameter("InPlaneAngle", 0, AdjustEnumOptions::ADJUST_PARAM),
		GeometryParameter("ProjectionOffsetX", 0, AdjustEnumOptions::ADJUST_PARAM),
		GeometryParameter("ProjectionOffsetY", 0, AdjustEnumOptions::ADJUST_PARAM),
		GeometryParameter("TauX", 0, AdjustEnumOptions::ADJUST_PARAM),
		GeometryParameter("TauY", 0, AdjustEnumOptions::ADJUST_PARAM),
		GeometryParameter("GantryAngleStep", 0, AdjustEnumOptions::ADJUST_PARAM)
	};
}


Eigen::Matrix<double, 3, 4> GeometryTilt::getProjectionMatrix(double gantry_angle)
{
	std::map<std::string, GeometryParameter*> _plist = getParameterMap();

	Eigen::Matrix<double, 3, 4> projMatrix = projectionMatrix<double>(
		&gantry_angle,
		&_plist["InPlaneAngle"]->m_value,
		&_plist["ProjectionOffsetX"]->m_value,
		&_plist["ProjectionOffsetY"]->m_value,
		&_plist["SDD"]->m_value, 
		&_plist["SRD"]->m_value);

	return projMatrix;
}

std::map<int, cv::Point2d> GeometryTilt::projectPoints(std::map<int, std::vector<double>>& object_points, double gantry_angle, bool in_pixel)
{
	std::map<std::string, GeometryParameter*> _plist = getParameterMap();

	std::map<int, cv::Point2d > proj_points;
	for (auto const& oPoint : object_points) {

		Eigen::Vector<double, 2> p = projectPoint<double>(
			&oPoint.second[0],
			&m_transformation[0],
			&gantry_angle,
			&_plist["InPlaneAngle"]->m_value,
			&_plist["ProjectionOffsetX"]->m_value,
			&_plist["ProjectionOffsetY"]->m_value,
			&_plist["TauX"]->m_value,
			&_plist["TauY"]->m_value,
			&_plist["SDD"]->m_value, 
			&_plist["SRD"]->m_value);

		if (in_pixel) {
			// convert to pixel
			proj_points[oPoint.first] = metric2pixel(p[0], p[1]);
		}
		else
		{
			proj_points[oPoint.first] = cv::Point2d(p[0], p[1]);
		}
	}

	return proj_points;
}


ceres::ResidualBlockId GeometryTilt::addResidualBlock(
	ceres::Problem& problem,
	ceres::LossFunction* loss_function,
	std::vector<double>& object_point,
	double& gantry,
	double& x,
	double& y)
{
	// Create map of parameter
	std::map<std::string, GeometryParameter*> _plist = getParameterMap();

	// create cost function
	ceres::CostFunction* cost_function = reprojectionError::Create(x, y);

	return problem.AddResidualBlock(cost_function,
		loss_function,
		&object_point[0],
		&m_transformation[0],
		&gantry, // index
		&_plist["GantryAngleStep"]->m_value,
		&_plist["InPlaneAngle"]->m_value,
		&_plist["ProjectionOffsetX"]->m_value,
		&_plist["ProjectionOffsetY"]->m_value,
		&_plist["TauX"]->m_value,
		&_plist["TauY"]->m_value,
		&_plist["SDD"]->m_value,
		&_plist["SRD"]->m_value);
}
