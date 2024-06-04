#include "GeometryModel.h"

GeometryParameter::GeometryParameter()
{
	m_name = "Empty";
	m_value = 0;
	m_adjust_option = ADJUST_PARAM;
}

GeometryParameter::GeometryParameter(std::string name, double value, AdjustEnumOptions option)
{
	m_name = name;
	m_value = value;
	m_adjust_option = option;
}

GeometryParameter::~GeometryParameter()
{
}

std::string GeometryParameter::toString() const
{
	return m_name + ": " + std::to_string(m_value) + " Adjust:" + std::to_string(m_adjust_option);
}

bool GeometryModel::setParameter(std::vector<GeometryParameter> parameter)
{
	for (auto const& p : parameter) {
		if (!setParameter(p)) {
			std::cerr << "Error: Unable set Parameter " << p.m_name << " in Geometry " << getName() << std::endl;
			return false;
		}
	}

	return true;
}


bool GeometryModel::setParameter(GeometryParameter parameter)
{
	for (auto& m_p : m_parameter) {
		if (parameter.m_name == m_p.m_name) {
			m_p = parameter;
			return true;
		}
	}
	std::cerr << "Error: Parameter " << parameter.toString() << " not found." << std::endl;
	return false;
}

cv::Point2d GeometryModel::pixel2metric(const double& x_pixel, const double& y_pixel)
{
	return cv::Point2d(
		(x_pixel - (double(getImageSizePixel().width) * 0.5)) * getPixelSize(),
		-(y_pixel - (double(getImageSizePixel().height) * 0.5)) * getPixelSize());
}

cv::Point2d GeometryModel::metric2pixel(const double& x_metric, const double& y_metric)
{
	return cv::Point2d(
		(x_metric / getPixelSize()) + (double(getImageSizePixel().width) * 0.5),
		-(y_metric / getPixelSize()) + (double(getImageSizePixel().height) * 0.5));
}


GeometryParameter& GeometryModel::getParameterByName(std::string name)
{
	auto pMap = getParameterMap();
	if (pMap.find(name) != pMap.end()) {
		return *pMap[name];
	}
	throw std::runtime_error("Error: Parameter " + name + " not found.");
}


std::string GeometryModel::toString()
{
	std::stringstream s;
	s << getName() << "\nSensor: " << m_image_size_px << " pixel size: " << m_pixel_size_mm;
	for (auto& p : m_parameter) {
		s << "\n" << p.toString();
		for (auto& op : m_overwrite_parameter) {
			if (op->m_name == p.m_name) {
				s << " *(" << op->toString() << ")";
			}
		}
	}
	s << "\nTransformation: " << m_transformation.transpose() << " Adjust:" << m_transformation_adjust << std::endl;
	return s.str();
}

std::map<std::string, GeometryParameter*> GeometryModel::getParameterMap()
{
	// Create map of parameter
	std::map<std::string, GeometryParameter*> _plist;
	for (auto& _p : m_parameter) {
		_plist[_p.m_name] = &_p;
	}

	// overwrite parameter
	for (auto& p_overwrite : m_overwrite_parameter) {
		_plist[p_overwrite->m_name] = p_overwrite;
	}
	return _plist;
}

std::map<int, cv::Point2d> GeometryModel::projectPoints(std::map<int, std::vector<double>>& object_points, double gantry_angle, bool in_pixel)
{
	Eigen::Matrix<double, 3, 4> mp_eigen = getProjectionMatrix(gantry_angle);
	std::map<int, cv::Point2d > proj_points;
	for (auto& p : object_points) {
		// Rotate and translate object
		double x[3];
		ceres::AngleAxisRotatePoint<double>(&m_transformation[0], &p.second[0], &x[0]);
		x[0] += m_transformation[3];
		x[1] += m_transformation[4];
		x[2] += m_transformation[5];
		Eigen::Vector<double, 4> epoint(x[0], x[1], x[2], double(1.0));

		//Eigen::Vector<double, 4> epoint(p.second[0], p.second[1], p.second[2], 1.0);
		Eigen::Vector<double, 3> pr_point = mp_eigen * epoint;
		double _x = (pr_point[0] / pr_point[2]);
		double _y = (pr_point[1] / pr_point[2]);

		if (in_pixel) {
			// convert to pixel
			proj_points[p.first] = metric2pixel(_x, _y);
		}
		else
		{
			proj_points[p.first] = cv::Point2d(_x, _y);
		}
	}

	return proj_points;
}
