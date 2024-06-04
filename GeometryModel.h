#ifndef GEOMETRYMODEL_H
#define GEOMETRYMODEL_H

#include "ceres/ceres.h"
#include <ceres/rotation.h>
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <vector>

#include "ProjectionData.h"

enum AdjustEnumOptions
{
	CONST_PARAM,
	ADJUST_PARAM
};

class GeometryParameter
{
public:
	GeometryParameter();
	GeometryParameter(std::string name, double value, AdjustEnumOptions option);
	~GeometryParameter();
	std::string m_name;
	double m_value;
	AdjustEnumOptions m_adjust_option;
	std::string toString() const;
};


class GeometryModel
{
public:
	// Method for creating a copy of the model
	virtual GeometryModel* clone() const = 0;

	// Get
	std::vector<GeometryParameter>& getParameter() { return m_parameter; }
	GeometryParameter& getParameterByName(std::string name);
	Eigen::Matrix<double, 6, 1>& getTransformation() { return m_transformation; }
	AdjustEnumOptions getTransformationAdjust() { return m_transformation_adjust; }
	virtual double getSDD(){ return getParameterByName("SDD").m_value; };
	virtual double getSOD(){ return getParameterByName("SRD").m_value; };
	virtual double getPrincipalPointX() = 0;
	virtual double getPrincipalPointY() = 0;
	virtual Eigen::Matrix<double, 3, 4> getProjectionMatrix(double gantry_angle) = 0;
	virtual std::map<int, cv::Point2d> projectPoints(std::map<int, std::vector<double>>& object_points, double gantry_angle, bool in_pixel = true);
	virtual std::string getName() const = 0;
	cv::Size2i getImageSizePixel() const { return m_image_size_px; }
	unsigned int getNumberOfParameters() const { return m_parameter.size(); }

	// Set
	bool setParameter(std::vector<GeometryParameter> parameter);
	bool setParameter(GeometryParameter parameter);
	void setOverwriteParameter(std::vector<GeometryParameter*> overwrite_parameter) { m_overwrite_parameter = overwrite_parameter; }
	void setTransformation(Eigen::Matrix<double, 6, 1> transformation) { m_transformation = transformation; }
	void setTransformationAdjust(AdjustEnumOptions adjust) { m_transformation_adjust = adjust; }
	void setImageSizePixel(cv::Size2i size) { m_image_size_px = size; }
	double getPixelSize() const { return m_pixel_size_mm; }
	void setPixelSize(double size) { m_pixel_size_mm = size; }

	/// Convert pixel coordinate to metric
	cv::Point2d pixel2metric(const double &x_pixel, const double& y_pixel);

	/// Convert metric coordinate to pixel
	cv::Point2d metric2pixel(const double& x_metric, const double& y_metric);

	/// Adding observation to adjustment
	virtual ceres::ResidualBlockId addResidualBlock(
		ceres::Problem& problem,
		ceres::LossFunction* loss_function,
		std::vector<double>& object_point,
		double& gantry,
		double& x,
		double& y) = 0;

	/// Summary of geometry
	std::string toString();

	/// Return map for easy accses to parameters
	std::map<std::string, GeometryParameter*> getParameterMap();

protected:
	// Vector of parameters for the geometry model
	std::vector<GeometryParameter> m_parameter;

	// Vector of pointers to paramers of other models (e.g. if two models are adjusted but with only one common SDD)
	std::vector<GeometryParameter*> m_overwrite_parameter;

	// Transformation for object points if the coordinats not adjusted
	Eigen::Matrix<double, 6, 1> m_transformation;
	AdjustEnumOptions m_transformation_adjust;

	// Sensor information
	cv::Size2i m_image_size_px;
	double m_pixel_size_mm;
};

#endif // GEOMETRYMODEL_H
