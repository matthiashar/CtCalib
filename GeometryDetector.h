#ifndef GEOMETRYDETECTOR_H
#define GEOMETRYDETECTOR_H

#include "GeometryModel.h"

class GeometryDetector : public GeometryModel
{
public:
	GeometryDetector(double SDD = 0.0,
		double SRD = 0.0,
		cv::Size2i imageSize = cv::Size2i(0, 0),
		double pixelSize = 1.0);

	GeometryModel* clone() const override { return new GeometryDetector(*this); }

	Eigen::Matrix<double, 3, 4> getProjectionMatrix(double gantry_angle) override;
	double getPrincipalPointX() override { return getParameterByName("ProjectionOffsetX").m_value; }
	double getPrincipalPointY() override { return getParameterByName("ProjectionOffsetY").m_value; }
	std::string getName() const override { return "Detector"; }

	ceres::ResidualBlockId addResidualBlock(
		ceres::Problem& problem,
		ceres::LossFunction* loss_function,
		std::vector<double>& object_point,
		double& gantry,
		double& x,
		double& y);

	// Templete function for creatin projection matrix
	template<typename T>
	static Eigen::Matrix<T, 3, 4> projectionMatrix(const T* const GantryAngle,
		const T* const OutOfPlaneAngle,
		const T* const InPlaneAngle,
		const T* const ProjectionOffsetX,
		const T* const ProjectionOffsetY,
		const T* const RotationOffsetX,
		const T* const SDD,
		const T* const SRD) {

		// Precompute trigonometric functions
		const T c_ip = cos(-*InPlaneAngle);
		const T s_ip = sin(-*InPlaneAngle);
		const T c_g = cos(-*GantryAngle);
		const T s_g = sin(-*GantryAngle);
		const T c_op = cos(-*OutOfPlaneAngle);
		const T s_op = sin(-*OutOfPlaneAngle);

		// Construct rotation matrices
		Eigen::Matrix<T, 3, 3> R_ip = Eigen::Matrix<T, 3, 3>::Identity();
		R_ip(0, 0) = c_ip;
		R_ip(0, 1) = -s_ip;
		R_ip(1, 0) = s_ip;
		R_ip(1, 1) = c_ip;
		Eigen::Matrix<T, 3, 3> R_op = Eigen::Matrix<T, 3, 3>::Identity();
		R_op(1, 1) = c_op;
		R_op(1, 2) = -s_op;
		R_op(2, 1) = s_op;
		R_op(2, 2) = c_op;
		Eigen::Matrix<T, 3, 3> R_g = Eigen::Matrix<T, 3, 3>::Identity();
		R_g(0, 0) = c_g;
		R_g(0, 2) = s_g;
		R_g(2, 0) = -s_g;
		R_g(2, 2) = c_g;

		// Rotation and offset of rotation axis
		Eigen::Matrix<T, 3, 4> rOffset = Eigen::Matrix<T, 3, 4>::Identity();
		rOffset(0, 3) = *RotationOffsetX;
		rOffset(2, 3) = -*SRD;
		rOffset.block<3, 3>(0, 0) = R_ip * R_op * R_g;

		// Projection to detector
		Eigen::Matrix<T, 3, 3> K = Eigen::Matrix<T, 3, 3>::Identity();
		K(0, 0) = -*SDD;
		K(1, 1) = -*SDD;
		K(0, 2) = *ProjectionOffsetX;
		K(1, 2) = *ProjectionOffsetY;

		// Return full projection matrix
		return K * rOffset;
	}

	struct reprojectionError {
		reprojectionError(double observed_x, double observed_y)
			: observed_x(observed_x), observed_y(observed_y) {}

		template <typename T>
		bool operator()(const T* const ObjectPoint,
			const T* const R_t,
			const T* const GantryIndex,
			const T* const GantryAngleStep,
			const T* const OutOfPlaneAngle,
			const T* const InPlaneAngle,
			const T* const ProjectionOffsetX,
			const T* const ProjectionOffsetY,
			const T* const RotationOffsetX,
			const T* const SDD,
			const T* const SRD,
			T* residuals) const {
			const T gantry_angle = *GantryIndex * *GantryAngleStep;
			Eigen::Matrix<T, 3, 4> matrix = projectionMatrix<T>(&gantry_angle,
				OutOfPlaneAngle,
				InPlaneAngle,
				ProjectionOffsetX,
				ProjectionOffsetY,
				RotationOffsetX,
				SDD,
				SRD);

			// Rotate and translate object
			T x[3];
			ceres::AngleAxisRotatePoint(R_t, ObjectPoint, x);
			x[0] += R_t[3];
			x[1] += R_t[4];
			x[2] += R_t[5];

			Eigen::Vector<T, 4> epoint(x[0], x[1], x[2], T(1.0));
			Eigen::Vector<T, 3> pr_point = matrix * epoint;

			// The error is the difference between the predicted and observed position.
			residuals[0] = (pr_point[0] / pr_point[2]) - observed_x;
			residuals[1] = (pr_point[1] / pr_point[2]) - observed_y;

			return true;
		}

		// Factory to hide the construction of the CostFunction object from
		// the client code.
		static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
			return (new ceres::AutoDiffCostFunction<reprojectionError, 2, 3, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1>(
				new reprojectionError(observed_x, observed_y)));
		}

		double observed_x;
		double observed_y;
	};
};

#endif // GEOMETRYDETECTOR_H

