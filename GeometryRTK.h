#ifndef GEOMETRYRTK_H
#define GEOMETRYRTK_H

#include "GeometryModel.h"

class GeometryRTK : public GeometryModel
{
public:
	GeometryRTK(double SDD = 0.0,
		double SRD = 0.0,
		cv::Size2i imageSize = cv::Size2i(0, 0),
		double pixelSize = 1.0);

	GeometryModel* clone() const override { return new GeometryRTK(*this); }

	Eigen::Matrix<double, 3, 4> getProjectionMatrix(double gantry_angle) override;
	double getPrincipalPointX() override { return getParameterByName("ProjectionOffsetX").m_value * -1; }
	double getPrincipalPointY() override { return getParameterByName("ProjectionOffsetY").m_value * -1; }
	std::string getName() const override { return "RTK"; }

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
		const T* const SourceOffsetX,
		const T* const SourceOffsetY,
		const T* const SourceToDetectorDistance,
		const T* const SourceToIsocenterDistance) {

		const T c_ip = cos(-*InPlaneAngle);
		const T s_ip = sin(-*InPlaneAngle);
		const T c_g = cos(-*GantryAngle);
		const T s_g = sin(-*GantryAngle);
		const T c_op = cos(-*OutOfPlaneAngle);
		const T s_op = sin(-*OutOfPlaneAngle);

		Eigen::Matrix<T, 4, 4> R_ip = Eigen::Matrix<T, 4, 4>::Identity();
		R_ip(0, 0) = c_ip;
		R_ip(0, 1) = -s_ip;
		R_ip(1, 0) = s_ip;
		R_ip(1, 1) = c_ip;
		Eigen::Matrix<T, 4, 4> R_op = Eigen::Matrix<T, 4, 4>::Identity();
		R_op(1, 1) = c_op;
		R_op(1, 2) = -s_op;
		R_op(2, 1) = s_op;
		R_op(2, 2) = c_op;
		Eigen::Matrix<T, 4, 4> R_g = Eigen::Matrix<T, 4, 4>::Identity();
		R_g(0, 0) = c_g;
		R_g(0, 2) = s_g;
		R_g(2, 0) = -s_g;
		R_g(2, 2) = c_g;

		Eigen::Matrix<T, 4, 4> R = R_ip * R_op * R_g;

		Eigen::Matrix<T, 3, 3> proj_offset = Eigen::Matrix<T, 3, 3>::Identity();
		proj_offset(0, 2) = *SourceOffsetX - *ProjectionOffsetX;
		proj_offset(1, 2) = *SourceOffsetY - *ProjectionOffsetY;

		Eigen::Matrix<T, 3, 4> K = Eigen::Matrix<T, 3, 4>::Identity();
		K(0, 0) = -*SourceToDetectorDistance;
		K(1, 1) = -*SourceToDetectorDistance;
		K(2, 3) = -*SourceToIsocenterDistance;

		Eigen::Matrix<T, 4, 4> sourceOffset = Eigen::Matrix<T, 4, 4>::Identity();
		sourceOffset(0, 3) = -*SourceOffsetX;
		sourceOffset(1, 3) = -*SourceOffsetY;

		Eigen::Matrix<T, 3, 4> MP = proj_offset * K * sourceOffset * R;
		return MP;
	}

	struct rtkReprojectionError {
		rtkReprojectionError(double observed_x, double observed_y)
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
			const T* const SourceOffsetX,
			const T* const SourceOffsetY,
			const T* const SourceToDetectorDistance,
			const T* const SourceToIsocenterDistance,
			T* residuals) const {
			const T gantry_angle = *GantryIndex * *GantryAngleStep;
			Eigen::Matrix<T, 3, 4> matrix = projectionMatrix<T>(&gantry_angle,
				OutOfPlaneAngle,
				InPlaneAngle,
				ProjectionOffsetX,
				ProjectionOffsetY,
				SourceOffsetX,
				SourceOffsetY,
				SourceToDetectorDistance,
				SourceToIsocenterDistance);

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
			return (new ceres::AutoDiffCostFunction<rtkReprojectionError, 2, 3, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>(
				new rtkReprojectionError(observed_x, observed_y)));
		}

		double observed_x;
		double observed_y;
	};
};

#endif // GEOMETRYRTK_H

