#ifndef GEOMETRYTILT_H
#define GEOMETRYTILT_H

#include "GeometryModel.h"

class GeometryTilt : public GeometryModel
{
public:
	GeometryTilt(double SDD = 0.0,
		double SRD = 0.0,
		cv::Size2i imageSize = cv::Size2i(0, 0),
		double pixelSize = 1.0);

	GeometryModel* clone() const override { return new GeometryTilt(*this); }

	Eigen::Matrix<double, 3, 4> getProjectionMatrix(double gantry_angle) override;
	std::map<int, cv::Point2d> projectPoints(std::map<int, std::vector<double>>& object_points, double gantry_angle, bool in_pixel = true) override;
	double getPrincipalPointX() override { return getParameterByName("ProjectionOffsetX").m_value; }
	double getPrincipalPointY() override { return getParameterByName("ProjectionOffsetY").m_value; }
	std::string getName() const override { return "Tilt"; }

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
		const T* const InPlaneAngle,
		const T* const ProjectionOffsetX,
		const T* const ProjectionOffsetY,
		const T* const SDD,
		const T* const SRD) {

		// Precompute trigonometric functions
		const T c_ip = cos(-*InPlaneAngle);
		const T s_ip = sin(-*InPlaneAngle);
		const T c_g = cos(-*GantryAngle);
		const T s_g = sin(-*GantryAngle);

		// Construct rotation matrices
		Eigen::Matrix<T, 4, 4> R_ip = Eigen::Matrix<T, 4, 4>::Identity();
		R_ip(0, 0) = c_ip;
		R_ip(0, 1) = -s_ip;
		R_ip(1, 0) = s_ip;
		R_ip(1, 1) = c_ip;

		Eigen::Matrix<T, 4, 4> R_g = Eigen::Matrix<T, 4,4>::Identity();
		R_g(0, 0) = c_g;
		R_g(0, 2) = s_g;
		R_g(2, 0) = -s_g;
		R_g(2, 2) = c_g;

		// Offset allong z axis
		Eigen::Matrix<T, 4, 4> rOffset = Eigen::Matrix<T, 4, 4>::Identity();
		rOffset(2, 3) = -*SRD;

		// Projection to detector
		Eigen::Matrix<T, 3, 4> K = Eigen::Matrix<T, 3, 4>::Identity();
		K(0, 0) = -*SDD;
		K(1, 1) = -*SDD;
		K(0, 2) = *ProjectionOffsetX;
		K(1, 2) = *ProjectionOffsetY;

		// Return full projection matrix
		return K * R_ip * rOffset * R_g;
	}

	template<typename T>
	static Eigen::Vector<T, 2> projectPoint(
		const T* const ObjectPoint,
		const T* const R_t,
		const T* const GantryAngle,
		const T* const InPlaneAngle,
		const T* const ProjectionOffsetX,
		const T* const ProjectionOffsetY,
		const T* const TauX,
		const T* const TauY,
		const T* const SDD,
		const T* const SRD) {
		// Rotate and translate object
		T x[3];
		ceres::AngleAxisRotatePoint(R_t, ObjectPoint, x);
		x[0] += R_t[3];
		x[1] += R_t[4];
		x[2] += R_t[5];
		Eigen::Vector<T, 4> epoint(x[0], x[1], x[2], T(1.0));

		// Gantry angle
		const T c_g = cos(-*GantryAngle);
		const T s_g = sin(-*GantryAngle);
		Eigen::Matrix<T, 4, 4> R_g = Eigen::Matrix<T, 4, 4>::Identity();
		R_g(0, 0) = c_g;
		R_g(0, 2) = s_g;
		R_g(2, 0) = -s_g;
		R_g(2, 2) = c_g;
		
		// Offset allong z axis
		Eigen::Matrix<T, 3, 4> rOffset = Eigen::Matrix<T, 3, 4>::Identity();
		rOffset(2, 3) = -*SRD;		

		// in plane roatation
		const T c_ip = cos(-*InPlaneAngle);
		const T s_ip = sin(-*InPlaneAngle);
		Eigen::Matrix<T, 3, 3> R_ip = Eigen::Matrix<T, 3, 3>::Identity();
		R_ip(0, 0) = c_ip;
		R_ip(0, 1) = -s_ip;
		R_ip(1, 0) = s_ip;
		R_ip(1, 1) = c_ip;

		// rotate point about gantry angle
		Eigen::Vector<T, 3> rpoint = R_ip * rOffset * R_g * epoint;
		
		 // tilt sensor (copy opencv https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/distortion_model.hpp)
		 T cTauX = cos(*TauX);
		 T sTauX = sin(*TauX);
		 T cTauY = cos(*TauY);
		 T sTauY = sin(*TauY);
		 Eigen::Matrix<T, 3, 3> matRotX = Eigen::Matrix<T, 3, 3>::Identity();
		 matRotX(1, 1) = cTauX;
		 matRotX(2, 1) = -sTauX;
		 matRotX(1, 2) = sTauX;
		 matRotX(2, 2) = cTauX;
		 Eigen::Matrix<T, 3, 3> matRotY = Eigen::Matrix<T, 3, 3>::Identity();
		 matRotY(0, 0) = cTauY;
		 matRotY(0, 2) = -sTauY;
		 matRotY(2, 0) = sTauY;
		 matRotY(2, 2) = cTauY;
		 const Eigen::Matrix<T, 3, 3> matRotXY = matRotY * matRotX;
		 Eigen::Matrix<T, 3, 3> matProjZ = Eigen::Matrix<T, 3, 3>::Identity();
		 matProjZ(0, 0) = matRotXY(2, 2);
		 matProjZ(0, 2) = -matRotXY(0, 2);
		 matProjZ(1, 1) = matRotXY(2, 2);
		 matProjZ(1, 2) = -matRotXY(1, 2);
		 const Eigen::Matrix<T, 3, 3>  matTilt = matProjZ * matRotXY;
		 
		 // tilt point
		 const Eigen::Vector<T, 3> tilt_point = matTilt * rpoint;		
		
		 // proj point
		 const Eigen::Vector<T, 2> proj_point(
		 	-*SDD * (tilt_point[0] / tilt_point[2]) + *ProjectionOffsetX,
		 	-*SDD * (tilt_point[1] / tilt_point[2]) + *ProjectionOffsetY);

		 // using matrix		
		 //Eigen::Matrix<T, 3, 3> K = Eigen::Matrix<T, 3, 3>::Identity();
		 //K(0, 0) = -*SDD;
		 //K(1, 1) = -*SDD;
		 //K(0, 2) = *ProjectionOffsetX;
		 //K(1, 2) = *ProjectionOffsetY;
		 //const Eigen::Vector<T, 3> projP = K * matTilt * rpoint;
		 //const Eigen::Vector<T, 2> proj_point((projP[0] / projP[2]),(projP[1] / projP[2]));

		return proj_point;
	}

	struct reprojectionError {
		reprojectionError(double observed_x, double observed_y)
			: observed_x(observed_x), observed_y(observed_y) {}

		template <typename T>
		bool operator()(const T* const ObjectPoint,
			const T* const R_t,
			const T* const GantryIndex,
			const T* const GantryAngleStep,
			const T* const InPlaneAngle,
			const T* const ProjectionOffsetX,
			const T* const ProjectionOffsetY,
			const T* const TauX,
			const T* const TauY,
			const T* const SDD,
			const T* const SRD,
			T* residuals) const {
			const T gantry_angle = *GantryIndex * *GantryAngleStep;

			Eigen::Vector<T, 2> p = projectPoint(
				ObjectPoint,
				R_t,
				&gantry_angle,
				InPlaneAngle,
				ProjectionOffsetX,
				ProjectionOffsetY,
				TauX,
				TauY,
				SDD,
				SRD);
			residuals[0] = p[0]  - observed_x;
			residuals[1] = p[1]  - observed_y;

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

