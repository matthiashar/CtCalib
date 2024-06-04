#ifndef CALIBRATION_H
#define CALIBRATION_H

#include "GeometryModel.h"

class Calibration
{
public:
	// Options
	struct Options
	{
		Options() {
			solver_options.linear_solver_type = ceres::DENSE_SCHUR;
			solver_options.minimizer_progress_to_stdout = false;
			covariance_options.apply_loss_function = false;
			adjust_object_points = true;
			huber_loss_a = 0.1;
		}
		ceres::Solver::Options solver_options;
		ceres::Covariance::Options covariance_options;
		bool adjust_object_points;
		double huber_loss_a;
	};

	// Result
	struct Result
	{
		ceres::Solver::Summary ceres_summary;
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> cor_parameter;
		std::map<GeometryParameter*, double> sd_parameter;
		std::map<int, std::vector<double>> sd_object_points;
		std::map<int, double> rmse_object_points;
		double s0, rmseX, rmseY, rmse;

		std::string toString() {
			std::stringstream s;
			s << "RMSE: " << rmse << "\n"
				<< "Parameters: " << ceres_summary.num_parameters_reduced << "\n"
				<< "Residuals: " << ceres_summary.num_residuals_reduced << "\n" 
				<< "Successful steps : " << ceres_summary.num_successful_steps << "\n"
				<< "Unsuccessful steps : " << ceres_summary.num_unsuccessful_steps << "\n"
				<< "Is constrained: " << ceres_summary.is_constrained << "\n"
				<< "Termination: " << ceres_summary.termination_type << std::endl;
			return s.str();
		}
	};

	// Struct for calibration of multiple data sets together
	struct InData
	{
		InData(std::string _name) : name(_name) {}
		ProjectionData projection_data;
		GeometryModel* model;
		std::string name;
		std::map<int, std::vector<double>> object_points;
	};

	Calibration(Options options = Options());

	// function to run adjustment of parameters for one data set
	bool runCalibration(ProjectionData& projection_data,
		GeometryModel& model,
		std::map<int, std::vector<double>>& object_points,
		bool calculate_covariance = false);

	// function to run adjustment of parameters for multiple data sets
	bool runCalibration(std::vector<InData>& data,
		std::map<int, std::vector<double>>& object_points,
		bool calculate_covariance = false);

	std::string getReport() const { return report.str(); }
	Result getResult() const { return result; }

private:
	std::stringstream report;
	Result result;
	Options m_options;
};

#endif // CALIBRATION_H

