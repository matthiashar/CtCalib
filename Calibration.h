#ifndef CALIBRATION_H
#define CALIBRATION_H

#include "GeometryModel.h"

class Calibration
{
public:
	/// Options for bundel adjustment.
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

	/// Struct to save results after bundle adjustment.
	struct Result
	{
		ceres::Solver::Summary ceres_summary;
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> cor_parameter;
		std::map<GeometryParameter*, double> sd_parameter;
		std::map<int, std::vector<double>> sd_object_points;
		std::map<int, double> rmse_object_points;
		double s0, rmseX, rmseY, rmse, mean_eccentricity_px;

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

		std::string finalStatsToString() {
			std::stringstream s;
			s << "\n-- Final Stats --";
			s << "\nRMSE_x " << rmseX;
			s << "\nRMSE_y " << rmseY;
			s << "\nRMSE_d " << rmse;
			s << "\nEccentricity_mean " << mean_eccentricity_px;
			s << "\nnumber_image_points " << ceres_summary.num_residual_blocks_reduced;
			s << "\nnumber_object_points " << sd_object_points.size();
			s << "\n-- Standard deviation parameter--\n";
			for (auto const& pm : sd_parameter)
				s << pm.first->m_name << " " << pm.first->m_value << " " << pm.second << std::endl;
			return s.str();
		}
	};

	/// Struct for calibration of multiple data sets together
	struct InData
	{
		InData(std::string _name) : name(_name) {}
		ProjectionData projection_data;
		GeometryModel* model;
		std::string name;
		std::map<int, std::vector<double>> object_points;
	};

	Calibration(Options options = Options());

	/// Function to run adjustment of parameters for one data set
	bool runCalibration(ProjectionData& projection_data,
		GeometryModel& model,
		std::map<int, std::vector<double>>& object_points,
		bool calculate_covariance = false);

	/// Function to run adjustment of parameters for multiple data sets
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

