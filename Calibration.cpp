#include "Calibration.h"

Calibration::Calibration(Options options)
{
	m_options = options;
}

bool Calibration::runCalibration(ProjectionData& projection_data,
	GeometryModel& model,
	std::map<int, std::vector<double>>& object_points,
	bool calculate_covariance)
{
	report << "-- Initial Geometry --\n" << model.toString() << std::endl;
	report << "\n-- Projection data --\n" << projection_data.toString();
	report << "Number object points: " << object_points.size() << std::endl;

	// reset projected points
	for (auto& proj : projection_data.m_projections) {
		for (auto& mp : proj.observation) {
			mp.residual_pixel = std::nullopt;
		}
	}

	// build problem
	ceres::Problem problem;

	std::map<int, std::vector<int>> missing_points;
	std::vector < std::pair<Projection::Observation*, ceres::ResidualBlockId> > residual_ids;
	for (auto& proj : projection_data.m_projections) {
		// create loss function
		ceres::LossFunction* loss = new ceres::HuberLoss(m_options.huber_loss_a);

		// Add observations
		for (auto& mp : proj.observation) {
			if (object_points.find(mp.id) == object_points.end()) {
				missing_points[mp.id].push_back(1);
				continue;
			}

			// scip points
			if (mp.point_typ != Projection::Observation::TYP::default) {
				continue;
			}

			// convert observations to mm
			if (!mp.point_metric) {
				mp.point_metric = model.pixel2metric(mp.x, mp.y);
			}

			// Correct eccentricity
			cv::Point2d obs = mp.point_metric.value();
			if (mp.eccentricity_correction) {
				obs.x += mp.eccentricity_correction.value().x;
				obs.y += mp.eccentricity_correction.value().y;
			}

			// add residual block
			auto resId = model.addResidualBlock(problem, loss, object_points[mp.id], proj.gantry_index, obs.x, obs.y);
			residual_ids.push_back(std::make_pair(&mp, resId));
		}

		// Gantry index always fix
		if (problem.HasParameterBlock(&proj.gantry_index)) {
			problem.SetParameterBlockConstant(&proj.gantry_index);
		}
	}

	// Report number of missing points
	if (missing_points.size() > 0) {
		report << "Warning: " << missing_points.size() << " point(s) without 3D coordinats ";
		for (auto const& m : missing_points)
			report << m.first << "(" << m.second.size() << ") ";
		report << std::endl;
	}

	// Set parameters const
	for (auto& param : model.getParameter()) {
		if (problem.HasParameterBlock(&param.m_value) &&
			param.m_adjust_option == AdjustEnumOptions::CONST_PARAM) {
			problem.SetParameterBlockConstant(&param.m_value);
		}
	}
	if (!m_options.adjust_object_points) {
		for (auto& op : object_points) {
			if (problem.HasParameterBlock(&op.second[0])) {
				problem.SetParameterBlockConstant(&op.second[0]);
			}
		}
	}
	// transformation
	if (problem.HasParameterBlock(&model.getTransformation()[0]) &&
		model.getTransformationAdjust() == AdjustEnumOptions::CONST_PARAM) {
		problem.SetParameterBlockConstant(&model.getTransformation()[0]);
	}

	// Solve
	ceres::Solve(m_options.solver_options, &problem, &result.ceres_summary);

	// Calc S0
	result.s0 = std::sqrt((result.ceres_summary.final_cost * 2.0) / double(result.ceres_summary.num_residuals - result.ceres_summary.num_parameters_reduced));

	// Calc Residuals
	for (auto& res : residual_ids) {
		double cost = 0;
		double residual[2] = { 0,0 };
		if (problem.EvaluateResidualBlock(res.second, false, &cost, residual, nullptr)) {
			res.first->residual_pixel = cv::Point2d(
				residual[0] / model.getPixelSize(),
				-residual[1] / model.getPixelSize());
		}
	}
	residual_ids.clear();
	double resSumX = 0, resSumY = 0;
	int n_points = 0;
	std::map<int, std::vector<double>> rmse_points;
	for (auto& proj : projection_data.m_projections) {
		double proj_resSumX = 0, proj_resSumY = 0;
		int proj_n_points = 0;
		for (auto& point : proj.observation) {
			if (point.residual_pixel) {
				double powX = pow(point.residual_pixel.value().x, 2);
				double powY = pow(point.residual_pixel.value().y, 2);
				proj_resSumX += powX;
				proj_resSumY += powY;
				rmse_points[point.id].push_back(powX + powY);
				proj_n_points++;
			}
		}

		// rmse per projection
		proj.projection_rmseD = sqrt((proj_resSumX + proj_resSumY) / double(proj_n_points));
		proj.projection_rmseX = sqrt(proj_resSumX / double(proj_n_points));
		proj.projection_rmseY = sqrt(proj_resSumY / double(proj_n_points));

		// all projections
		resSumX += proj_resSumX;
		resSumY += proj_resSumY;
		n_points += proj_n_points;
	}

	// RMSE for calibration
	result.rmseX = std::sqrt(resSumX / double(n_points));
	result.rmseY = std::sqrt(resSumY / double(n_points));
	result.rmse = std::sqrt((resSumX + resSumY) / double(n_points));

	// RMSE for each point
	for (auto const& rp : rmse_points) {
		double sum = std::accumulate(rp.second.begin(), rp.second.end(), 0.0);
		result.rmse_object_points[rp.first] = std::sqrt(sum) / rp.second.size();
	}

	// calculate mean eccentricity correction
	double mean_eccentricity = 0;
	int n_eccentricity = 0;
	for (auto const& proj : projection_data.m_projections) {
		for (auto const& obs : proj.observation) {
			if (obs.eccentricity_correction.has_value()) {
				n_eccentricity++;
				mean_eccentricity += cv::norm(obs.eccentricity_correction.value());
			}
		}
	}
	mean_eccentricity = mean_eccentricity / double(n_eccentricity);
	result.mean_eccentricity_px = mean_eccentricity / model.getPixelSize();

	// Compute covariance matrix
	if (calculate_covariance) {
		ceres::Covariance covariance(m_options.covariance_options);
		std::vector<GeometryParameter*> adjusted_parameter;
		for (auto& param : model.getParameter()) {
			if (problem.HasParameterBlock(&param.m_value)) {
				if (!problem.IsParameterBlockConstant(&param.m_value)) {
					adjusted_parameter.push_back(&param);
				}
			}
		}
		std::vector<std::pair<const double*, const double*> > covariance_blocks;
		for (int i = 0; i < adjusted_parameter.size(); i++) {
			covariance_blocks.push_back(std::make_pair(&adjusted_parameter[i]->m_value, &adjusted_parameter[i]->m_value));

			//Correlation (make sure every pair is just added once)
			for (int j = 0; j < adjusted_parameter.size(); j++) {
				if (j < i) {
					covariance_blocks.push_back(std::make_pair(&adjusted_parameter[i]->m_value, &adjusted_parameter[j]->m_value));
				}
			}
		}
		for (auto& opoint : object_points) {
			if (problem.HasParameterBlock(&opoint.second[0])) {
				if (!problem.IsParameterBlockConstant(&opoint.second[0])) {
					covariance_blocks.push_back(std::make_pair(&opoint.second[0], &opoint.second[0]));
				}
			}
		}

		bool evaluation_succsesfull = covariance.Compute(covariance_blocks, &problem);
		result.cor_parameter = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(adjusted_parameter.size(), adjusted_parameter.size());
		if (evaluation_succsesfull) {
			// sd parameter
			std::map<GeometryParameter*, double> cov_parameter;
			for (auto& param : adjusted_parameter) {
				Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> cov_intrinsics(1, 1); // assuming only one parameter in each block
				covariance.GetCovarianceBlock(&param->m_value, &param->m_value, cov_intrinsics.data());
				cov_parameter[param] = sqrt(cov_intrinsics(0, 0));
				result.sd_parameter[param] = cov_parameter[param] * result.s0;
			}

			// correlation parameter
			for (int i = 0; i < adjusted_parameter.size(); i++) {
				for (int j = 0; j < adjusted_parameter.size(); j++) {
					if (j > i) {
						result.cor_parameter(i, j) = 0;
					}
					else {
						Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> cor(1, 1); // assuming only one parameter in each block
						covariance.GetCovarianceBlock(&adjusted_parameter[i]->m_value, &adjusted_parameter[j]->m_value, cor.data());
						result.cor_parameter(i, j) = cor(0, 0) / (cov_parameter[adjusted_parameter[i]] * cov_parameter[adjusted_parameter[j]]);
					}
				}
			}

			// sd object points
			for (auto& opoint : object_points) {
				if (problem.HasParameterBlock(&opoint.second[0])) {
					if (!problem.IsParameterBlockConstant(&opoint.second[0])) {
						Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> cov_opoints(3, 3);
						covariance.GetCovarianceBlock(&opoint.second[0], &opoint.second[0], cov_opoints.data());
						auto sd_p = (cov_opoints.diagonal().cwiseSqrt() * result.s0);
						result.sd_object_points[opoint.first] = { sd_p(0), sd_p(1) ,sd_p(2) };
					}
				}
			}
		}
		else
		{
			report << "Error: Unable to compute covariance matrix (rank deficiency in the Jacobian)." << std::endl;
		}
	}

	report << "\n-- Ceres report --" << result.ceres_summary.FullReport() << std::endl;
	report << "-- Final Geometry --\n" << model.toString() << std::endl;
	if (calculate_covariance) {
		report << "\n-- Final Stats --";
		report << "\nS0 " << result.s0 / model.getPixelSize();
		report << "\nRMSE_x " << result.rmseX;
		report << "\nRMSE_y " << result.rmseY;
		report << "\nRMSE_d " << result.rmse;
		report << "\nEccentricity_mean " << result.mean_eccentricity_px;
		report << "\nnumber_image_points " << result.ceres_summary.num_residual_blocks_reduced;
		report << "\nnumber_object_points " << object_points.size();
		report << "\n-- Standard deviation parameter--\n";
		for (auto const& pm : result.sd_parameter)
			report << "  " << pm.first->m_name << " " << pm.first->m_value << " " << pm.second << std::endl;
		report << "\n-- Correlation parameter --\n" << result.cor_parameter << std::endl;
		report << "\n-- Object points --\nID X Y Z sdX sdY sdZ rmse\n";
		std::vector<double> max_sd_object_points = { 0.0, 0.0, 0.0 };
		for (auto const& sd : result.sd_object_points) {
			report << sd.first << " " << object_points[sd.first][0] << " " << object_points[sd.first][1] << " "
				<< object_points[sd.first][2] << " " << sd.second[0] << " " << sd.second[1] << " " << sd.second[2] <<
				" " << result.rmse_object_points[sd.first] << std::endl;
			if (sd.second[0] > max_sd_object_points[0])
				max_sd_object_points[0] = sd.second[0];
			if (sd.second[1] > max_sd_object_points[1])
				max_sd_object_points[1] = sd.second[1];
			if (sd.second[2] > max_sd_object_points[2])
				max_sd_object_points[2] = sd.second[2];
		}
		report << "Maximum: " << max_sd_object_points[0] << " " << max_sd_object_points[1] << " " << max_sd_object_points[2] << std::endl;
	}
	report << std::endl;

	return true;
}


bool Calibration::runCalibration(std::vector<InData>& data,
	std::map<int, std::vector<double>>& object_points,
	bool calculate_covariance)
{
	report << "Number object points: " << object_points.size() << std::endl;
	report << "\n-- Input data --\n";
	for (auto const& d : data) {
		report << d.model->toString() << std::endl;
		report << d.projection_data.toString() << std::endl;;
	}
	report << std::endl;

	// check data
	// todo

	// reset projected points
	for (auto& d : data) {
		for (auto& proj : d.projection_data.m_projections) {
			for (auto& mp : proj.observation) {
				mp.residual_pixel = std::nullopt;
			}
		}
	}

	// build problem
	ceres::Problem problem;
	std::map<int, std::vector<int>> missing_points;
	std::vector < std::pair<Projection::Observation*, ceres::ResidualBlockId> > residual_ids;
	for (auto& d : data) {
		for (auto& proj : d.projection_data.m_projections) {
			// create loss function
			ceres::LossFunction* loss = new ceres::HuberLoss(0.05);

			// Add observations
			for (auto& mp : proj.observation) {
				if (object_points.find(mp.id) == object_points.end()) {
					missing_points[mp.id].push_back(1);
					continue;
				}

				// scip points
				if (mp.point_typ != Projection::Observation::TYP::default) {
					continue;
				}

				// convert observations to mm
				if (!mp.point_metric) {
					mp.point_metric = d.model->pixel2metric(mp.x, mp.y);
				}

				// Correct eccentricity
				cv::Point2d obs = mp.point_metric.value();
				if (mp.eccentricity_correction) {
					obs.x += mp.eccentricity_correction.value().x;
					obs.y += mp.eccentricity_correction.value().y;
				}

				// add residual block
				auto resId = d.model->addResidualBlock(problem, loss, object_points[mp.id], proj.gantry_index, obs.x, obs.y);
				residual_ids.push_back(std::make_pair(&mp, resId));
			}

			// -- Set parameters const
			// Gantry index always fix
			if (problem.HasParameterBlock(&proj.gantry_index)) {
				problem.SetParameterBlockConstant(&proj.gantry_index);
			}
			// model parameters
			for (auto& param : d.model->getParameter()) {
				if (problem.HasParameterBlock(&param.m_value) &&
					param.m_adjust_option == AdjustEnumOptions::CONST_PARAM) {
					problem.SetParameterBlockConstant(&param.m_value);
				}
			}
			// transformation
			if (problem.HasParameterBlock(&d.model->getTransformation()[0]) &&
				d.model->getTransformationAdjust() == AdjustEnumOptions::CONST_PARAM) {
				problem.SetParameterBlockConstant(&d.model->getTransformation()[0]);
			}
		}

		// Report missing points
		if (missing_points.size() > 0) {
			report << "Warning: " << missing_points.size() << " point(s) without 3D coordinats ";
			for (auto const& m : missing_points)
				report << m.first << "(" << m.second.size() << ") ";
			report << std::endl;
		}
	}

	// Set object points const
	if (!m_options.adjust_object_points) {
		for (auto& op : object_points) {
			if (problem.HasParameterBlock(&op.second[0])) {
				problem.SetParameterBlockConstant(&op.second[0]);
			}
		}
	}

	// Solve
	ceres::Solve(m_options.solver_options, &problem, &result.ceres_summary);

	// Calc S0
	result.s0 = std::sqrt((result.ceres_summary.final_cost * 2.0) / double(result.ceres_summary.num_residuals - result.ceres_summary.num_parameters_reduced));

	// Calc Residuals
	for (auto& res : residual_ids) {
		double cost = 0;
		double residual[2] = { 0.0 ,0.0 };
		if (problem.EvaluateResidualBlock(res.second, false, &cost, residual, nullptr)) {
			res.first->residual_pixel = cv::Point2d(
				residual[0] / data.front().model->getPixelSize(), // assuming the same pixel size for all models 
				-residual[1] / data.front().model->getPixelSize());
		}
	}
	residual_ids.clear();
	double resSumX = 0, resSumY = 0;
	int n_points = 0;
	for (auto& d : data) {
		for (auto& proj : d.projection_data.m_projections) {
			double proj_resSumX = 0, proj_resSumY = 0;
			int proj_n_points = 0;
			for (auto& point : proj.observation) {
				if (point.residual_pixel) {
					proj_resSumX += pow(point.residual_pixel.value().x, 2);
					proj_resSumY += pow(point.residual_pixel.value().y, 2);
					proj_n_points++;
				}
			}

			// rmse per projection
			proj.projection_rmseD = sqrt((proj_resSumX + proj_resSumY) / double(proj_n_points));
			proj.projection_rmseX = sqrt(proj_resSumX / double(proj_n_points));
			proj.projection_rmseY = sqrt(proj_resSumY / double(proj_n_points));

			// all projections
			resSumX += proj_resSumX;
			resSumY += proj_resSumY;
			n_points += proj_n_points;
		}
	}
	// RMSE for calibration
	result.rmseX = std::sqrt(resSumX / double(n_points));
	result.rmseY = std::sqrt(resSumY / double(n_points));
	result.rmse = std::sqrt((resSumX + resSumY) / (double(n_points) / 2.0));

	// Compute covariance matrix
	if (calculate_covariance) {
		ceres::Covariance covariance(m_options.covariance_options);
		std::vector<GeometryParameter*> adjusted_parameter;
		for (auto& d : data) {
			for (auto& param : d.model->getParameter()) {
				if (problem.HasParameterBlock(&param.m_value)) {
					if (!problem.IsParameterBlockConstant(&param.m_value)) {
						adjusted_parameter.push_back(&param);
					}
				}
			}
		}

		std::vector<std::pair<const double*, const double*> > covariance_blocks;
		for (int i = 0; i < adjusted_parameter.size(); i++) {
			covariance_blocks.push_back(std::make_pair(&adjusted_parameter[i]->m_value, &adjusted_parameter[i]->m_value));

			//Correlation (make sure every pair is just added once)
			for (int j = 0; j < adjusted_parameter.size(); j++) {
				if (j < i) {
					covariance_blocks.push_back(std::make_pair(&adjusted_parameter[i]->m_value, &adjusted_parameter[j]->m_value));
				}
			}
		}
		for (auto& opoint : object_points) {
			if (problem.HasParameterBlock(&opoint.second[0])) {
				if (!problem.IsParameterBlockConstant(&opoint.second[0])) {
					covariance_blocks.push_back(std::make_pair(&opoint.second[0], &opoint.second[0]));
				}
			}
		}

		bool evaluation_succsesfull = covariance.Compute(covariance_blocks, &problem);
		result.cor_parameter = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(adjusted_parameter.size(), adjusted_parameter.size());
		if (evaluation_succsesfull) {
			// sd parameter
			std::map<GeometryParameter*, double> cov_parameter;
			for (auto& param : adjusted_parameter) {
				Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> cov_intrinsics(1, 1); // assuming only one parameter in each block
				covariance.GetCovarianceBlock(&param->m_value, &param->m_value, cov_intrinsics.data());
				cov_parameter[param] = sqrt(cov_intrinsics(0, 0));
				result.sd_parameter[param] = cov_parameter[param] * result.s0;
			}

			// correlation parameter
			for (int i = 0; i < adjusted_parameter.size(); i++) {
				for (int j = 0; j < adjusted_parameter.size(); j++) {
					if (j > i) {
						result.cor_parameter(i, j) = 0;
					}
					else {
						Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> cor(1, 1); // assuming only one parameter in each block
						covariance.GetCovarianceBlock(&adjusted_parameter[i]->m_value, &adjusted_parameter[j]->m_value, cor.data());
						result.cor_parameter(i, j) = cor(0, 0) / (cov_parameter[adjusted_parameter[i]] * cov_parameter[adjusted_parameter[j]]);
					}
				}
			}

			// sd object points
			for (auto& opoint : object_points) {
				if (problem.HasParameterBlock(&opoint.second[0])) {
					if (!problem.IsParameterBlockConstant(&opoint.second[0])) {
						Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> cov_opoints(3, 3);
						covariance.GetCovarianceBlock(&opoint.second[0], &opoint.second[0], cov_opoints.data());
						auto sd_p = (cov_opoints.diagonal().cwiseSqrt() * result.s0);
						result.sd_object_points[opoint.first] = { sd_p(0), sd_p(1) ,sd_p(2) };
					}
				}
			}
		}
		else
		{
			report << "Error: Unable to compute covariance matrix (rank deficiency in the Jacobian)." << std::endl;
		}
	}

	report << "\n-- Ceres report --" << result.ceres_summary.FullReport() << "-- Final Geometry --\n" << std::endl;
	for (auto& d : data) {
		report << d.model->toString() << std::endl;
	}
	report << "\n-- Stats --" << std::endl;
	report << "S0: " << result.s0 << " mm (Pixel: " << result.s0 / data.front().model->getPixelSize() << ")" << std::endl;
	report << "RMSE (Pixel): " << result.rmseX << " " << result.rmseY << std::endl;
	if (calculate_covariance) {
		report << "\n-- Standard deviation parameter--\n";
		for (auto const& pm : result.sd_parameter)
			report << "  " << pm.first->m_name << " " << pm.first->m_value << " " << pm.second << std::endl;
		report << "\n-- Correlation parameter --\n" << result.cor_parameter << std::endl;
		report << "\n-- Standard deviation object points --" << std::endl;
		std::vector<double> max_sd_object_points = { 0.0, 0.0, 0.0 };
		for (auto const& sd : result.sd_object_points) {
			report << sd.first << " " << object_points[sd.first][0] << " " << object_points[sd.first][1] << " "
				<< object_points[sd.first][2] << " " << sd.second[0] << " " << sd.second[1] << " " << sd.second[2] << std::endl;
			if (sd.second[0] > max_sd_object_points[0])
				max_sd_object_points[0] = sd.second[0];
			if (sd.second[1] > max_sd_object_points[1])
				max_sd_object_points[1] = sd.second[1];
			if (sd.second[2] > max_sd_object_points[2])
				max_sd_object_points[2] = sd.second[2];
		}
		report << "Maximum: " << max_sd_object_points[0] << " " << max_sd_object_points[1] << " " << max_sd_object_points[2] << std::endl;
	}
	report << std::endl;

	return true;
}