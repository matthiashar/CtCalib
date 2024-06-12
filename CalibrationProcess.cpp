#include "CalibrationProcess.h"

#include <filesystem>
#include "ProjectionProcessing.h"
#include "io.h"

namespace fs = std::filesystem;

CalibrationProcess::CalibrationProcess() :
	searchForAdditionalObservationsInImage(true),
	residualsWithBackground(true),
	saveResiduals(false),
	removeOutliers(true),
	correctEccentricity(true),
	number_iterations(3),
	object_point_merge_threshold(1.0),
	outlier_iqr_multiplier(2.0),
	observation_search_threshold(1.0)
{

	// Enable this in case of rank deficientcy. Nessesary for calculating the covarianz matrix.
	calibration_options.covariance_options.algorithm_type = ceres::DENSE_SVD;
	calibration_options.covariance_options.null_space_rank = -1;
}

void CalibrationProcess::runCalibration(ProjectionData& pdata, GeometryModel& m, std::string out_path)
{
	std::cerr << "\n+++++++++++++ Start Calibration +++++++++++++\n" << pdata.toString() << std::endl;

	// create directory for output
	std::filesystem::create_directories(out_path);

	// calculate initial angle step value for model if not set
	GeometryParameter& GantryAngleStep = m.getParameterByName("GantryAngleStep");
	if (abs(GantryAngleStep.m_value) < 1.0e-8) {
		GantryAngleStep.m_value = (pdata.m_end_angle - pdata.m_start_angle) / double(pdata.m_projections.size());
		std::cout << "Info: Setting initial angle step value to " << GantryAngleStep.m_value << std::endl;
	}

	// -- Object points
	std::map<int, std::vector<double>> object_points = ProjectionProcessing::calculateInitialObjectPoints(pdata, m);
	io::writeObjectPoints(out_path + "initial_object_points.txt", object_points);
	std::cout << "Number of initial object points: " << std::to_string(object_points.size()) << std::endl << std::endl;

	// reset outliers
	for (auto& proj : pdata.m_projections) {
		for (auto& obs : proj.observation) {
			obs.point_typ = Projection::Observation::TYP::default;
		}
	}

	// set points at image edge to outlier (only use points that are fully visible)
	for (auto& proj : pdata.m_projections) {
		for (auto& obs : proj.observation) {
			double edge_threshold = obs.a;
			if (obs.x < edge_threshold || obs.y < edge_threshold)
				obs.point_typ = Projection::Observation::TYP::outlier;

			if (obs.x > m.getImageSizePixel().width - edge_threshold ||
				obs.y > m.getImageSizePixel().height - edge_threshold)
				obs.point_typ = Projection::Observation::TYP::outlier;
		}
	}

	// Initial calibration
	Calibration calib(calibration_options);
	calib.runCalibration(pdata, m, object_points);
	std::cout << "Iteration 0 - RMSE: " << calib.getResult().rmse << std::endl;

	for (auto iteration = 1; iteration < number_iterations; iteration++) {
		// Merge overlapping object points
		std::vector<std::pair<int, int>> merged_points = ProjectionProcessing::mergeOverlappingObjectPoints(pdata, m, object_points, object_point_merge_threshold);

		// Search for missing points
		marker_detection::Parameter copy_parameter;
		copy_parameter.sub_pixel_method = 1;
		int found_observations = ProjectionProcessing::searchMissingObservations(pdata, m, object_points, observation_search_threshold, searchForAdditionalObservationsInImage, copy_parameter);

		// Correct projection centers of sphere
		if (correctEccentricity) {
			ProjectionProcessing::correctEccentricity(pdata, m);
		}

		// Remove outliers
		int detected_outlier = 0;
		if (remove) {
			detected_outlier = ProjectionProcessing::detectOutlier(pdata, outlier_iqr_multiplier);
		}

		// Calibration
		if (iteration == 1 ||
			(merged_points.size() > 0 || found_observations > 0 || detected_outlier > 0)) {
			calib.runCalibration(pdata, m, object_points, true);
			std::cout << "Iteration " << iteration << " - RMSE: "
				<< calib.getResult().rmse << " ("
				<< merged_points.size() << " 3D points merged, "
				<< found_observations << " aditional observations found, "
				<< detected_outlier << " outlieres found)" << std::endl;
		}
		else
		{
			break;
		}
	}

	// Save geometry
	io::writeGeometry(out_path + "final_geometry.xml", m);

	// Save report
	std::ofstream report_out(out_path + "report.txt");
	report_out << calib.getReport();

	// Save final stats
	std::ofstream stats_out(out_path + "final_stats.txt");
	stats_out << calib.getResult().finalStatsToString();

	// Save calibration data
	pdata.write(out_path + "adjusted_calibration.xml");
	pdata.writeTextfile(out_path + "obs.txt");

	// Save adjusted object points
	io::writeObjectPoints(out_path + "adjusted_points.txt", object_points);

	// Residuals in one image (average over multible projections)
	cv::Mat resImages = ProjectionProcessing::plotAllResiduals(pdata, m.getImageSizePixel(), out_path + "averageResiduals.txt");
	cv::imwrite(out_path + "residuals.png", resImages);

	// Residuals for each projection
	if (saveResiduals) {
		std::string out_residuals = out_path + "res/";
		std::filesystem::create_directories(out_residuals);
		for (auto& p : pdata.m_projections) {
			cv::Mat _i = residualsWithBackground ? p.drawResiduals() : p.drawResiduals(m.getImageSizePixel());
			cv::imwrite(out_residuals + p.getImageName() + ".jpg", _i);
		}
	}

	// Save Correlation
	std::ofstream correlation(out_path + "correlation.txt");
	for (auto par : m.getParameter()) {
		if (par.m_adjust_option == ADJUST_PARAM)
			correlation << par.m_name << " ";
	}
	correlation << std::endl << calib.getResult().cor_parameter << std::endl;

	// Save RMSE by angle
	std::ofstream angleRMSE(out_path + "angleRMSE.txt");
	angleRMSE << "Index Angle[rad] RMSE_D[pixel] RMSE_X[pixel] RMSE_Y[pixel]\n";
	for (auto const& d : pdata.m_projections) {
		angleRMSE <<
			d.gantry_index << " " <<
			double(d.gantry_index) * GantryAngleStep.m_value << " " <<
			d.projection_rmseD << " " <<
			d.projection_rmseX << " " <<
			d.projection_rmseY << std::endl;
	}

	std::cout << "\nFinished Calibration\n" << calib.getResult().toString() << "\n" << m.toString() << "+++++++++++++\n" << std::endl;
}
