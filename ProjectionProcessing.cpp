#include "ProjectionProcessing.h"

#include <thread>
#include <mutex>
#include <filesystem>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

ProjectionData ProjectionProcessing::readProjectionImages(std::string path, marker_detection::Parameter parameter)
{
	ProjectionData pdata;
	pdata.path_projections = path;
	// Load projection image and detect spheres
	for (auto const& p : std::filesystem::directory_iterator(path)) {
		Projection _d(pdata.m_projections.size(), p.path().filename().string());

		// Load Images
		_d.setImagePath(p.path().u8string());
		cv::Mat image = _d.getImage();
		if (image.empty()) {
			std::cerr << "Error: unable to load " << _d.getImagePath() << " as image." << std::endl;
			continue;
		}

		// Detect shperes and save data
		std::vector<marker_detection::Ellipse> marker;
		marker_detection::detectAndDecode(image, marker, parameter);

		for (auto const& m : marker) {
			_d.observation.push_back(Projection::Observation(m.x, m.y, m.a, m.b, m.angle));
		}
		pdata.m_projections.push_back(_d);
	}
	return pdata;
}

ProjectionData ProjectionProcessing::readProjectionImagesParallel(std::string path, marker_detection::Parameter parameter, int number_of_threads)
{
	// Set number of threads
	cv::setNumThreads(number_of_threads);

	// Create projection data, todo: only for image files 
	ProjectionData pdata;
	pdata.path_projections = path;
	for (auto const& p : std::filesystem::directory_iterator(path)) {
		Projection _d(pdata.m_projections.size(), p.path().filename().string());
		_d.setImagePath(p.path().u8string());
		pdata.m_projections.push_back(_d);
	}

	ProjectionProcessing::ParallelProcess parallelProcess(parameter, pdata.m_projections);
	cv::parallel_for_(cv::Range(0, pdata.m_projections.size()), parallelProcess);

	return pdata;
}


bool ProjectionProcessing::trackPoints(ProjectionData& pdata)
{
	// Assign marker id's
	std::vector<int> point_index;
	for (int i = 0; i < pdata.m_projections.size(); i++) {
		for (auto& marker : pdata.m_projections[i].observation) {
			// skip if marker has id
			if (marker.id > 0)
				continue;

			// if first image assign markers
			if (i == 0) {
				marker.id = point_index.size() + 1;
				point_index.push_back(marker.id);
			}
			else { // find closest marker in previos image
				// list of visible ids
				std::vector<int> visible_ids;
				for (auto const& _temp : pdata.m_projections[i].observation) {
					if (_temp.id > 0)
						visible_ids.push_back(_temp.id);
				}

				double dist = 10;
				int closest_id = -1;
				for (auto const& previos_img_m : pdata.m_projections[i - 1].observation) {
					double _d = sqrt(pow(previos_img_m.x - marker.x, 2) + pow(previos_img_m.y - marker.y, 2));
					if (_d < dist && std::find(visible_ids.begin(), visible_ids.end(), previos_img_m.id) == visible_ids.end()) {
						dist = _d;
						closest_id = previos_img_m.id;
					}
				}
				if (closest_id > 0)
					marker.id = closest_id;
			}

			if (marker.id < 0) {
				marker.id = point_index.size() + 1;
				point_index.push_back(marker.id);
			}
		}
	}

	return true;
}

std::map<int, std::vector<double>> ProjectionProcessing::calculateInitialObjectPoints(ProjectionData& projection_data, GeometryModel& geometry)
{
	// todo: chack data, check size of angle step, check if projection matrix is available
	double angle_step = geometry.getParameterByName("GantryAngleStep").m_value;

	std::map<int, std::vector<double>> _opoints;
	for (auto const& d1 : projection_data.m_projections) {
		// count number of unknown points
		int n_unknown_points = 0;
		for (auto const& p1 : d1.observation) {
			if (_opoints.find(p1.id) == _opoints.end() && p1.id > 0)
				n_unknown_points++;
		}
		if (n_unknown_points < 1)
			continue;

		for (auto const& d2 : projection_data.m_projections) {
			// scip suboptimal angles 
			double angle_diff = abs(abs(d1.gantry_index - d2.gantry_index) * angle_step);
			if (angle_diff < 0.35 * CV_PI || angle_diff > 0.65 * CV_PI) {
				continue;
			}

			// count number of unknown points
			int n_unknown_points_d2 = 0;
			for (auto const& p1 : d2.observation) {
				if (_opoints.find(p1.id) == _opoints.end() && p1.id > 0)
					n_unknown_points_d2++;
			}
			if (n_unknown_points_d2 < 1)
				continue;

			// Collect image points where no 3d coordinats are known
			std::vector<cv::Point2f> points1, points2;
			std::vector<int> ids;
			for (auto const& p1 : d1.observation) {
				if (_opoints.find(p1.id) == _opoints.end() && p1.id > 0) {
					// save 2d coordiants for triangulation for unknown points
					for (auto const& p2 : d2.observation) {
						if (p1.id == p2.id) {
							cv::Point2d _p1 = geometry.pixel2metric(p1.x, p1.y);
							cv::Point2d _p2 = geometry.pixel2metric(p2.x, p2.y);
							points1.push_back(_p1);
							points2.push_back(_p2);
							ids.push_back(p1.id);
						}
					}
				}
			}

			// Calc 3d coordinats
			if (!points1.empty()) {
				Eigen::Matrix<double, 3, 4> P1_eigen = geometry.getProjectionMatrix(d1.gantry_index * angle_step);
				Eigen::Matrix<double, 3, 4> P2_eigen = geometry.getProjectionMatrix(d2.gantry_index * angle_step);
				cv::Mat MP;

				cv::Mat P1, P2;
				cv::eigen2cv(P1_eigen, P1);
				cv::eigen2cv(P2_eigen, P2);

				cv::Mat points4D;
				std::vector<cv::Point3f> points3d;
				cv::triangulatePoints(P1, P2, points1, points2, points4D);
				cv::convertPointsFromHomogeneous(cv::Mat(points4D.t()).reshape(4, 1), points3d);
				for (int i = 0; i < ids.size(); i++) {
					_opoints[ids[i]] = { points3d[i].x, points3d[i].y , points3d[i].z };
				}
			}
		}
	}

	// count number of observations without object point
	int n_unknown_points = 0;
	for (auto& d1 : projection_data.m_projections) {
		// scip projection if now new points
		for (auto& p1 : d1.observation) {
			if (_opoints.find(p1.id) == _opoints.end()) {
				n_unknown_points++;
				p1.id = -1; // set -1
			}
		}
	}

	return _opoints;
}


cv::Mat ProjectionProcessing::plotAllResiduals(ProjectionData& projection_data,
	cv::Size2i image_size, std::string path_residual_out, int step_width, bool plotStats, double fontSize,
	double lineWidth, double error_scale, double line_spacing, int font, cv::Scalar color)
{
	// Textfile
	std::ofstream averageResidualOut(path_residual_out);
	averageResidualOut << "index id x y dx dy\n";

	// Create images
	cv::Mat image = cv::Mat::ones(image_size, CV_8UC3);
	image = cv::Scalar(255, 255, 255);
	int thickness = (image.size().width > 1000) ? image.size().width / 1000.0 : 1;
	double _font_size = fontSize * thickness;
	double _l_width = lineWidth * thickness;
	std::vector<double> diffs;
	std::map<int, std::vector<double>> obs_x, obs_y, obs_dx, obs_dy; // point_id, vector of (x, y, dx, dy)
	int n_obs = 0;
	for (auto const& proj : projection_data.m_projections) {
		for (auto const& m : proj.observation) {
			if (m.residual_pixel) {
				obs_x[m.id].push_back(m.x);
				obs_y[m.id].push_back(m.y);
				obs_dx[m.id].push_back(m.residual_pixel.value().x);
				obs_dy[m.id].push_back(m.residual_pixel.value().y);
				diffs.push_back(sqrt(pow(m.residual_pixel.value().x, 2) + pow(m.residual_pixel.value().y, 2)));
			}
		}
		n_obs++;
		if (n_obs >= step_width) {
			for (auto const& x : obs_x) {
				int id = x.first;
				double size = double(obs_x[id].size());
				double mean_x = std::accumulate(obs_x[id].begin(), obs_x[id].end(), 0.0) / size;
				double mean_y = std::accumulate(obs_y[id].begin(), obs_y[id].end(), 0.0) / size;
				double mean_dx = std::accumulate(obs_dx[id].begin(), obs_dx[id].end(), 0.0) / size;
				double mean_dy = std::accumulate(obs_dy[id].begin(), obs_dy[id].end(), 0.0) / size;
				averageResidualOut << proj.gantry_index << " " << id << " "
					<< mean_x << " " << mean_y << " " << mean_dx << " " << mean_dy << std::endl;

				if (abs(mean_dx + mean_dy) < 10000) {
					cv::line(image, cv::Point2d(mean_x, mean_y), cv::Point2d(mean_x + mean_dx * error_scale, mean_y - mean_dy * error_scale), color, thickness);
					cv::drawMarker(image, cv::Point2d(mean_x, mean_y), color, cv::MARKER_CROSS, thickness * 3, thickness);
				}
			}
			obs_x.clear();
			obs_y.clear();
			obs_dx.clear();
			obs_dy.clear();
			n_obs = 0;
		}
	}
	averageResidualOut.close();

	int baseline = 0;
	cv::Size tsize = cv::getTextSize("O", font, _font_size, thickness, &baseline);
	int l_spacing = tsize.height * line_spacing;

	// add scale 
	cv::line(image, cv::Point2d(30, 30), cv::Point2d(30 + 0.5 * error_scale, 30), color, thickness);
	cv::putText(image, "0.5 pixel", cv::Point(40 + 0.5 * error_scale, l_spacing * 1), font, _font_size, color, _l_width);

	// add text
	if (plotStats) {
		// RGB Image
		cv::putText(image, "Residuals (average over " + std::to_string(step_width) + " projections)", cv::Point(20, l_spacing * 2), font, _font_size, color, _l_width);
		cv::putText(image, "Scale: " + std::to_string(error_scale), cv::Point(20, l_spacing * 3), font, _font_size, color, _l_width);
		if (diffs.size() > 3) {
			auto result = std::minmax_element(diffs.begin(), diffs.end());
			cv::putText(image, "Min: " + std::to_string(*result.first), cv::Point(20, l_spacing * 4), font, _font_size, color, _l_width);
			cv::putText(image, "Mean: " + std::to_string(std::accumulate(diffs.begin(), diffs.end(), 0.0) / double(diffs.size())), cv::Point(20, l_spacing * 5), font, _font_size, color, _l_width);
			cv::putText(image, "Max: " + std::to_string(*result.second), cv::Point(20, l_spacing * 6), font, _font_size, color, _l_width);
		}
	}
	return image;
}

int ProjectionProcessing::detectOutlier(ProjectionData& projection_data, double multiplier)
{
	std::vector<double> resD;
	for (auto& proj : projection_data.m_projections) {
		for (auto const& point : proj.observation) {
			if (point.residual_pixel) {
				resD.push_back(point.residual_pixel.value().x);
				resD.push_back(point.residual_pixel.value().y);
			}
		}
	}

	int n_outlier = 0;
	if (resD.size() > 5) {
		std::sort(resD.begin(), resD.end());
		double q25 = resD[(resD.size() - 1) * 0.25];
		double q75 = resD[(resD.size() - 1) * 0.75];
		double iqr = q75 - q25;
		double threshold25 = q25 - multiplier * iqr;
		double threshold75 = q75 + multiplier * iqr;
		for (auto& proj : projection_data.m_projections) {
			for (auto& point : proj.observation) {
				if (point.residual_pixel) {
					if (point.residual_pixel.value().x > threshold75 ||
						point.residual_pixel.value().x < threshold25 ||
						point.residual_pixel.value().y > threshold75 ||
						point.residual_pixel.value().y < threshold25) {
						point.point_typ = Projection::Observation::TYP::outlier;
						n_outlier++;
					}
				}
			}
		}
	}

	return n_outlier;
}


std::vector<std::pair<int, int>> ProjectionProcessing::mergeOverlappingObjectPoints(ProjectionData& projection_data,
	GeometryModel& model, std::map<int, std::vector<double>>& object_points, double max_distance)
{
	// changing marker idis if points are overlapping
	std::vector<std::pair<int, int>> duplicate_points;
	for (auto const& p1 : object_points) {
		for (auto const& p2 : object_points) {
			if (p1.first == p2.first || p1.first > p2.first)
				continue;

			double dist = sqrt(
				pow(p1.second[0] - p2.second[0], 2) +
				pow(p1.second[1] - p2.second[1], 2) +
				pow(p1.second[2] - p2.second[2], 2));

			if (dist < max_distance) {
				duplicate_points.push_back(std::pair<int, int>(p1.first, p2.first));
			}
		}
	}

	// rename duplicate points
	for (auto& dp1 : duplicate_points) {
		for (auto& dp2 : duplicate_points) {
			if (dp2.first == dp1.second)
				dp2.first = dp1.first;
		}
	}

	for (auto& dp : duplicate_points) {
		// remove object point
		object_points.erase(dp.second);

		// rename observation
		for (auto& pd : projection_data.m_projections) {
			for (auto& m : pd.observation) {
				if (m.id == dp.second)
					m.id = dp.first;
			}
		}
	}

	return duplicate_points;
}


int ProjectionProcessing::searchMissingObservations(ProjectionData& projection_data,
	GeometryModel& geometry,
	std::map<int, std::vector<double>>& object_points,
	double max_search_distance,
	bool searchInImage,
	marker_detection::Parameter parameter) {
	double gantryAngleStep = geometry.getParameterByName("GantryAngleStep").m_value;
	int n_found_observations = 0;
	for (auto& proj : projection_data.m_projections) {
		int n_found_observations_proj = 0;

		// project object points
		std::map<int, cv::Point2d> proj_points = geometry.projectPoints(object_points, proj.gantry_index * gantryAngleStep, true);

		// remove obsvered objectpoints
		for (auto& obs : proj.observation) {
			if (proj_points.find(obs.id) != proj_points.end())
				proj_points.erase(obs.id);
		}

		// for remaining object points
		cv::Mat image;
		for (auto const& pp : proj_points) {
			// Check if in image
			if (pp.second.x < 0 || pp.second.y < 0)
				continue;
			if (pp.second.x > geometry.getImageSizePixel().width ||
				pp.second.y > geometry.getImageSizePixel().height)
				continue;

			// search points in observations without id
			double dist = max_search_distance;
			Projection::Observation* closest = nullptr;
			for (auto& obs : proj.observation) {
				double _d = sqrt(pow(obs.x - pp.second.x, 2) + pow(obs.y - pp.second.y, 2));
				if (_d < dist) {
					dist = _d;
					closest = &obs;
				}
			}
			if (closest != nullptr) {
				if (closest->id < 0) {
					closest->id = pp.first;
					n_found_observations_proj++;
				}
				else {
					// todo: change id?
				}
			}
			else if (searchInImage) { // Search directly in image
				if (image.empty()) {
					image = proj.getImage();
				}
				marker_detection::Ellipse ell = marker_detection::Ellipse(pp.second.x, pp.second.y, 1, 1, 0);
				if (marker_detection::searchMarker(image, ell, parameter)) {
					if (sqrt(pow(pp.second.x - ell.x, 2) + pow(pp.second.y - ell.y, 2)) < max_search_distance) {
						proj.observation.push_back(Projection::Observation(ell.x, ell.y, ell.a, ell.b, ell.angle, pp.first));
						n_found_observations_proj++;
					}
				}
			}
		}

		n_found_observations += n_found_observations_proj;
	}
	return n_found_observations;
}

bool ProjectionProcessing::correctEccentricity(ProjectionData& projection_data, GeometryModel& geometry, int method)
{
	double xH = geometry.getPrincipalPointX();
	double yH = geometry.getPrincipalPointY();
	double sdd = geometry.getSDD();

	for (auto& proj : projection_data.m_projections) {
		for (auto& obs : proj.observation) {
			// convert pixel to mm
			if (!obs.point_metric) {
				obs.point_metric = geometry.pixel2metric(obs.x, obs.y);
			}
			double a = 2 * obs.a * geometry.getPixelSize(); // * 2 because obs.a is half of major axis
			double b = 2 * obs.b * geometry.getPixelSize(); // * 2 because obs.b is half of minor axis

			// distance between principal point and observation on sensor
			double c = sqrt(pow(xH - obs.point_metric.value().x, 2) + pow(yH - obs.point_metric.value().y, 2));

			// distance between principal point and sphere center on sensor
			double m;
			switch (method)
			{
			case 0: // Deng 2015
				m = c - ((a * a) - (b * b)) / (4 * c);
				break;
			default: // Butzhammer 2023
				m = c * (1 - (b * b) / (4 * (sdd * sdd)));
				break;
			}

			// corrected position
			double cX = xH + (m / c) * (obs.point_metric.value().x - xH);
			double cY = yH + (m / c) * (obs.point_metric.value().y - yH);

			// correction
			double dx = obs.point_metric.value().x - cX;
			double dy = obs.point_metric.value().y - cY;

			// save correction
			obs.eccentricity_correction = cv::Point2d(dx, dy);
		}
	}

	return false;
}


