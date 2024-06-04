#ifndef PROJECTIONPROCESSING_H
#define PROJECTIONPROCESSING_H

#include <opencv2/core.hpp>

#include "ProjectionData.h"
#include "GeometryModel.h"
#include "marker_detection/markerdetection.h"

namespace ProjectionProcessing {

	/// Read projections in single thread. 
	ProjectionData readProjectionImages(std::string path, marker_detection::Parameter parameter);

	/// Read projections in multiple threads. 
	ProjectionData readProjectionImagesParallel(std::string path, marker_detection::Parameter parameter, int number_of_threads = -1);

	/// Read detected markers over multiple projections and assign IDs. 
	bool trackPoints(ProjectionData& pdata);

	/// Calculate 3D coordinates for observations using geometry model. 
	std::map<int, std::vector<double>> calculateInitialObjectPoints(ProjectionData& projection_data, GeometryModel& geometry);

	/// Plot the residuals after the calibration. Average over multiple (set "step_width") projections.
	cv::Mat plotAllResiduals(ProjectionData& projection_data,
		cv::Size2i image_size,
		std::string path_residual_out = "",
		int step_width = 20,
		bool plotStats = true,
		double fontSize = 1.7,
		double lineWidth = 1,
		double error_scale = 250.0,
		double line_spacing = 1.5,
		int font = cv::FONT_HERSHEY_PLAIN,
		cv::Scalar color = cv::Scalar(0, 0, 0));

	/// Method for detecting outliers in projection data using interquartile range (IQR).
	int detectOutlier(ProjectionData& projection_data, double multiplier = 1.5);

	/// Method is searching for close points. Points within a threshold will be merged and IDs in the projection data reassigned. 
	std::vector<std::pair<int, int>> mergeOverlappingObjectPoints(
		ProjectionData& projection_data,
		GeometryModel& model,
		std::map<int, std::vector<double>>& object_points,
		double max_distance = 1.0);

	/// Method will project object points to image and search for markers without ID at location. If "searchInImage"=true the image will be loaded to directly search there.
	int searchMissingObservations(ProjectionData& projection_data,
		GeometryModel& geometry,
		std::map<int, std::vector<double>>& object_points,
		double max_search_distance,
		bool searchInImage = false,
		marker_detection::Parameter parameter = marker_detection::Parameter());

	/// Calculate eccentricity and save to projection data
	bool correctEccentricity(ProjectionData& projection_data, GeometryModel& geometry, int method = 1);

	/// Class for parallel projection image loading
	class ParallelProcess : public cv::ParallelLoopBody {
	private:
		std::vector<Projection>& outputImages;
		const marker_detection::Parameter& parameter;

	public:
		ParallelProcess(const marker_detection::Parameter& _parameter, std::vector<Projection>& output)
			: parameter(_parameter), outputImages(output) { }

		virtual void operator()(const cv::Range& range) const override {
			for (int i = range.start; i < range.end; i++) {
				Projection* _d = &outputImages[i];

				cv::Mat image = _d->getImage();
				if (!image.empty()) {
					// Detect spheres and save data
					std::vector<marker_detection::Ellipse> marker;
					marker_detection::Parameter local_parameter = parameter;
					marker_detection::detectAndDecode(image, marker, local_parameter);
					_d->setImageSizePixel(image.size());
					for (auto const& m : marker) {
						_d->observation.push_back(Projection::Observation(m.x, m.y, m.a, m.b, m.angle));
					}
				}
			}
		}
	};

} // end namespace ProjectionProcessing

#endif // PROJECTIONPROCESSING_H
