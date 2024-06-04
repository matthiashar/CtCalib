// Ceres
#include "ceres/ceres.h"

// Opencv
#include <opencv2/core.hpp>

#include <iostream>
#include <filesystem>

#include "io.h"
#include "ProjectionProcessing.h"
#include "CalibrationProcess.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
	// Get comand line arguments
	if (argc != 6) {
		std::cerr << "Usage: input_path sdd[mm] srd[mm] pixel_size[mm] rotation_max[rad] \n";
		return EXIT_FAILURE;
	}	
	fs::path path = fs::path(argv[1]);
	double sdd = std::atof(argv[2]);
	double srd = std::atof(argv[3]);
	double pixel_size = std::atof(argv[4]);
	double rotation_max = std::atof(argv[5]);

	std::cout << "Using the folwoing arguments:"
		<< "\n   Path: " << path
		<< "\n   SDD: " << sdd
		<< "\n   SRD: " << srd
		<< "\n   Pixel size: " << pixel_size
		<< "\n   Max rotation angle: " << rotation_max << "\n\n";

	// marker detection parameters
	marker_detection::Parameter parameter;
	parameter.marker_contrast_consistency = 0.3;
	parameter.marker_max_diameter = 450;
	parameter.marker_min_contrast = 20;
	parameter.detect_coded_marker = false;
	parameter.robust_ellipse_fit = true;
	parameter.max_ellipse_ratio = 1.2;
	parameter.sub_pixel_method = 0;
	parameter.sub_pixel_scan_lines = 100;
	parameter.median_blur_kernel = 3;

	// Try reading observation xml file
	ProjectionData pdata;
	bool new_detection = true;
	for (auto const& p : fs::directory_iterator(path)) {
		if (p.path().extension() == ".xml") {
			if (pdata.read(p.path().string())) {
				new_detection = false;
				break;
			}
		}
	}

	// Read images if no xml file
	if (new_detection) {
		std::cout << "\nInfo: Start detection..." << std::endl;
		double t = (double)cv::getTickCount();
		pdata = ProjectionProcessing::readProjectionImagesParallel(path.string(), parameter);
		ProjectionProcessing::trackPoints(pdata);
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		std::cout << "Info: Finished loading " << pdata.m_projections.size() << " images in " << t << " seconds." << std::endl;

		pdata.m_end_angle = rotation_max;
	}

	if (pdata.m_projections.empty()) {
		std::cerr << "Error: No projection data found.\n";
		return EXIT_FAILURE;
	}

	// System geometry model
	GeometryModel *model = new GeometryTilt(sdd, srd, pdata.m_projections.begin()->getImageSizePixel(), pixel_size);
	//GeometryModel* model = new GeometryDetector(sdd, srd, pdata.m_projections.begin()->getImageSizePixel(), pixel_size);
	//GeometryModel* model = new GeometryRTK(sdd, srd, pdata.m_projections.begin()->getImageSizePixel(), pixel_size);

	// Calbration
	CalibrationProcess c_process;
	if (new_detection) {
		c_process.searchForAditionalObservationsInImage = true;
	}		
	c_process.runCalibration(pdata, *model, (path.parent_path() / "out/").string());

	// save detected points to data directory
	if (new_detection)
		pdata.write(path.string() + "/observations.xml");

	exit(0);
}
