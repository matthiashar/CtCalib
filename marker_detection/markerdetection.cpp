#include "markerdetection.h"
#include <iostream>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "codeLookUp.h"

namespace marker_detection {

	void detectAndDecode(const cv::Mat& image, std::vector<Ellipse>& markers, Parameter param)
	{
		int codeBit = 14; // Only for 14
		if (image.type() != CV_8U) {
			std::cerr << "Error: only for images of type CV_8U!" << std::endl;
			return;
		}

		// Detect edge points of marker in image
		std::vector<std::vector<cv::Point>> edgePoints;
		findConnectedEdgePoints(image, edgePoints, param);

		unsigned int n_scan_lines = param.code_scan_resolution_per_element * codeBit;
		double scan_resolutionRad = (CV_PI * 2.0) / double(n_scan_lines);
		std::vector<int> code_pixel(n_scan_lines);
		std::string code_string("0", codeBit);
		unsigned int minEdgePixel = round(CV_PI * param.marker_min_diameter); // circumference = diameter*PI (assuming circle)
		unsigned int maxEdgepixel = round(CV_PI * param.marker_max_diameter);
		Ellipse e;
		std::vector<Ellipse> temp_uncodedmarkers;
		std::vector<int> found_ids;
		for (auto const& edgePixel : edgePoints)
		{
			// Ignore large and small contours
			if (edgePixel.size() < minEdgePixel ||
				edgePixel.size() > maxEdgepixel) {
				continue;
			}

			e = fitEllipse(edgePixel, param.ellipse_fit_type, param.robust_ellipse_fit);

			// small ellipse
			if (e.a * 2.0 < param.marker_min_diameter || e.b * 2.0 < param.marker_min_diameter) {
				continue;
			}

			// Ratio
			if (e.a / e.b > param.max_ellipse_ratio) {
				continue;
			}

			// Get mean pixel value of marker and of surounding area
			double marker_value, marker_rmse, surrounding_value, surrounding_rmse;
			check_marker_surrounding(e, image, marker_value, marker_rmse, surrounding_value, surrounding_rmse);

			// Contrast between marker and suroundings and rmse
			if (abs(marker_value - surrounding_value) < param.marker_min_contrast ||
				marker_rmse > param.max_marker_value_rmse ||
				surrounding_rmse > param.max_surrounding_value_rmse) {
				continue;
			}

			// Check if white marker on black background
			bool marker_white = (marker_value > surrounding_value);
			float middle_value = (marker_value + surrounding_value) * 0.5;

			// Optional sub pixel measurement
			if (!subPixelMeasurment(image, e, e, param, marker_value, surrounding_value))
				continue;

			// Scip code detection if "detect_coded_marker" is set to false
			if (!param.detect_coded_marker) {
				temp_uncodedmarkers.push_back(e);
				continue;
			}

			// Scanning code ring (todo: more robust with mutible scan rings)
			double alpha = sin(e.angle - CV_PI * 0.5) * param.code_ring_radius;
			double beta = cos(e.angle - CV_PI * 0.5) * param.code_ring_radius;
			double x, y;
			int n1 = 0;
			cv::Point2f point = cv::Point2f();
			bool uncompleat = false;
			for (unsigned int i = 0; i < n_scan_lines; i++) {
				x = e.a * cos(double(i) * scan_resolutionRad);
				y = e.b * sin(double(i) * scan_resolutionRad);
				point.x = x * beta - y * alpha + e.x;
				point.y = x * alpha + y * beta + e.y;
				float pixel_value;
				if(getSubPixValueWithChecks(image, point, pixel_value)){
					if ((pixel_value > middle_value) ^ marker_white) {
						code_pixel[i] = 0;
					}
					else {
						code_pixel[i] = 1;
						n1++;
					}
				} else {
					// return as uncoded marker
					uncompleat = true;
					break;
				}
			}

			// Check if non-coded marker
			if ((n1 < param.code_scan_resolution_per_element * 1.5) || uncompleat) {
				if (param.return_uncoded_marker) {
					temp_uncodedmarkers.push_back(e);
				}
				continue;
			}

			// Find first element (todo: find more robust version)
			int indexFirstElement = -1;
			for (unsigned int i = 1; i < n_scan_lines && indexFirstElement == -1; i++) {
				if (code_pixel[i - 1] == 0 && code_pixel[i] == 1) {
					indexFirstElement = i;
				}
			}
			if (indexFirstElement == -1) {
				if (param.return_uncoded_marker) {
					temp_uncodedmarkers.push_back(e);
				}
				continue;
			}

			// Reduction to Code bit starting with first element
			unsigned int max_code_error = ceil(double(param.code_scan_resolution_per_element) / 2.0) - 1;
			bool error = false;
			for (int i = 0; i < codeBit && !error; i++) {
				unsigned int v = 0;
				for (unsigned int j = 0; j < param.code_scan_resolution_per_element; j++) {
					unsigned int index = indexFirstElement + i * param.code_scan_resolution_per_element + j;
					index = (index < n_scan_lines) ? index : index - n_scan_lines;
					v += code_pixel[index];
				}

				// Check if clear code
				if (v > param.code_scan_resolution_per_element - max_code_error)
					code_string[i] = '1';
				else if (v < max_code_error)
					code_string[i] = '0';
				else
					error = true;
			}
			if (error) {
				if (param.return_uncoded_marker) {
					temp_uncodedmarkers.push_back(e);
				}
				continue;
			}

			// Decoding number
			int point_id = std::stoi(code_string, nullptr, 2);
			std::string code2 = code_string + code_string;
			for (int i = 1; i < codeBit; i++) {
				int _p = std::stoi(code2.substr(i, codeBit), nullptr, 2);
				if (point_id > _p)
					point_id = _p;
			}

			// Save if id in look up table
			e.id = findValue(point_id);
			if (std::find(found_ids.begin(), found_ids.end(), e.id) != found_ids.end())
				continue;
			if (e.id > 0) {
				markers.push_back(e);
				found_ids.push_back(e.id);
			}
			else if (param.return_uncoded_marker) {
				temp_uncodedmarkers.push_back(e);
			}
		}

		// Add uncoded markers to markers if not close to other coded marker
		for (auto const& ucm : temp_uncodedmarkers) {
			bool skip = false;
			for (auto const& cm : markers) {
				double lim = (cm.id < 0) ? cm.a * param.min_distance_closest_point : cm.a * 3 * param.min_distance_closest_point;
				if (abs(ucm.x - cm.x) < lim && abs(ucm.y - cm.y) < lim)
					skip = true;
			}

			if (!skip) {
				markers.push_back(ucm);
			}
		}
	}

	float getSubPixValue(const cv::Mat& image, const cv::Point2f& point)
	{
		// Bilinear interpolation
		int x0 = (int)point.x;
		int y0 = (int)point.y;

		float a = point.x - float(x0);
		float c = point.y - float(y0);

		return (float(image.at<uchar>(y0, x0)) * (1.f - a) + float(image.at<uchar>(y0, x0 + 1)) * a) * (1.f - c)
			+ (float(image.at<uchar>(y0 + 1, x0)) * (1.f - a) + float(image.at<uchar>(y0 + 1, x0 + 1)) * a) * c;
	}

	bool getSubPixValueWithChecks(const cv::Mat& image, const cv::Point2f& point, float& value)
	{
		// Check if point outside of image
		if ((int)point.x < 0 || (int)point.x > image.cols - 2 ||
			(int)point.y < 0 || (int)point.y > image.rows - 2)
			return false;

		value = getSubPixValue(image, point);
		return true;
	}

	template<typename T>
	void momentPreservation(const std::vector<T>& values, float& pos, float& h1, float& h2)
	{
		// ToDo: check math!
		pos = 0, h1 = 0, h2 = 0;
		if (values.size() < 5)
			return;
		double sum = 0, sum2 = 0, sum3 = 0;
		T last = values[0];
		T step = T(0);
		for (auto const& v : values) {
			sum += v;
			sum2 += v * v;
			sum3 += v * v * v;

			step += (v - last);
			last = v;
		}

		// Moments
		double m1, m2, m3;
		double size = double(values.size());
		m1 = sum / size;
		m2 = sum2 / size;
		m3 = sum3 / size;

		double sigma = sqrt(m2 - m1 * m1);

		if (sigma > 0) {
			double s = (m3 + 2.0 * (m1 * m1 * m1) - 3 * m1 * m2) / (sigma * sigma * sigma);
			double p1 = (1.0 + s * sqrt(1.0 / (4.0 + s * s))) / 2.0;
			double p2 = 1.0 - p1;
			pos = size * p1 - 0.5;
			double pos2 = size * p2 - 0.5;
			double hh1 = m1 - sigma * sqrt(p2 / p1);
			double hh2 = m1 + sigma * sqrt(p1 / p2);

			if (step < T(0)) {
				pos = pos2;
				h1 = hh2;
				h2 = hh1;
			}
			else {
				h1 = hh1;
				h2 = hh2;
			}
		}
	}
	// nessesary?
	template void momentPreservation<int>(const std::vector<int>& values, float& x, float& min, float& max);
	template void momentPreservation<float>(const std::vector<float>& values, float& x, float& min, float& max);
	template void momentPreservation<double>(const std::vector<double>& values, float& x, float& min, float& max);

	Ellipse fitEllipse(cv::InputArray& edge_points, int type, bool robust)
	{
		cv::RotatedRect box;
		switch (type) {
		case 0:
			box = cv::fitEllipse(edge_points);
			break;
		case 1:
			box = cv::fitEllipseAMS(edge_points);
			break;
		case 2:
			box = cv::fitEllipseDirect(edge_points);
			break;
		default:
			box = cv::fitEllipse(edge_points);
		}

		// Ellipse (box.height allways larger than box.width)
		Ellipse e = Ellipse(box.center.x, box.center.y,
			box.size.height / 2.0, box.size.width / 2.0,
			box.angle * CV_PI / 180.0);

		// Performe robust fit
		if (robust)
			fitEllipseRobust(e,edge_points,type);

		return e;
	}

    bool fitEllipseRobust(Ellipse &ellipse, cv::InputArray &edge_points, int type)
    {
		// see:  https://doi.org/10.1016/j.isprsjprs.2021.04.010
		cv::Mat _points = edge_points.getMat();
		const cv::Point* ptsi = _points.ptr<cv::Point>();
		const cv::Point2f* ptsf = _points.ptr<cv::Point2f>();
		bool isFloat = edge_points.depth() == CV_32F;
		int n_points = _points.size().width;

		for (int itt = 0; itt < 10; itt++) {
			// transform points to circle
			cv::Mat t1 = (cv::Mat_<double>(2, 2) << 1 / ellipse.b, 0, 0, 1 / ellipse.a);
			cv::Mat t2 = (cv::Mat_<double>(2, 2) << cos(ellipse.angle), sin(ellipse.angle), -sin(ellipse.angle), cos(ellipse.angle));
			std::vector<cv::Point2f> circle_points;
			circle_points.reserve(n_points); // Pre-allocate memory
			std::vector<double> dist;
			dist.reserve(n_points);
			for (int j = 0; j < n_points; j++) {
				cv::Point2f p = isFloat ? ptsf[j] : cv::Point2f((float)ptsi[j].x, (float)ptsi[j].y);
				cv::Mat cp = t1 * t2 * (cv::Mat_<double>(2, 1) << p.x - ellipse.x, p.y - ellipse.y);
				circle_points.emplace_back(cp.at<double>(0), cp.at<double>(1));
				dist.push_back(norm(circle_points.back())); 
			}

			double median = marker_detection::median(dist);
			std::vector<double> diff;
			for (auto const& p : dist) {
				diff.push_back(abs(p - median));
			}
			double median_dif = marker_detection::median(diff);
			double madn = median_dif / 0.67499;

			std::vector<cv::Point2f> inl_points;
			for (int j = 0; j < circle_points.size(); j++) {
				if (abs(dist[j] - median) / madn <= 2.45) {
					inl_points.push_back(isFloat ? ptsf[j] : cv::Point2f((float)ptsi[j].x, (float)ptsi[j].y));
				}
			}

			if (inl_points.size() == circle_points.size()) {
				ellipse.mdan = madn;
				break;
			}

			if (inl_points.size() < 5)
				break;

			// Fit ellipse
			cv::RotatedRect box;
			switch (type) {
			case 0:
				box = cv::fitEllipse(inl_points);
				break;
			case 1:
				box = cv::fitEllipseAMS(inl_points);
				break;
			case 2:
				box = cv::fitEllipseDirect(inl_points);
				break;
			default:
				box = cv::fitEllipse(inl_points);
			}
			ellipse = Ellipse(box.center.x, box.center.y,
				box.size.height / 2.0, box.size.width / 2.0,
				box.angle * CV_PI / 180.0);
			ellipse.mdan = madn;
		}
        return true;
    }

    bool starOperator(const cv::Mat& image, const Ellipse& in, Ellipse& out,
		Parameter param, double marker_value, double marker_around)
	{
		float min_radius = 0.5;
		float max_radius = 1.5;

		int scan_line_steps = ceil(in.a * 2.0); //~0,5 pixel
		if (scan_line_steps < 5) {
			scan_line_steps = 5;
		}

		out = in;
		double scan_angle_step = (CV_PI * 2.0) / double(param.sub_pixel_scan_lines);

		// Scanning around marker
		for (int itt = 0; itt < param.sub_pixel_iterations; itt++) {
			std::vector<cv::Point2f> edgePoints;
			cv::Point2f point = cv::Point2f();
			double alpha = sin(out.angle - CV_PI / 2.0);
			double beta = cos(out.angle - CV_PI / 2.0);
			for (unsigned int i = 0; i < param.sub_pixel_scan_lines; i++) {
				double scan_angle = double(i) * scan_angle_step;
				float xPos = out.a * cos(scan_angle);
				float yPos = out.b * sin(scan_angle);

				std::vector<double> scanline;
				for (int j = 0; j < scan_line_steps; j++) {
					double radius = min_radius + double(j) * (double(max_radius - min_radius) / double(scan_line_steps));
					point.x = out.x + xPos * beta * radius - yPos * alpha * radius;
					point.y = out.y + xPos * alpha * radius + yPos * beta * radius;
					float v;
					if (getSubPixValueWithChecks(image, point, v))
						scanline.push_back(v);
				}

				float pos, h1, h2;
				momentPreservation(scanline, pos, h1, h2);

				// Vis
				if (param.debug_vis >= 3) {
					cv::Mat copy;
					image.copyTo(copy);
					cv::Point point1, point2;
					point1.x = out.x + xPos * beta * min_radius - yPos * alpha * min_radius;
					point1.y = out.y + xPos * alpha * min_radius + yPos * beta * min_radius;
					point2.x = out.x + xPos * beta * max_radius - yPos * alpha * max_radius;
					point2.y = out.y + xPos * alpha * max_radius + yPos * beta * max_radius;
					cv::line(copy, point1, point2, cv::Scalar(127));

					int buffer = 100;
					int x = (out.x - buffer) < 0 ? 0 : out.x - buffer;
					int y = (out.y - buffer) < 0 ? 0 : out.y - buffer;
					int w = x + 2 * buffer >= image.cols ? image.cols - x : 2 * buffer;
					int h = y + 2 * buffer >= image.rows ? image.rows - y : 2 * buffer;
					cv::namedWindow("debugView", cv::WINDOW_GUI_EXPANDED);
					cv::imshow("debugView", copy(cv::Rect(x, y, w, h)));
					cv::waitKey();
				}

				double min_contrast = abs(marker_value - marker_around) * param.marker_contrast_consistency;
				if (abs(h1 - h2) > min_contrast && pos > 0) {
					double radius = min_radius + 1.0 / double(scanline.size()) * (pos); // -1???
					point.x = out.x + xPos * beta * radius - yPos * alpha * radius;
					point.y = out.y + xPos * alpha * radius + yPos * beta * radius;
					edgePoints.push_back(point);
				}
			}

			if (edgePoints.size() > (double(param.sub_pixel_scan_lines) * 0.7) && edgePoints.size() > 4) {
				out = marker_detection::fitEllipse(edgePoints, param.ellipse_fit_type, param.robust_ellipse_fit);

				// Vis
				if (param.debug_vis >= 2) {
					cv::Mat copy;
					image.copyTo(copy);
					for (auto const& c : edgePoints) {
						copy.at<uchar>(c.y, c.x) = 255;
					}
					int buffer = 100;
					int x = (out.x - buffer) < 0 ? 0 : out.x - buffer;
					int y = (out.y - buffer) < 0 ? 0 : out.y - buffer;
					int w = x + 2 * buffer >= image.cols ? image.cols - x : 2 * buffer;
					int h = y + 2 * buffer >= image.rows ? image.rows - y : 2 * buffer;
					cv::namedWindow("debugView", cv::WINDOW_GUI_EXPANDED);
					cv::imshow("debugView", copy(cv::Rect(x, y, w, h)));
					cv::waitKey();
				}
			}
			else {
				// Vis
				if (param.debug_vis >= 2) {
					cv::Mat copy;
					image.copyTo(copy);
					for (auto const& c : edgePoints) {
						copy.at<uchar>(c.y, c.x) = 255;
					}
					int buffer = 100;
					int x = (out.x - buffer) < 0 ? 0 : out.x - buffer;
					int y = (out.y - buffer) < 0 ? 0 : out.y - buffer;
					int w = x + 2 * buffer >= image.cols ? image.cols - x : 2 * buffer;
					int h = y + 2 * buffer >= image.rows ? image.rows - y : 2 * buffer;
					cv::namedWindow("debugView", cv::WINDOW_GUI_EXPANDED);
					cv::imshow("debugView", copy(cv::Rect(x, y, w, h)));
					cv::waitKey();
				}

				return false;
			}
		}
		return true;
	}

	void debugView(cv::Mat image, std::vector<cv::Point> points, Ellipse e, std::string error)
	{
		cv::Mat c;
		image.copyTo(c);
		cv::cvtColor(image, c, cv::COLOR_GRAY2BGR);
		if (points.size() == 0)
			cv::ellipse(c, cv::Point2f(e.x, e.y), cv::Size2f(e.b, e.a), e.angle / CV_PI * 180.0, 0, 360, cv::Scalar(120), 1);
		for (auto const& p : points) {
			c.at<cv::Vec3b>(p) = cv::Vec3b(255, 0, 0);
		}
		cv::drawMarker(c, e.point(), cv::Scalar(0, 255, 255));
		int buffer = 100;
		int x = (e.x - buffer) < 0 ? 0 : e.x - buffer;
		int y = (e.y - buffer) < 0 ? 0 : e.y - buffer;
		int w = (e.x + buffer) >= image.cols ? image.cols - e.x : 2 * buffer;
		int h = (e.y + buffer) >= image.rows ? image.rows - e.y : 2 * buffer;
		std::cerr << "Debug Error: " << error << std::endl;
		cv::namedWindow("debugView", cv::WINDOW_GUI_EXPANDED);
		cv::imshow("debugView", c(cv::Rect(x, y, w, h)));
		cv::waitKey();
	}

	void findConnectedEdgePoints(const cv::Mat& image, std::vector<std::vector<cv::Point> >& markers, Parameter param)
	{
		// Bluring image
		cv::Mat _image;
		if (param.median_blur_kernel > 2) {
			if (param.median_blur_kernel % 2 == 0)
				param.median_blur_kernel++;
			cv::medianBlur(image, _image, param.median_blur_kernel);
		}
		if (param.blur_kernel > 2) {
			if (param.blur_kernel % 2 == 0)
				param.blur_kernel++;
			cv::blur(image, _image, cv::Size(param.blur_kernel, param.blur_kernel));
		}
		if (_image.empty())
			_image = image;

		// Threshold
		cv::Mat threshold;
		if (param.threshold_method == 0) {
			// Just Otsu threshold
			cv::threshold(_image, threshold, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
		}
		else if (param.threshold_method == 1) {
			// Bradley adaptive threshold
			bradley_adaptive_thresholding(_image, threshold);
		}
		else if (param.threshold_method == 2) {
			// Normalize image with Clahe and Otsu threshold
			cv::Mat norm;
			cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
			clahe->setClipLimit(param.clahe_clip_limit);
			clahe->apply(_image, norm);
			cv::threshold(norm, threshold, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
		}
		else if (param.threshold_method == 3) {
			// Normalize image and Otsu threshold
			// copied from: https://stackoverflow.com/questions/14872306/local-normalization-in-opencv
			// Guanglei Xiong (2021). Local Normalization (https://www.mathworks.com/matlabcentral/fileexchange/8303-local-normalization), MATLAB Central File Exchange. Retrieved August 24, 2021.
			// convert to floating-point image
			cv::Mat float_gray, num, den, blur, diff, norm;
			_image.convertTo(float_gray, CV_32F, 1.0 / 255.0);

			// numerator = img - gauss_blur(img)
			cv::GaussianBlur(float_gray, blur, cv::Size(0, 0), param.sigma_1, 0);
			num = float_gray - blur;

			// denominator = sqrt(gauss_blur(img^2))
			cv::GaussianBlur(num.mul(num), blur, cv::Size(0, 0), param.sigma_2, 0);
			cv::pow(blur, 0.5, den);

			// output = numerator / denominator
			diff = num / den;

			// normalize output and threshold
			cv::normalize(diff, norm, 0, 255, cv::NORM_MINMAX, CV_8U);
			cv::threshold(norm, threshold, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
		}
		else {
			threshold = _image;
		}

		// Find edge
		cv::Mat edge;
		if (param.edge_method == 1) {
			if (param.blur_kernel % 2 == 0)
				param.blur_kernel++;
			cv::adaptiveThreshold(threshold, edge, 255, param.adaptive_threshold_method, cv::THRESH_BINARY, param.adaptive_threshold_block_size, param.adaptive_threshold_C);

			// invert if markers are black and background is white
			if (cv::mean(edge)[0] > 127)
				cv::bitwise_not(edge, edge);

			// thinning
			// copy from https://github.com/opencv/opencv_contrib/blob/4.x/modules/ximgproc/src/thinning.cpp
			int  thinningType = 0;
			cv::Mat processed = edge.clone();
			CV_CheckTypeEQ(processed.type(), CV_8UC1, "");
			processed /= 255;
			cv::Mat prev = cv::Mat::zeros(processed.size(), CV_8UC1);
			cv::Mat diff;
			do {
				thinningIteration(processed, 0, thinningType);
				thinningIteration(processed, 1, thinningType);
				absdiff(processed, prev, diff);
				processed.copyTo(prev);
			} while (countNonZero(diff) > 0);

			processed *= 255;
			edge = processed;
		}
		else {
			cv::Canny(threshold, edge, param.canny_threshold1, param.canny_threshold2);
		}

		// Connected components
		cv::Mat lable;
		int nComponents = cv::connectedComponents(edge, lable);
		markers.resize(nComponents);
		for (int r = 0; r < lable.rows; r++) {
			int* ptr = lable.ptr<int>(r);
			for (int c = 0; c < lable.cols; c++) {
				int* l = ptr++;
				if (*l > 0)
					markers[*l].push_back(cv::Point(c, r));
			}
		}

		if (param.debug_vis >= 1) {
			cv::namedWindow("image", cv::WINDOW_GUI_EXPANDED);
			cv::imshow("image", image);
			cv::namedWindow("threshold", cv::WINDOW_GUI_EXPANDED);
			cv::imshow("threshold", threshold);
			cv::namedWindow("edge", cv::WINDOW_GUI_EXPANDED);
			cv::imshow("edge", edge);
			cv::waitKey();
		}

		// ----- Other posible methods/notes -----
		// just THRESH_OTSU (without canny) is causing slightly wrong size markers
		// cv::adaptiveThreshold(image,diff,255,cv::ADAPTIVE_THRESH_GAUSSIAN_C,cv::ThresholdTypes::THRESH_BINARY,3,5);
		// cv::normalize(diff, norm,0,255,cv::NormTypes::NORM_MINMAX);
		// cv::Ptr<cv::SimpleBlobDetector>  detector = cv::SimpleBlobDetector::create();
		// std::vector<cv::KeyPoint> keypoints;
		// detector->detect(image, keypoints);
		// cv::findContours(binaryImage, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE); // causing double contours
	}

	bool zhouOperator(const cv::Mat& image, const Ellipse& in, Ellipse& out,
		double marker_value, double marker_around,
		const Parameter& param)
	{
		double bandwidth = 0.75;
		unsigned int numberOfSteps = 120;
		double min_contrast = abs(marker_value - marker_around) * 0.75;

		// vertical: key=y, value=x, horizontal: key=x, value=y
		std::map<int, double> coor_upper, coor_lower, coor_left, coor_right;
		out = in;
		cv::Point2i point = cv::Point2i();
		double scan_angle_step = (CV_PI * 2.0) / double(numberOfSteps);

		for (int itt = 0; itt < param.sub_pixel_iterations; itt++) {
			double alpha = sin(out.angle - CV_PI / 2.0);
			double beta = cos(out.angle - CV_PI / 2.0);

			for (unsigned int a = 0; a < numberOfSteps; a++) {
				// Calculate Point
				double scan_angle = double(a) * scan_angle_step;
				float xPos = out.a * cos(scan_angle);
				float yPos = out.b * sin(scan_angle);
				point.x = round(out.x + xPos * beta - yPos * alpha);
				point.y = round(out.y + xPos * alpha + yPos * beta);

				if (point.x < 0 || point.y < 0 ||
					point.x >= image.cols || point.y >= image.rows)
					return false;

				// -- Row
				int bandwidth_p = round(abs(point.x - out.x) * bandwidth);
				auto mappos = (point.x < out.x) ? &coor_left : &coor_right;
				if (bandwidth_p > 2 && mappos->find(point.y) == mappos->end()) {
					// Check if on edge
					if (point.x - bandwidth_p < 0 || point.x + bandwidth_p >= image.cols)
						return false;

					cv::Mat row = image.row(point.y);
					std::vector<int> values;
					for (int v = point.x - bandwidth_p; v < point.x + bandwidth_p; v++) {
						values.push_back(row.at<uchar>(v));
					}

					float h_pos, h_h1, h_h2;
					momentPreservation(values, h_pos, h_h1, h_h2);

					if (abs(h_h1 - h_h2) > min_contrast) {
						mappos->insert(std::make_pair(point.y, point.x + h_pos - double(bandwidth_p) /*- 1.0*/));
					}

					if (param.debug_vis > 2) {
						cv::Mat copy;
						image.copyTo(copy);
						cv::Point2d startPoint = cv::Point2d(point.x - double(bandwidth_p), point.y);
						cv::Point2d endPoint = cv::Point2d(point.x + double(bandwidth_p), point.y);
						cv::line(copy, startPoint, endPoint, cv::Scalar(50), 1);
						cv::drawMarker(copy, cv::Point2d(point.x + h_pos - double(bandwidth_p) - 1.0, point.y), cv::Scalar(255), cv::MARKER_TILTED_CROSS, 2);
						int buffer = 100;
						int x = (out.x - buffer) < 0 ? 0 : out.x - buffer;
						int y = (out.y - buffer) < 0 ? 0 : out.y - buffer;
						int w = (out.x + buffer) >= image.cols ? image.cols - out.x + buffer : 2 * buffer;
						int h = (out.y + buffer) >= image.rows ? image.rows - out.y + buffer : 2 * buffer;
						cv::namedWindow("debugView", cv::WINDOW_GUI_EXPANDED);
						cv::imshow("debugView", copy(cv::Rect(x, y, w, h)));
						cv::waitKey();
					}
				}

				// -- Col
				bandwidth_p = round(abs(point.y - out.y) * bandwidth);
				mappos = (point.y < out.y) ? &coor_upper : &coor_lower;
				if (bandwidth_p > 2 && mappos->find(point.y) == mappos->end()) {
					if (point.y - bandwidth_p < 0 || point.y + bandwidth_p >= image.rows)
						return false;

					cv::Mat col = image.col(point.x);
					std::vector<int> values;
					for (int v = point.y - bandwidth_p; v < point.y + bandwidth_p; v++) {
						values.push_back(col.at<uchar>(v));
					}

					float v_pos, v_h1, v_h2;
					momentPreservation(values, v_pos, v_h1, v_h2);

					if (abs(v_h1 - v_h2) > min_contrast) {
						mappos->insert(std::make_pair(point.x, point.y + v_pos - double(bandwidth_p)/*- 1.0*/));
					}

					if (param.debug_vis > 2) {
						cv::Mat copy;
						image.copyTo(copy);
						cv::Point2d startPoint = cv::Point2d(point.x, point.y - double(bandwidth_p));
						cv::Point2d endPoint = cv::Point2d(point.x, point.y + double(bandwidth_p));
						cv::line(copy, startPoint, endPoint, cv::Scalar(50), 1);
						cv::drawMarker(copy, cv::Point2d(point.x, point.y + v_pos - double(bandwidth_p) - 1.0), cv::Scalar(255), cv::MARKER_TILTED_CROSS, 2);
						int buffer = 100;
						int x = (out.x - buffer) < 0 ? 0 : out.x - buffer;
						int y = (out.y - buffer) < 0 ? 0 : out.y - buffer;
						int w = (out.x + buffer) >= image.cols ? image.cols - out.x + buffer : 2 * buffer;
						int h = (out.y + buffer) >= image.rows ? image.rows - out.y + buffer : 2 * buffer;
						cv::namedWindow("debugView", cv::WINDOW_GUI_EXPANDED);
						cv::imshow("debugView", copy(cv::Rect(x, y, w, h)));
						cv::waitKey();
					}
				}
			}

			// Calc vertical middle points and fit line
			std::vector<cv::Point2d> middle;
			std::vector<float> v_line, h_line;
			for (auto const& v : coor_left) {
				auto right = coor_right.find(v.first);
				if (right != coor_right.end())
					middle.push_back(cv::Point2d((v.second + right->second) / 2.0, double(v.first)));
			}
			if (middle.size() < 4)
				return false;
			cv::fitLine(middle, v_line, cv::DIST_L2, 0, 0.001, 0.001);

			// Calc horizontal middle points and fit line
			middle.clear();
			for (auto const& v : coor_upper) {
				auto lower = coor_lower.find(v.first);
				if (lower != coor_lower.end())
					middle.push_back(cv::Point2d(double(v.first), (v.second + lower->second) / 2.0));
			}
			if (middle.size() < 4)
				return false;
			cv::fitLine(middle, h_line, cv::DIST_L2, 0, 0.001, 0.001);

			if (h_line.size() < 4 || v_line.size() < 4)
				return false;

			// line intersection
			// https://stackoverflow.com/questions/7446126/opencv-2d-line-intersection-helper-function
			cv::Point2f x = cv::Point2f(h_line[2], h_line[3]) - cv::Point2f(v_line[2], v_line[3]);
			float cross = v_line[0] * h_line[1] - v_line[1] * h_line[0];
			if (abs(cross) > /*EPS*/1e-8) {
				double t1 = (x.x * h_line[1] - x.y * h_line[0]) / cross;
				cv::Point2f center = cv::Point2f(v_line[2], v_line[3]) + cv::Point2f(v_line[0], v_line[1]) * t1;

				if (abs(center.x - out.x) > out.a || abs(center.y - out.y) > out.a)
					return false;

				out.x = center.x;
				out.y = center.y;

				// Vis
				// cv::Mat copy;
				// image.copyTo(copy);
				// for (auto const &c : coor_left){
				//     copy.at<uchar>(c.first,c.second) = 255;
				// }
				// for (auto const &c : coor_right){
				//     copy.at<uchar>(c.first,c.second) = 255;
				// }
				// for (auto const &c : coor_lower){
				//     copy.at<uchar>(c.second,c.first) = 255;
				// }
				// for (auto const &c : coor_upper){
				//     copy.at<uchar>(c.second,c.first) = 255;
				// }
				// double t = std::max(copy.cols,copy.rows);
				// cv::Point2d startPoint, endPoint;
				// startPoint.x = h_line[2]- t*h_line[0];// x0
				// startPoint.y = h_line[3] - t*h_line[1];// y0
				// endPoint.x = h_line[2]+ t*h_line[0];//x[1]
				// endPoint.y = h_line[3] + t*h_line[1];//y[1]
				// cv::line(copy,startPoint,endPoint,cv::Scalar(25),1);
				// startPoint.x = v_line[2]- t*v_line[0];// x0
				// startPoint.y = v_line[3] - t*v_line[1];// y0
				// endPoint.x = v_line[2]+ t*v_line[0];//x[1]
				// endPoint.y = v_line[3] + t*v_line[1];//y[1]
				// cv::line(copy,startPoint,endPoint,cv::Scalar(125),1);
				//
				// cv::drawMarker(copy,center,cv::Scalar(255),cv::MARKER_TILTED_CROSS,10);
				// int buffer = 100;
				// int x = (out.x-buffer)< 0 ? 0 : out.x-buffer;
				// int y = (out.y-buffer)< 0 ? 0 : out.y-buffer;
				// int w = (out.x+buffer)>= image.cols ? image.cols-out.x+buffer : 2*buffer;
				// int h = (out.y+buffer)>= image.rows ? image.rows-out.y+buffer : 2*buffer;
				// cv::namedWindow("debugView", cv::WINDOW_GUI_EXPANDED );
				// cv::imshow("debugView", copy(cv::Rect(x, y, w, h)));
				// cv::waitKey();

				return true;
			}
		}

		return false;
	}


	bool searchMarker(const cv::Mat& image, Ellipse& ell, Parameter& param) {
		// search in a star pattern for a contrast change (> min contrast in param)
		int search_lines = 10;
		std::vector<std::vector<uchar>> pixel_values(search_lines);
		std::vector<cv::Point> edge;
		cv::Point2i point = cv::Point2i();
		double scan_angle_step = (CV_PI * 2.0) / double(search_lines);

		for (unsigned int i = 0; i < search_lines; i++) {
			double sin_angle = sin(double(i) * scan_angle_step);
			double cos_angle = cos(double(i) * scan_angle_step);
			double search = true;
			for (int j = 1; j < param.marker_max_diameter / 2 && search; j++) {
				point.x = (double(j) * sin_angle) + ell.x;
				point.y = (double(j) * cos_angle) + ell.y;

				if (point.x < 0 || point.x >= image.cols || point.y < 0 || point.y >= image.rows) {
					search = false;
				}
				else {
					uchar v = image.at<uchar>(point);
					pixel_values[i].push_back(v);
					if (pixel_values[i].size() > param.marker_min_diameter) {
						if (abs(pixel_values[i][0] - v) > param.marker_min_contrast) {
							edge.push_back(point);
							search = false;
						}
					}
				}
			}
		}

		if (edge.size() < 6)
			return false;

		Ellipse out = fitEllipse(edge, param.ellipse_fit_type, param.robust_ellipse_fit);

		double marker_value, marker_rmse, surrounding_value, surrounding_rmse;
		check_marker_surrounding(out, image, marker_value, marker_rmse, surrounding_value, surrounding_rmse);

		// Contrast between marker and suroundings
		if (abs(marker_value - surrounding_value) < param.marker_min_contrast) {
			return false;
		}

		if (!subPixelMeasurment(image, out, out, param, marker_value, surrounding_value)) {
			return false;
		}

		// small/large ellipse
		if (out.a > param.marker_max_diameter || out.b < param.marker_min_diameter) {
			return false;
		}

		// Ratio
		if (out.a / out.b > param.max_ellipse_ratio) {
			return false;
		}

		if (abs(out.x - ell.x) > out.a / 2 || abs(out.y - ell.y) > out.a / 2) {
			return false;
		}

		ell = out;
		return true;
	}

	bool subPixelMeasurment(const cv::Mat& image, const Ellipse& in, Ellipse& out,
		Parameter& param, double marker_value, double marker_around)
	{
		switch (param.sub_pixel_method) {
		case 0:// No sub pixel measurment
			return true;
		case 1:
			return starOperator(image, in, out, param, marker_value, marker_around);
		case 2:
			return zhouOperator(image, in, out, marker_value, marker_around, param);
		case 3:
			return starOperator2(image, in, out, param);
		default:
			return true;
		}
	}

	void bradley_adaptive_thresholding(const cv::Mat& in, cv::Mat& out)
	{
		out = cv::Mat(in.size(), CV_8U);

		// rows -> height -> y
		int nRows = in.rows;
		// cols -> width -> x
		int nCols = in.cols;

		// create the integral image
		cv::Mat intImage;
		cv::integral(in, intImage);

		int S = MAX(nRows, nCols) / 8;
		double T = 0.15;

		// perform thresholding
		int s2 = S / 2;
		int x1, y1, x2, y2, count, sum;

		int* p_y1, * p_y2;
		const uchar* p_inputMat;
		uchar* p_outputMat;

		for (int i = 0; i < nRows; ++i)
		{
			y1 = i - s2;
			y2 = i + s2;

			if (y1 < 1)
			{
				y1 = 1;
			}
			if (y2 >= nRows)
			{
				y2 = nRows - 1;
			}
			//         p_y1 = intImage.ptr<int>(y1);
			p_y1 = intImage.ptr<int>(y1 - 1);
			p_y2 = intImage.ptr<int>(y2);
			p_inputMat = in.ptr<uchar>(i);
			p_outputMat = out.ptr<uchar>(i);

			for (int j = 0; j < nCols; ++j)
			{
				// set the SxS region
				x1 = j - s2;
				x2 = j + s2;

				if (x1 < 1)
				{
					x1 = 1;
				}
				if (x2 >= nCols)
				{
					x2 = nCols - 1;
				}

				count = (x2 - x1) * (y2 - y1);

				//             sum = p_y2[x2] - p_y1[x2] - p_y2[x1] + p_y1[x1];
				sum = p_y2[x2] - p_y1[x2] - p_y2[x1 - 1] + p_y1[x1 - 1];

				if (p_inputMat[j] * count < sum * (1.0 - T))
					p_outputMat[j] = 0;
				else
					p_outputMat[j] = 255;
			}
		}
	}

	bool on_edge(const Ellipse& e, const cv::Mat& image, double radius)
	{
		int x_sub = e.x - radius;
		int y_sub = e.y - radius;
		int size_sub = 2 * radius;
		return (x_sub < 0 || y_sub < 0 ||
			x_sub + size_sub >= image.cols ||
			y_sub + size_sub >= image.rows);
	}

	bool starOperator2(const cv::Mat& image, const Ellipse& in, Ellipse& out, Parameter param)
	{
		float min_radius = 0.5;
		float max_radius = 1.5;

		int scan_line_steps = ceil(in.a * 2.0); //~0,5 pixel
		if (scan_line_steps < 5) {
			scan_line_steps = 5;
		}

		out = in;
		double scan_angle_step = (CV_PI * 2.0) / double(param.sub_pixel_scan_lines);

		// Scanning around marker
		for (int itt = 0; itt < param.sub_pixel_iterations; itt++) {
			// Check if on edge
			//if (on_edge(out, image, out.a * 2.0))
			//	return false;

			std::vector<cv::Point2f> edgePoints;
			cv::Mat value = cv::Mat(param.sub_pixel_scan_lines, scan_line_steps, CV_8U);
			cv::Point2f point = cv::Point2f();
			double alpha = sin(out.angle - CV_PI / 2.0);
			double beta = cos(out.angle - CV_PI / 2.0);
			for (unsigned int i = 0; i < param.sub_pixel_scan_lines; i++) {
				double scan_angle = double(i) * scan_angle_step;
				float xPos = out.a * cos(scan_angle);
				float yPos = out.b * sin(scan_angle);

				for (int j = 0; j < scan_line_steps; j++) {
					double radius = min_radius + double(j) * (double(max_radius - min_radius) / double(scan_line_steps));
					point.x = out.x + xPos * beta * radius - yPos * alpha * radius;
					point.y = out.y + xPos * alpha * radius + yPos * beta * radius;
					float val;
					if (getSubPixValueWithChecks(image, point,val))
						value.at<uchar>(i, j) = val;
				}
			}

			cv::Mat sobelx, sobelxabs;
			cv::Sobel(value, sobelx, CV_64F, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
			sobelxabs = cv::abs(sobelx);

			double minVal, maxVal;
			cv::minMaxLoc(sobelxabs, &minVal, &maxVal);
			maxVal *= 0.2;

			double* p;
			for (int i = 0; i < sobelxabs.rows; i++) {
				// Find first edge
				p = sobelxabs.ptr<double>(i);
				unsigned int edge_loc = 0;
				for (int j = 1; j < sobelxabs.cols - 1; j++) {
					if (p[j] > maxVal && p[j + 1] < p[j] && p[j - 1] < p[j]) {
						edge_loc = j;
						break;
					}
				}
				if (edge_loc < 1)
					continue;

				// calculate location of edge
				double scan_angle = double(i) * scan_angle_step;
				float xPos = out.a * cos(scan_angle);
				float yPos = out.b * sin(scan_angle);
				double radius = min_radius + 1.0 / double(value.cols) * double(edge_loc); // -1???
				point.x = out.x + xPos * beta * radius - yPos * alpha * radius;
				point.y = out.y + xPos * alpha * radius + yPos * beta * radius;
				edgePoints.push_back(point);
			}

			if (edgePoints.size() > (double(param.sub_pixel_scan_lines) * 0.7) && edgePoints.size() > 4) {
				out = marker_detection::fitEllipse(edgePoints, param.ellipse_fit_type, param.robust_ellipse_fit);

				// Vis
				if (param.debug_vis >= 2) {
					cv::Mat copy;
					image.copyTo(copy);
					for (auto const& c : edgePoints) {
						copy.at<uchar>(c.y, c.x) = 255;
					}
					int buffer = 100;
					int x = (out.x - buffer) < 0 ? 0 : out.x - buffer;
					int y = (out.y - buffer) < 0 ? 0 : out.y - buffer;
					int w = (out.x + buffer) >= image.cols ? image.cols - out.x + buffer : 2 * buffer;
					int h = (out.y + buffer) >= image.rows ? image.rows - out.y + buffer : 2 * buffer;
					cv::namedWindow("debugView", cv::WINDOW_GUI_EXPANDED);
					cv::imshow("debugView", copy(cv::Rect(x, y, w, h)));
					cv::waitKey();
				}
			}
			else {
				return false;
			}
		}
		return true;
	}

	void check_marker_surrounding(const Ellipse& e, const cv::Mat& image, double& marker_value, double& marker_rmse, double& surrounding_value, double& surrounding_rmse)
	{
		marker_value = 0;
		marker_rmse = -1;
		surrounding_value = 0;
		surrounding_rmse = -1;
		double alpha = sin(e.angle - CV_PI * 0.5);
		double beta = cos(e.angle - CV_PI * 0.5);
		double r[4] = { 0.25,0.75,1.2,1.3 }; // addapt depending on the marker size, for markers > 1000 pixel
		std::vector<double> mar, out;
		for (int i = 0; i < 12; i++) {
			double a = (CV_PI * 2.0) / 12.0;
			double x = e.a * cos(double(i) * a);
			double y = e.b * sin(double(i) * a);
			for (int j = 0; j < 4; j++) {
				cv::Point2f point1 = e.point();
				point1.x += x * beta * r[j] - y * alpha * r[j];
				point1.y += x * alpha * r[j] + y * beta * r[j];
				if (point1.x < 1 || point1.y < 1)
					continue;
				if (point1.x >= image.cols - 1 || point1.y >= image.rows - 1)
					continue;

				float value;
				if (marker_detection::getSubPixValueWithChecks(image, point1, value)) {
					if (r[j] < 1.0)
						mar.push_back(value);
					else
						out.push_back(value);
				}
			}
		}
		if (mar.size() < 12 || out.size() < 12)
			return;

		marker_value = median(mar);
		marker_rmse = rmse(mar);
		surrounding_value = median(out);
		surrounding_rmse = rmse(out);
	}

	// Applies a thinning iteration to a binary image
	// copyed: https://github.com/opencv/opencv_contrib/blob/4.x/modules/ximgproc/src/thinning.cpp
	static void thinningIteration(cv::Mat img, int iter, int thinningType) {
		cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

		if (thinningType == 0) { // THINNING_ZHANGSUEN
			for (int i = 1; i < img.rows - 1; i++)
			{
				for (int j = 1; j < img.cols - 1; j++)
				{
					uchar p2 = img.at<uchar>(i - 1, j);
					uchar p3 = img.at<uchar>(i - 1, j + 1);
					uchar p4 = img.at<uchar>(i, j + 1);
					uchar p5 = img.at<uchar>(i + 1, j + 1);
					uchar p6 = img.at<uchar>(i + 1, j);
					uchar p7 = img.at<uchar>(i + 1, j - 1);
					uchar p8 = img.at<uchar>(i, j - 1);
					uchar p9 = img.at<uchar>(i - 1, j - 1);

					int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
						(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
						(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
						(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
					int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
					int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
					int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

					if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
						marker.at<uchar>(i, j) = 1;
				}
			}
		}
		if (thinningType == 1) { // THINNING_GUOHALL
			for (int i = 1; i < img.rows - 1; i++)
			{
				for (int j = 1; j < img.cols - 1; j++)
				{
					uchar p2 = img.at<uchar>(i - 1, j);
					uchar p3 = img.at<uchar>(i - 1, j + 1);
					uchar p4 = img.at<uchar>(i, j + 1);
					uchar p5 = img.at<uchar>(i + 1, j + 1);
					uchar p6 = img.at<uchar>(i + 1, j);
					uchar p7 = img.at<uchar>(i + 1, j - 1);
					uchar p8 = img.at<uchar>(i, j - 1);
					uchar p9 = img.at<uchar>(i - 1, j - 1);

					int C = ((!p2) & (p3 | p4)) + ((!p4) & (p5 | p6)) +
						((!p6) & (p7 | p8)) + ((!p8) & (p9 | p2));
					int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
					int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
					int N = N1 < N2 ? N1 : N2;
					int m = iter == 0 ? ((p6 | p7 | (!p9)) & p8) : ((p2 | p3 | (!p5)) & p4);

					if ((C == 1) && ((N >= 2) && ((N <= 3)) & (m == 0)))
						marker.at<uchar>(i, j) = 1;
				}
			}
		}

		img &= ~marker;
	}

	double median(std::vector<double> values)
	{
		unsigned int size = values.size();
		double median = 0.0;
		if (size > 2) {
			std::sort(values.begin(), values.end());
			median = size % 2 == 0 ? (values[size / 2 - 1] + values[size / 2]) / 2 : values[size / 2];
		}
		return median;
	}

	double rmse(std::vector<double>& values)
	{
		if (values.empty())
			return 0.0;

		double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
		double sum_squared_diff = 0.0;

		for (const double& value : values) {
			sum_squared_diff += std::pow(value - mean, 2);
		}

		double mean_squared_diff = sum_squared_diff / values.size();
		return std::sqrt(mean_squared_diff);	}

} // end namespace marker_detection

