#include "Projection.h"

#include <opencv2/highgui.hpp>

Projection::Projection(int gantry_index, std::string image_name) : gantry_index(gantry_index), m_image_name(image_name), projection_rmseD(-1.0) { }

Projection::~Projection() { }

void Projection::generateThumbnail(cv::Mat& image) {
	cv::resize(image, m_thumbnail, cv::Size(), scale, scale, cv::InterpolationFlags::INTER_AREA);
}
cv::Mat Projection::getThumbnail() {
	if (m_thumbnail.empty()) {
		getImage();
	}
	return m_thumbnail;
}
cv::Mat Projection::getImage(bool normalize_8bit, bool generate_thumbnail) {
	cv::Mat _image = cv::imread(m_image_path, cv::IMREAD_ANYDEPTH);
	if (_image.empty())
		return cv::Mat();

	if (normalize_8bit)
	{
		cv::normalize(_image, _image, 0, 255, cv::NormTypes::NORM_MINMAX, CV_8U);

		// Normalizing image to different maximum value does not increase accuracy
		//// Convert the image pixels to a linear array
		//std::vector<ushort> pixels;
		//pixels.assign((ushort*)_image.datastart, (ushort*)_image.dataend);
		//
		//// Sort the pixel values
		//std::sort(pixels.begin(), pixels.end());
		//
		//// Find the median
		//double median;
		//size_t size = pixels.size();
		//if (size % 2 == 0) {
		//	median = (pixels[size / 2 - 1] + pixels[size / 2]) / 2.0;
		//}
		//else {
		//	median = pixels[size / 2];
		//}
		//
		//// Normalize the image within the mask
		//_image.convertTo(_image, CV_8U, 255.0 / (median - pixels[0]), pixels[0] * 255.0 / (pixels[0] - median));
	}

	if (generate_thumbnail)
		generateThumbnail(_image);
	return _image;
}

//Write serialization for this class
void Projection::write(cv::FileStorage& fs) const
{
	fs << "{";
	fs << "image_name" << m_image_name;
	fs << "gantry_index" << gantry_index;
	fs << "image_size_x" << m_image_size_px.width;
	fs << "image_size_y" << m_image_size_px.height;
	fs << "marker" << "[";
	for (auto& m : observation) {
		fs << "{" << "id" << m.id << "x" << m.x << "y" << m.y << "a" << m.a << "b" << m.b << "angle" << m.angle << "}";
	}
	fs << "]";
	fs << "}";
}

//Read serialization for this class
void Projection::read(const cv::FileNode& node)
{
	m_image_name = (std::string)node["image_name"];
	gantry_index = (double)node["gantry_index"];
	m_image_size_px.width = (int)node["image_size_x"];
	m_image_size_px.height = (int)node["image_size_y"];
	cv::FileNode n = node["marker"];
	if (n.type() == cv::FileNode::SEQ)
	{
		cv::FileNodeIterator it = n.begin(), it_end = n.end();
		for (; it != it_end; ++it) {
			double x, y, a, b, angle, madn_ell;
			int id;
			(*it)["id"] >> id;
			(*it)["x"] >> x;
			(*it)["y"] >> y;
			(*it)["a"] >> a;
			(*it)["b"] >> b;
			(*it)["angle"] >> angle;
			observation.push_back(Observation(x, y, a, b, angle, id));
		}
	}
}

cv::Mat Projection::drawResiduals(cv::Size2i image_size, double fontSize, double lineWidth, double error_scale, double line_spacing, int font, cv::Scalar color, bool useThumbnailSize)
{
	// Get image
	double _scale = useThumbnailSize ? scale : 1.0;
	double eccentricity_scale = 100000;
	cv::Scalar ellipse_orientation_color = cv::Scalar(0, 200, 0);
	cv::Scalar eccentricity_color = cv::Scalar(255, 0, 0);
	cv::Mat image;
	if (image_size.area() < 1) {
		cv::cvtColor(useThumbnailSize ? getThumbnail() : getImage(), image, cv::COLOR_GRAY2BGR);
	}
	else {
		image = cv::Mat(image_size.height * scale, image_size.width * scale, CV_8UC3);
		image = cv::Scalar(255, 255, 255);
	}
	int thickness = (image.size().width > 1000) ? image.size().width / 1000.0 : 1;

	// Add markers and error lines
	double max_residual = -1;
	double max_eccentricity = -1;
	for (auto& m : observation) {
		cv::Point2d p = cv::Point2d(m.x * scale, m.y * scale);
		cv::Size2f  s = cv::Size2f(m.b * scale, (m.a * scale));
		if (m.id > 0) {
			std::string label = std::to_string(m.id);
			cv::putText(image, label, p, font, thickness, color, thickness);

			switch (m.point_typ)
			{
			case Observation::TYP::outlier:
				cv::drawMarker(image, p, color, cv::MARKER_CROSS, m.a * scale * 2.0, thickness);
				break;
			default:
				cv::ellipse(image, p, s, m.angle / CV_PI * 180.0, 0, 360, color, thickness);
				break;
			}

			if (m.residual_pixel) {
				cv::Point2d projPoint = cv::Point2d(
					p.x + m.residual_pixel.value().x * error_scale,
					p.y + m.residual_pixel.value().y * error_scale);
				cv::line(image, p, projPoint, color, thickness);

				double res_norm = cv::norm(m.residual_pixel.value());
				if (res_norm > max_residual)
					max_residual = res_norm;
			}

			if (m.eccentricity_correction) {
				cv::Point2d corrPoint = cv::Point2d(
					p.x - m.eccentricity_correction.value().x * eccentricity_scale,
					p.y - -m.eccentricity_correction.value().y * eccentricity_scale);
				cv::line(image, p, corrPoint, eccentricity_color, thickness);

				double ecc_norm = cv::norm(m.eccentricity_correction.value());
				if (ecc_norm > max_eccentricity)
					max_eccentricity = ecc_norm;
			}
		}
		else {
			cv::ellipse(image, p, s, m.angle / CV_PI * 180.0, 0, 360, cv::Scalar(0, 0, 255), thickness);
		}
		// Orientation
		cv::Point2f point1 = cv::Point2f(
			cos(m.angle + CV_PI * 0.5) * (m.a - m.b) * 400 + p.x,
			sin(m.angle + CV_PI * 0.5) * (m.a - m.b) * 400 + p.y);
		cv::Point2f point2 = cv::Point2f(
			cos(m.angle - CV_PI * 0.5) * (m.a - m.b) * 400 + p.x,
			sin(m.angle - CV_PI * 0.5) * (m.a - m.b) * 400 + p.y);
		cv::line(image, point1, point2, ellipse_orientation_color, thickness);
	}

	// Add text
	int baseline = 0;
	cv::Size tsize = cv::getTextSize("O", font, thickness, thickness, &baseline);
	cv::putText(image, "Index: " + std::to_string(round(gantry_index)), cv::Point(20, (tsize.height + 3) * 1), font, thickness, color, thickness);
	cv::putText(image, "Scale: " + std::to_string(error_scale), cv::Point(20, (tsize.height + 3) * 2), font, thickness, color, thickness);
	cv::putText(image, "RMSE: " + std::to_string(projection_rmseD) + " px", cv::Point(20, (tsize.height + 3) * 3), font, thickness, color, thickness);
	cv::putText(image, "Ellipse orientation", cv::Point(20, (tsize.height + 3) * 4), font, thickness, ellipse_orientation_color, thickness);
	if (max_residual > 0) {
		cv::putText(image, "Max residual: " + std::to_string(max_residual) + " px", cv::Point(20, (tsize.height + 3) * 5), font, thickness, color, thickness);
	}
	if (max_eccentricity > 0) {
		cv::putText(image, "Max eccentricity: " + std::to_string(max_eccentricity) + " mm", cv::Point(20, (tsize.height + 3) * 6), font, thickness, eccentricity_color, thickness);
	}

	return image;
}
