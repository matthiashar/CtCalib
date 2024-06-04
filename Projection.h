#ifndef PROJECTION_H
#define PROJECTION_H


#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <iostream>
#include <optional>

class Projection
{
public:
	struct Observation
	{
		enum TYP
		{
			default = 0,
			outlier = 1
		};
		double x, y, a, b, angle;
		int id;
		std::optional<cv::Point2d> point_metric;
		
		/// observation + residual = projection 
		std::optional<cv::Point2d> residual_pixel;
		
		std::optional<cv::Point2d> eccentricity_correction;
		TYP point_typ;
		Observation(double x, double y, double a, double b, double angle, int id = -1) :
			x(x), y(y), a(a), b(b), angle(angle), id(id) {
			point_typ = TYP::default;
		};
		cv::Point2d point() { return cv::Point2d{ x,y }; }
	};

	Projection(int gantry_index = -1, std::string image_name = "");
	~Projection();

	void generateThumbnail(cv::Mat& image);
	cv::Mat getThumbnail();
	cv::Mat getImage(bool normalize_8bit = true, bool generate_thumbnail = true);
	
	std::string getImagePath() const { return m_image_path; }
	std::string getImageName() const { return m_image_name; }
	cv::Size2i getImageSizePixel() const { return m_image_size_px; }

	void setImagePath(std::string path) { m_image_path = path; }
	void setImageName(std::string name) { m_image_name = name; }
	void setImageSizePixel(cv::Size2i size) { m_image_size_px = size; }

	void write(cv::FileStorage& fs) const;
	void read(const cv::FileNode& node);

	// vis
	cv::Mat drawResiduals(cv::Size2i image_size = cv::Size2i(0,0),
		double fontSize = 1.5,
		double lineWidth = 1,
		double error_scale = 600.0,
		double line_spacing = 1.5,
		int font = cv::FONT_HERSHEY_PLAIN,
		cv::Scalar color = cv::Scalar(0, 0, 255),
		bool useThumbnailSize = true);

	// data
	std::vector<Observation> observation;
	double gantry_index = 0;
	double projection_rmseD = 0, projection_rmseX = 0, projection_rmseY = 0;

private:
	double scale = 0.5;

	cv::Mat m_thumbnail;
	std::string m_image_path, m_image_name;
	cv::Size2i m_image_size_px;
};

//These write and read functions must be defined for the serialization in FileStorage to work
static void write(cv::FileStorage& fs, const std::string&, const Projection& x)
{
	x.write(fs);
}
static void read(const cv::FileNode& node, Projection& x, const Projection& default_value = Projection()) {
	if (node.empty())
		x = default_value;
	else
		x.read(node);
}

#endif // PROJECTION_H
