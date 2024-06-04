#include "markerdetection.h"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <opencv2/imgcodecs.hpp>


int subpixel_marker(int method, double tollerance = 0.01)
{
	cv::Mat img = cv::Mat::zeros(100, 100, CV_8U);
	//cv::ellipse(img, cv::Point2f(50,50), cv::Size2f(25,20), 25, 0, 360, cv::Scalar(255), -1);
	cv::circle(img, cv::Point2f(50, 50), 25, cv::Scalar(255), -1);
	marker_detection::Ellipse ell;
	ell.x = 49;
	ell.y = 52;
	marker_detection::Parameter param;
	param.sub_pixel_method = method;
	marker_detection::searchMarker(img, ell, param);
	assert(abs(ell.x - 50) < tollerance);
	assert(abs(ell.y - 50) < tollerance);
	return 0;
}

int subpixel_value() {

	cv::Mat img = cv::Mat::zeros(5, 5, CV_8U);
	img.at<uchar>(2, 2) = 255;
	assert(marker_detection::getSubPixValue(img, cv::Point2f(2, 2)) == 255);
	assert(marker_detection::getSubPixValue(img, cv::Point2f(1.5, 2)) == 127.5);
	assert(marker_detection::getSubPixValue(img, cv::Point2f(1.75, 2)) == 191.25);
	assert(marker_detection::getSubPixValue(img, cv::Point2f(1.5, 1.5)) == 63.75);
	return 0;
}

int moment_preservation() {
	std::vector<double> v = { 0,0,0,0,25,50,75,100,100,100,100 };
	float p, h1, h2;
	marker_detection::momentPreservation(v, p, h1, h2);
	assert(abs(p - 5) < 0.001);

	v = { 0,0,0,0,10,10,10,10 };
	marker_detection::momentPreservation(v, p, h1, h2);
	assert(abs(p - 3.5) < 0.001);

	v = { 10,10,10,10,1,1,1,1 };
	marker_detection::momentPreservation(v, p, h1, h2);
	assert(abs(p - 3.5) < 0.001);
	return 0;
}

int main()
{
	subpixel_value();
	moment_preservation();
	subpixel_marker(0,0.5);
	subpixel_marker(1);
	subpixel_marker(2);
	return 0;
}