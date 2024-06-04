#ifndef MARKERDETECTION_H
#define MARKERDETECTION_H

#include <map>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace marker_detection {

/**
 * @brief struct for ellipse
 * x, y = center
 * a, b = semi-major and semi-minor axis (a > b)
 * angle = angle in rad
 * id = point id, for unknown id == -1
 * madn = Normalized Median Absolute Deviation (will only be computed when robust_ellipse_fit = true)
 */
struct Ellipse
{
    Ellipse() : x(0.0), y(0.0), a(0.0), b(0.0), angle(0.0), id(-1) {}
    Ellipse(double x, double y, double a, double b, double angle) :
        x(x), y(y), a(a), b(b), angle(angle), id(-1), mdan(-1) {}
    double x, y, a, b, angle, mdan;
    int id;

    cv::Point2d point() const {
        return cv::Point2d(x, y);
    }
};

/**
 * @brief The Parameter struct
 *
 */
struct Parameter
{
    Parameter() :
        ellipse_fit_type(0),
        marker_min_diameter(4),
        marker_max_diameter(300),
        code_scan_resolution_per_element(5),
        code_ring_radius(2.5),
        marker_min_contrast(20),
        return_uncoded_marker(true),
        detect_coded_marker(true),
        robust_ellipse_fit(true),
        median_blur_kernel(5),
        blur_kernel(-1),
        max_ellipse_ratio(2.0),
        canny_threshold1(40),
        canny_threshold2(100),
        threshold_method(-1),
        clahe_clip_limit(5),
        sigma_1(5),
        sigma_2(25),
        sub_pixel_method(1),
        sub_pixel_scan_lines(50),
        sub_pixel_iterations(2),
        debug_vis(0),
        min_distance_closest_point(1.0),
        edge_method(0),
        adaptive_threshold_method(cv::ADAPTIVE_THRESH_GAUSSIAN_C),
        adaptive_threshold_C(3),
        adaptive_threshold_block_size(7),
        marker_contrast_consistency(0.60),
        max_marker_value_rmse(10),
        max_surrounding_value_rmse(100)
    {}

    /// Type of fiting ellipse: 0: cv::fitEllipse(), 1: cv::fitEllipseAMS(), 2: cv::fitEllipseDirect()
    int ellipse_fit_type;
    /// Minimum diameter of marker in pixel
    unsigned int marker_min_diameter;
    /// Maximum diameter of marker in pixel
    unsigned int marker_max_diameter;
    /// Number of measurments per code element (default: 5)
    unsigned int code_scan_resolution_per_element;
    /// Radius of code ring around marker, where marker radius=1 (default: 2.5)
    double code_ring_radius;
    /// Minimum contrast between marker and surrounding area
    unsigned int marker_min_contrast;
    /// Set true if uncoded markers shoud be returned
    bool return_uncoded_marker;
    /// Set true if coded markers shoud be searched for
    bool detect_coded_marker;
    /// If true outlier edge points will be removed during ellipse fitt
    bool robust_ellipse_fit;
    /// Kernel for median blur befor initial marker detection
    int median_blur_kernel;
    /// Kernel for blur befor initial marker detection
    int blur_kernel;
    /// Maximum ratio between minor and major axsis of ellipse (a/b)
    double max_ellipse_ratio;
    /// Canny threshold1 for edge detection
    unsigned int canny_threshold1;
    /// Canny threshold2 for edge detection
    unsigned int canny_threshold2;
    /// Threshold method (-1 - Non, 0 - Otsu, 1 - Bradley, 2 - Clahe, 3 - Xiong)
    int threshold_method;
    /// Clahe clip
    int clahe_clip_limit;
    /// Sigma 1 for Local Normalization
    int sigma_1;
    /// Sigma 2 for Local Normalization
    int sigma_2;
    /// Sup pixel method: 0 - non, 1 - star operator, 2 = zhou
    int sub_pixel_method;
    /// Number of sub pixel scan lines
    unsigned int sub_pixel_scan_lines;
    /// Number of iterations for adjusting sub pixel position
    int sub_pixel_iterations;
    /// Debugging visualization (0 = Non, 1 = Image, 2 = Marker, 3 = Detailed)
    int debug_vis;
    /// If true uncoded points will be returned if they are close to coded points
    double min_distance_closest_point;
    /// Method for extracting edge. (0 = Canny, 1 = cv::adaptiveThreshold())
    int edge_method;
    /// Parameter for cv::adaptiveThreshold(), see OpenCV documentation.
    int adaptive_threshold_method;
    /// Parameter for cv::adaptiveThreshold(), see OpenCV documentation.
    int adaptive_threshold_C;
    /// Parameter for cv::adaptiveThreshold(), see OpenCV documentation.
    int adaptive_threshold_block_size;
    /// Consistency of contrast at the edte of the marker. 0 = inconsistend contrast around marker, 1 = same contrast at edge of marker
    double marker_contrast_consistency;
    /// Maximum rmse of the marker pixel values
    double max_marker_value_rmse;
    /// Maximum rmse of the pixel values surrounding the marker
    double max_surrounding_value_rmse;
};

/**
 * @brief Fast method for detecting and decoding 14bit marker
 * @param image - Input: image (CV_8U)
 * @param markers - Output: detected markers
 */
void detectAndDecode(const cv::Mat &image, std::vector<Ellipse> &markers, Parameter param = Parameter());

/**
 * @brief Moment preservation (Luhmann 2018, S 453ff.) Note: The Center of the first Element in the Vector is 0.
 */
template <typename T>
void momentPreservation(const std::vector<T> &values, float &pos, float &h1, float &h2);

/**
 * @brief Bilinear interpolation of pixel value. Note: No checks are performed!
 * @param image - Image with one channel (CV_8U)
 * @param point - Subpixel point
 * @return interpolated pixel value
 */
float getSubPixValue(const cv::Mat &image, const cv::Point2f &point);

/**
 * @brief Bilinear interpolation of pixel value. Note: Same as getSubPixValue, but with checks.
 * @param image - Image with one channel (CV_8U)
 * @param point - Subpixel point
 * @param value - interpolated pixel value
 * @return true if success
 */
bool getSubPixValueWithChecks(const cv::Mat& image, const cv::Point2f& point, float &value);

/**
 * @brief Fitting points to ellipse using opencv methods.
 * @param edge_points - e.g. std::vector<cv::Point2f>
 * @param type - 0=cv::fitEllipse(), 1=cv::fitEllipseAMS(), 2=cv::fitEllipseDirect()
 * @param robust - outlier edge points will be removed during fit
 * @return adjusted ellipse
 */
Ellipse fitEllipse(cv::InputArray &edge_points, int type, bool robust);

/**
 * @brief Robust check if points are on ellipse (see: https://doi.org/10.1016/j.isprsjprs.2021.04.010)
 * @param initial_ellipse - initial ellipse, will be updated
 * @param edge_points - e.g. std::vector<cv::Point2f>
 * @param type - 0=cv::fitEllipse(), 1=cv::fitEllipseAMS(), 2=cv::fitEllipseDirect()
 * @return true if roubust adjustment was successful 
 */
bool fitEllipseRobust(Ellipse &ellipse, cv::InputArray &edge_points, int type);

/**
 * @brief subPixelMeasurment
 * @param image
 * @param in
 * @param out
 * @param param
 * @return
 */
bool subPixelMeasurment(const cv::Mat &image, const Ellipse &in,
                        Ellipse &out, Parameter &param,
                        double marker_value, double marker_around);

/**
 * @brief Subpixel marker detection using the star operator.
 * @param image
 * @param ellipse
 * @param sub_pixel_scan_lines
 * @param min_contrast
 * @return
 */
bool starOperator(const cv::Mat &image, const Ellipse &in, Ellipse &out,
                  Parameter param, double marker_value, double marker_around);

/**
 * @brief Different Method, Alpha
 * @param image
 * @param ellipse
 * @param sub_pixel_scan_lines
 * @return
 */
bool starOperator2(const cv::Mat &image, const Ellipse &in, Ellipse &out, Parameter param);

/**
 * @brief zhouOperator
 * @param image
 * @param in
 * @param out
 * @return
 */
bool zhouOperator(const cv::Mat &image, const Ellipse &in, Ellipse &out,
                  double marker_value, double marker_around,
                  const Parameter &param = marker_detection::Parameter());

/**
 * @brief Detect and retun connected edge points
 * @param image
 * @param markers
 * @param param
 */
void findConnectedEdgePoints(const cv::Mat &image, std::vector<std::vector<cv::Point>> &markers, Parameter param = Parameter());

/**
 * @brief Draws points or ellipse to image for debugging
 * @param image
 * @param markers
 */
void debugView(cv::Mat image, std::vector<cv::Point> points, Ellipse e = Ellipse(), std::string error = "");

/**
 * @brief Search marker at specific location in image and detect with sub-pixel accuracy
 * @param image
 * @param center_guess
 * @param param
 * @return true if marker was found
 */
bool searchMarker(const cv::Mat& image, Ellipse& ell, Parameter& param);

/**
 * @brief Adaptive Thresholding Meshode (D. Bradley and G. Roth, “Adaptive Thresholding using the Integral Image,” Journal of Graphics Tools, vol. 12, Art. no. 2, Jan. 2007, doi: 10.1080/2151237x.2007.10129236).
 * @param in
 * @param out
 */
void bradley_adaptive_thresholding(const cv::Mat &in, cv::Mat &out);

bool on_edge(const Ellipse &e, const cv::Mat &image, double radius);

void check_marker_surrounding(const Ellipse &e, const cv::Mat &image, double& marker_value, double& marker_rmse, double& surrounding_value, double& surrounding_rmse);

/**
 * @brief Applies a thinning iteration to a binary image. 
 * copy from: https://github.com/opencv/opencv_contrib/blob/4.x/modules/ximgproc/src/thinning.cpp
 */
static void thinningIteration(cv::Mat img, int iter, int thinningType);

/**
 * @brief Function for calculation the median.
 * @param values = vector of values
 * @return median
 */
double median(std::vector<double> values);

/**
 * @brief Function for calculation the rmse.
 * @param values = vector of values
 * @return rmse
 */
double rmse(std::vector<double>& values);

} // end namespace marker_detection

#endif // MARKERDETECTION_H
