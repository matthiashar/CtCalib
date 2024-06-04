#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "markerdetection.h"

int main(int argc, char* argv[]) {

    // Check for parameters
    if (argc == 1) {
        std::cerr << "usage: marker_detection <path to image folder> [<draw and show detected marker (0/1)>]" << std::endl;
        exit(1);
    }

    // Read image path
    std::string folderpath = argv[1];

    // Visualization
    int vis_debug = 0;
    if (argc == 3) {
        vis_debug = (std::atoi(argv[2]));
    }

    // Open output files
    std::ofstream resultfile_param(folderpath + "/image_points.txt");
    std::ofstream stats_file(folderpath + "/stats.txt");

    if (!stats_file.is_open() || !resultfile_param.is_open()) {
        std::cerr << "Error: Unable to write to " << folderpath << std::endl;
        exit(1);
    }
    resultfile_param << "image_name marker_id x y a b angle" << std::endl;
    stats_file << "image_name found_markers coded_markers time_needed_ms" << std::endl;

    // Iteration through images
    std::vector<cv::String> filenames;
    try {
        cv::glob(folderpath, filenames);
    } catch (const cv::Exception& e) {
        std::cerr << e.msg << std::endl;
        exit(1);
    }

    cv::Mat image, image_gray;
    int n_markers = 0;
    int n_images = 0;
    double time = 0;
    for (auto const &p : filenames) {
        image = cv::imread(p);
        if (image.empty()) {
            continue;
        }
        n_images++;

        // Convert to grayscale
        if (image.channels() == 3) {
            cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
        } else {
            image_gray = image;
        }

        // Find markers
        double t = (double)cv::getTickCount();
        std::vector<marker_detection::Ellipse> detectedMarkers;
        marker_detection::detectAndDecode(image_gray, detectedMarkers);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        time += t;

        // generating output files
        int codedM = 0;
        std::string file_name = p.substr(p.length() - (p.length() - folderpath.length() - 1));
        for (auto const &m : detectedMarkers) {
            resultfile_param << std::setprecision(4) << std::fixed
                             << file_name << " "
                             << m.id << " "
                             << m.x << " "
                             << m.y << " "
                             << m.a << " "
                             << m.b << " "
                             << m.angle << " "
                             << m.mdan << std::endl;
            if (m.id != -1) {
                codedM++;
            }
        }

        stats_file << file_name << " "
                   << detectedMarkers.size() << " "
                   << codedM << " "
                   << t * 1000 << std::endl;

        n_markers+=codedM;

        if (vis_debug > 0){
            int thickness = (image.cols > 1000) ? image.cols/1000.0: 1;
            // Draw
            for (auto &m : detectedMarkers){
                cv::ellipse(image,cv::Point2f(m.x,m.y),cv::Size2f(m.b, m.a),m.angle/CV_PI*180.0,0,360,cv::Scalar(0,255,255),thickness);
                if (m.id > 0){
                    std::string label = std::to_string(m.id);
                    cv::putText(image, label, cv::Point(m.x, m.y), cv::FONT_HERSHEY_PLAIN,thickness, cv::Scalar(0,0,255),thickness);
                }
            }

            // drawing angle of ellipse
            for (auto const &m : detectedMarkers){
                cv::Point2f point2 = cv::Point2f(
                    m.a * cos(m.angle - CV_PI*0.5) * 2.0 + m.x,
                    m.a * sin(m.angle - CV_PI*0.5) * 2.0 + m.y);
                cv::line(image,m.point(),point2,cv::Scalar(0,0,255),thickness);
            }

            // Show
            cv::namedWindow("Detected Markers", cv::WINDOW_GUI_EXPANDED );
            cv::imshow("Detected Markers", image);
            cv::waitKey();

            // Write
            // cv::imwrite(folderpath + "/" + file_name + "_out.jpg", image);
        }
    }
    std::cerr << time << " seconds to detect " << n_markers << " coded markers in " << n_images << " images (" << (time/double(n_images))*1000 << " ms per image)." << std::endl;

    resultfile_param.close();
    stats_file.close();
    
    exit(0);
}
