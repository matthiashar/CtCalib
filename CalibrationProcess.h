#ifndef CALIBRATIONPROCESS_H
#define CALIBRATIONPROCESS_H

#include "GeometryModel.h"
#include "ProjectionData.h"
#include "Calibration.h"

class CalibrationProcess
{
public:
    CalibrationProcess();

    /// If true, undetected points will be searched in the image by projecting object points to the detector and searching the image at this location.
    bool searchForAdditionalObservationsInImage;

    /// Maximum distance in pixels for the image point search.
    double observation_search_threshold;

    /// If true, for each projection, a white image will be saved with detected points and residuals.
    bool saveResiduals;

    /// If true, the background of the residual image will be the original projection.
    bool residualsWithBackground;

    /// If true, outliers will be removed.
    bool removeOutliers;

    /// Multiplier for outlier removal.
    double outlier_iqr_multiplier;

    /// If true, eccentricity will be corrected.
    bool correctEccentricity;

    /// Maximum number of iterations for removing outliers and searching for new points.
    unsigned int number_iterations;

    /// Maximum distance of object points in 3D space where points will be merged.
    double object_point_merge_threshold;

    /// Bundle adjustment options.
    Calibration::Options calibration_options;

    /// Function to run the calibration.
    void runCalibration(ProjectionData& pdata, GeometryModel& m, std::string out_path);
};


#endif // CALIBRATIONPROCESS_H