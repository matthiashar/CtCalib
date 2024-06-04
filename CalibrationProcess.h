#ifndef CALIBRATIONPROCESS_H
#define CALIBRATIONPROCESS_H

#include "GeometryModel.h"
#include "ProjectionData.h"
#include "Calibration.h"

class CalibrationProcess
{
public:
	CalibrationProcess();

	bool searchForAditionalObservationsInImage, residualsWithBackground,
		saveResiduals, removeOutlier, correctEccentricity;
	unsigned int number_iteratations;
	double object_point_merge_threshold;
	double outlier_iqr_multiplier;
	double oberservation_search_threshold;

	Calibration::Options calibration_options;

	void runCalibration(ProjectionData& pdata, GeometryModel& m, std::string out_path);
};

#endif // CALIBRATIONPROCESS_H