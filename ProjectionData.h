#ifndef PROJECTIONDATA_H
#define PROJECTIONDATA_H

#include <Eigen/Dense>

#include <fstream>
#include <iostream>
#include "Projection.h"

class ProjectionData
{
public:
	ProjectionData(double start_angle = 0, double end_angle = CV_2PI);
	~ProjectionData();	

	// Parameter
	double m_start_angle, m_end_angle;

	// Projections
	std::string path_projections;
	std::vector<Projection> m_projections;

	bool read(std::string path_file);
	bool write(std::string path_file);
	bool writeTextfile(std::string path_file);

	std::string toString() const;

private:
	std::string m_name;
};



#endif // PROJECTIONDATA_H
