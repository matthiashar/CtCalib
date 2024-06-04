#include "ProjectionData.h"
#include <iomanip>

ProjectionData::ProjectionData(double start_angle, double end_angle) : m_start_angle(start_angle), m_end_angle(end_angle)
{

}


ProjectionData::~ProjectionData()
{
}

bool ProjectionData::read(std::string path_file) {
	cv::FileStorage fs;
	if (!fs.open(path_file, cv::FileStorage::READ)) {
		std::cerr << "Warning: Unable to read data from " << path_file << std::endl;
		return false;
	}

	fs["path_projections"] >> path_projections;
	fs["start_angle"] >> m_start_angle;
	fs["end_angle"] >> m_end_angle;
	// read projections
	cv::FileNode n = fs["projections"];
	if (n.type() == cv::FileNode::SEQ)
	{
		cv::FileNodeIterator it = n.begin(), it_end = n.end();
		int i = 0;
		for (; it != it_end; ++it, i++) {
			Projection _pd;
			cv::FileNode pdnode = (*it);
			pdnode["pj"] >> _pd;
			_pd.setImagePath(path_projections + "/" + _pd.getImageName()); // create path
			m_projections.push_back(_pd);
		}
	}
	return true;
}

bool ProjectionData::write(std::string path_file) {

	cv::FileStorage fs;
	if (!fs.open(path_file, cv::FileStorage::WRITE)) {
		std::cerr << "Error: Unable to write data to " << path_file << std::endl;
		return false;
	}

	fs << "path_projections" << path_projections;
	fs << "start_angle" << m_start_angle;
	fs << "end_angle" << m_end_angle;
	fs << "projections" << "[";
	for (auto& proj : m_projections) {
		fs << "{:" << "pj" << proj << "}";
	}
	fs << "]";
	fs.release();

	return true;
}

bool ProjectionData::writeTextfile(std::string path_file)
{
	std::ofstream data_out(path_file);
	data_out << std::setprecision(8) << "name index id x y typ rmseX rmseY" << std::endl;
	for (auto const& proj : m_projections) {
		for (auto const& point : proj.observation) {
			data_out << proj.getImageName() << " "
				<< proj.gantry_index << " "
				<< point.id << " "
				<< point.x << " "
				<< point.y << " ";
			data_out << point.point_typ << " ";
			if (point.residual_pixel) {
				data_out << point.residual_pixel.value().x << " "
					<< point.residual_pixel.value().y << " ";
			}
			else {
				data_out << "NaN NaN ";
			}
			data_out << std::endl;
		}
	}
	data_out.close();
	return true;
}

std::string ProjectionData::toString() const {
	std::stringstream s;
	s << "Path projections: " << path_projections << std::endl;
	s << "Number projections : " << m_projections.size() << std::endl;
	int n_marker = 0;
	int n_outlier = 0;
	for (auto const& p : m_projections) {
		for (auto const& obs : p.observation) {
			if (obs.id > 0)
				n_marker++;
			if (obs.point_typ == Projection::Observation::TYP::outlier)
				n_outlier++;
		}
	}
	s << "Number detected marker: " << n_marker << " (" << double(n_marker) / double(m_projections.size()) << " per projection)" << std::endl;
	s << "Number outliers: " << n_outlier << std::endl;

	return s.str();
}



