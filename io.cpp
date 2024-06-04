#include "io.h"


GeometryModel* io::readGeometry(std::string url)
{
	cv::FileStorage fs;
	if (!fs.open(url, cv::FileStorage::READ)) {
		std::cerr << "Error: Unable to read data from " << url << std::endl;
		return nullptr;
	}

	// Get typ
	std::string gemetry_typ;
	fs["gemetry_typ"] >> gemetry_typ;

	// Vector of posible geometries
	std::vector<GeometryModel*> geometries = Geometries::listGeometries();

	// Find correct geomtry
	GeometryModel* geometry = nullptr;
	for (auto& g : geometries) {
		if (g->getName() == gemetry_typ) {
			geometry = g;
		}
	}

	// Read object
	if (geometry != nullptr) {
		cv::Size2i image_size;
		fs["image_size_px"] >> image_size;

		double pixel_size;
		fs["pixel_size_mm"] >> pixel_size;

		std::vector<GeometryParameter> param;
		cv::FileNode n = fs["parameter"];
		if (n.type() == cv::FileNode::SEQ)
		{
			cv::FileNodeIterator it = n.begin(), it_end = n.end();
			for (; it != it_end; ++it) {
				GeometryParameter gp;
				(*it)["name"] >> gp.m_name;
				(*it)["value"] >> gp.m_value;
				(*it)["adjust"] >> gp.m_adjust_option;
				param.push_back(gp);
			}
		}

		geometry->setImageSizePixel(image_size);
		geometry->setPixelSize(pixel_size);
		geometry->setParameter(param);
	}
	else {
		std::cerr << "Error: Unknown geometry typ " << gemetry_typ << std::endl;
	}

	return geometry;
}

bool io::writeGeometry(std::string url, GeometryModel& geometry)
{
	cv::FileStorage fs;
	if (!fs.open(url, cv::FileStorage::WRITE)) {
		std::cerr << "Error: Unable to write data to " << url << std::endl;
		return false;
	}
	fs << "gemetry_typ" << geometry.getName();
	fs << "image_size_px" << geometry.getImageSizePixel();
	fs << "pixel_size_mm" << geometry.getPixelSize();
	fs << "parameter" << "[";

	std::vector<GeometryParameter> &param = geometry.getParameter();

	for (auto& p : param) {
		fs << "{:" << "name" << p.m_name << "value" << p.m_value << "adjust" << p.m_adjust_option << "}";
	}
	fs << "]";
	fs.release();

	return true;
}

std::map<int, std::vector<double>> io::readObjectPoints(const std::string& path) {
	std::map<int, std::vector<double>> oPoints;
	std::ifstream file(path);
	if (!file.is_open()) {
		std::cerr << "Error opening file: " << path << std::endl;
		return oPoints;
	}

	std::string line;
	while (getline(file, line)) {
		std::istringstream ss(line);
		double x, y, z;
		int id;
		if (ss >> x >> y >> z >> id) {
			oPoints[id] = { x, y, z };
		}
	}

	return oPoints;
}

bool io::writeObjectPoints(const std::string& path, std::map<int, std::vector<double>>& points)
{
	std::ofstream points_out(path);
	points_out << std::setprecision(8) << std::endl;
	for (auto const& op : points) {
		points_out << op.second[0] << " " << op.second[1] <<
			" " << op.second[2] << " " << op.first << std::endl;
	}
	points_out.close();
	return false;
}
