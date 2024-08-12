# CtCalib
Project for the calibration of cone beam computed tomography systems using projections of a calibration phantom with unknown coordinates.

**Limitations:** Note that the project was developed as part of a research project between July 2023 and June 2024. It will most likely not receive any updates and may contain bugs. Please use the code with caution.

## Introduction
The misalignment of cone beam computed tomography systems can be modeled by several different approaches. In this project, three different models are implemented. Two of which are described in the [corresponding publication](https://doi.org/10.3390/s24165139). The third model is the one used in the  [Reconstruction Toolkit (RTK)](https://www.openrtk.org/) and described in [RTK 3D circular projection geometry](http://www.openrtk.org/Doxygen/DocGeo3D.html). The geometry models describe the relationship between the 3D points on the rotation axis and their projection onto the detector frame. The parameters of these models are adjusted using the projections of a calibration phantom with steel ball bearings.

**This project contains:**
- Methods for the detection of ellipses in projection images.
- Methods for adjusting system geometry parameters to a number of observations.
- The structure to create new geometry models.

**This is not for:**
- The Reconstruction of the 3D geometry from projections.
- Using the determined parameters to correct projection images.

## Installation
### Dependencies
- OpenCV (tested with 4.8.0)
- Ceres Solver (tested with 2.1.0)
### Windows
- Use VCPKG to install dependencies `vcpkg install opencv` and `vcpkg install ceres`
- Use VisualStudio to open CMakeList.txt
- Build

## Usage
### Run with other data
To run a simple calibration with other data use: `example` with command line arguments.
```
example <path> <SDD> <SRD> <pixel size> <maximum rotation angle>
```
where:
- `<path>` - The path to the image files (every format that can be read with `cv::imread(..., cv::IMREAD_ANYDEPTH)`)
- `<SDD>`- Sensor Detector Distance in mm (initial value, will be adjusted)
- `<SRD>`- Sensor Rotation Axis Distance in mm (will not be adjusted)
- `<pixel size>` - Size of one pixel in mm
- `<maximum rotation angle>` - Gantry angle of last projection in rad to calculate the angle difference between two projections.
Depending on the direction of rotation this value can be negative or positive.  Examples:
	- For a full 360° degree **clockwise** rotation: `6.28`
	- For a full 360° degree **counter clockwise** rotation: `-6.28`
	- For a 180° degree **clockwise** rotation: `3.14`

### Adding other geometry models
For a quick start with the Ceres Solver library check first: [introduction](http://ceres-solver.org/nnls_tutorial.html#introduction). 
The class for a new CT geometry model should inherit from the base class `GeometryModel`.  See `GeometryDetector.h` and `GeometryDetector.cpp` for a full example.
The main elements of a geometry model are explained in the following.  The geometry parameters are defined in the constructor (with name, initial value, and if they are adjusted): 
```cpp
m_parameter = {
	GeometryParameter("SDD", SDD, AdjustEnumOptions::ADJUST_PARAM),
	GeometryParameter("SRD", SRD, AdjustEnumOptions::CONST_PARAM),
	GeometryParameter("OutOfPlaneAngle", 0, AdjustEnumOptions::ADJUST_PARAM),
	GeometryParameter("InPlaneAngle", 0, AdjustEnumOptions::ADJUST_PARAM),
	GeometryParameter("ProjectionOffsetX", 0, AdjustEnumOptions::ADJUST_PARAM),
	GeometryParameter("ProjectionOffsetY", 0, AdjustEnumOptions::ADJUST_PARAM),
	GeometryParameter("RotationOffsetX", 0, AdjustEnumOptions::ADJUST_PARAM),
	GeometryParameter("GantryAngleStep", 0, AdjustEnumOptions::ADJUST_PARAM)
};
```
In the header file the template function for creating the 3x4 projection matrix is defined. 
```cpp
template<typename T>
static Eigen::Matrix<T, 3, 4> projectionMatrix(const T* const GantryAngle,
	const T* const OutOfPlaneAngle,
	const T* const InPlaneAngle,
	const T* const ProjectionOffsetX,
	const T* const ProjectionOffsetY,
	const T* const RotationOffsetX,
	const T* const SDD,
	const T* const SRD) {

	// Create projection matrix
	// ...

	return P;
}
``` 
With this the cost function for the Ceres Solver is created in a struct. Make shure to use a consistent order of the arguments in the template operator.
The template parameters for ``ceres::AutoDiffCostFunction<...>()`` contain the cost function, the size of residuals (for a 2D observaton = 2), and followed by the size of each parameter.
``R_t`` is a 6 parameter transformation which is a constant identity matrix by default but can be adjusted if the object point coordinates are known and not adjusted.
```cpp
struct reprojectionError {
	reprojectionError(double observed_x, double observed_y)
		: observed_x(observed_x), observed_y(observed_y) {}

	template <typename T>
	bool operator()(const T* const ObjectPoint,
		const T* const R_t,
		const T* const GantryIndex,
		const T* const GantryAngleStep,
		const T* const OutOfPlaneAngle,
		const T* const InPlaneAngle,
		const T* const ProjectionOffsetX,
		const T* const ProjectionOffsetY,
		const T* const RotationOffsetX,
		const T* const SDD,
		const T* const SRD,
		T* residuals) const {
			
		// Project point and calculate difference to observations
		// ...

		return true;
	}

	// Factory to hide the construction of the CostFunction object from
	// the client code.
	static ceres::CostFunction* Create(const double observed_x, const double observed_y) {
		return (new ceres::AutoDiffCostFunction<reprojectionError, 2, 3, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1>(
			new reprojectionError(observed_x, observed_y)));
	}

	double observed_x;
	double observed_y;
};
```
Finally the function for adding an observation to the adjustment is created:
```cpp
ceres::ResidualBlockId GeometryDetector::addResidualBlock(
	ceres::Problem& problem,
	ceres::LossFunction* loss_function,
	std::vector<double>& object_point,
	double& gantry,
	double& x,
	double& y)
{
	// Create map of parameter
	std::map<std::string, GeometryParameter*> _plist = getParameterMap();

	// Create cost function
	ceres::CostFunction* cost_function = reprojectionError::Create(x, y);

	// Add residual block
	return problem.AddResidualBlock(cost_function,
		loss_function,
		&object_point[0],
		&m_transformation[0],
		&gantry, // index
		&_plist["GantryAngleStep"]->m_value,
		&_plist["OutOfPlaneAngle"]->m_value,
		&_plist["InPlaneAngle"]->m_value,
		&_plist["ProjectionOffsetX"]->m_value,
		&_plist["ProjectionOffsetY"]->m_value,
		&_plist["RotationOffsetX"]->m_value,
		&_plist["SDD"]->m_value,
		&_plist["SRD"]->m_value);
}
```


## How to cite
The corresponding publication can be found [here](https://doi.org/10.3390/s24165139).
```
@Article{s24165139,
AUTHOR = {Hardner, Matthias and Liebold, Frank and Wagner, Franz and Maas, Hans-Gerd},
TITLE = {Investigations into the Geometric Calibration and Systematic Effects of a Micro-CT System},
JOURNAL = {Sensors},
VOLUME = {24},
YEAR = {2024},
NUMBER = {16},
ARTICLE-NUMBER = {5139},
URL = {https://www.mdpi.com/1424-8220/24/16/5139},
ISSN = {1424-8220},
DOI = {10.3390/s24165139}
}
```
