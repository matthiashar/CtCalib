# CtCalib
Project for the calibration of cone beam computed tomography systems using projections of a calibration phantom with unknown coordinates.

**Limitations:** Note that the project was developed as part of a research project and may contain bugs. Please use code with caution. 

## Introduction
The misalignment of cone beam computed tomography systems can be modeled by several different approaches. In this project, three different models are implemented. Two of which are described in the corresponding publication (see How to cite). The third model is the one used in the  [Reconstruction Toolkit (RTK)](https://www.openrtk.org/) and described in [RTK 3D circular projection geometry](http://www.openrtk.org/Doxygen/DocGeo3D.html). The geometry models describe the relationship between the 3D points on the rotation axis and their projection onto the detector frame. The parameters of these models are adjusted using the projections of a calibration phantom with steel ball bearings.

This project contanins:
- Methods for the detection of ellipses
- todo
This project cannot be used for:
- Reconstruction of the 3D geometry

## Installation
### Dependencies
- OpenCV (tested with 4.8.0)
- Ceres Solver (tested with 2.1.0)
### Windows
- Use VCPKG to install dependencies `vcpkg install opencv` and `vcpkg install ceres`
- Use VisualStudio to open CMakeList.txt
- Build

## Usage
### Run with own data
  TODO
### Adding other geometry models
	TODO

## How to cite
	TODO
