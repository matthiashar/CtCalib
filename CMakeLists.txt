cmake_minimum_required(VERSION 3.9)

project(ct_calibration)

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

# Project
set (project_SOURCES
    marker_detection/markerdetection.cpp
    Geometries.cpp
    GeometryModel.cpp
    GeometryRTK.cpp
    GeometryDetector.cpp
    GeometryTilt.cpp
    Projection.cpp
    ProjectionData.cpp
    io.cpp
    ProjectionProcessing.cpp
    Calibration.cpp
    CalibrationProcess.cpp
    )
set (project_HEADERS
    marker_detection/markerdetection.h
    marker_detection/codeLookUp.h
    Geometries.h
    GeometryModel.h
    GeometryRTK.h
    GeometryDetector.h
    GeometryTilt.h
    Projection.h
    ProjectionData.h
    io.h
    ProjectionProcessing.h
    Calibration.h
    CalibrationProcess.h
    )

# Options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Executable
add_executable(example ${project_SOURCES} ${project_HEADERS} example.cpp)
target_link_libraries(example ${OpenCV_LIBS} Ceres::ceres ) 