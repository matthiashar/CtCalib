cmake_minimum_required(VERSION 3.9)

project(marker_detection)

# OpenCV
find_package(OpenCV REQUIRED)

# Eigen
#find_package (Eigen3 3.3 REQUIRED)

# Project
set (project_SOURCES
    markerdetection.cpp)
set (project_HEADERS
    markerdetection.h
    codeLookUp.h)

# Options
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Executable
add_executable(marker_detection ${project_SOURCES} ${project_HEADERS} "main.cpp")
target_link_libraries(marker_detection ${OpenCV_LIBS})

# Test
add_executable(test_marker_detection ${project_SOURCES} ${project_HEADERS} "test.cpp")
add_test(marker_detection test_marker_detection)
enable_testing()
target_link_libraries(test_marker_detection ${OpenCV_LIBS})
