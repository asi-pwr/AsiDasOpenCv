cmake_minimum_required(VERSION 3.8)
project(AsiDasOpenCv)

set(CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED )
set(SOURCE_FILES main.cpp)
add_executable(AsiDasOpenCv ${SOURCE_FILES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( AsiDasOpenCv ${OpenCV_LIBS} )