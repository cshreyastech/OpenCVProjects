cmake_minimum_required(VERSION 3.0.0)
project(OpenCV)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/out/bin)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

### Uncomment the line below and specify the path to OpenCV directory i.e. the path to the OpenCVConfig.cmake file. Check the examples given below.
###SET PATH TO OPENCV CMAKE CONFIG FILES

# find_package( OpenCV 
#   NO_DEFAULT_PATH
#   PATHS "/mnt/hd/disk2/softwares/OpenCV/installation/OpenCV"
#   REQUIRED 
#   )

find_package( OpenCV REQUIRED )
# INCLUDE OPENCV DIRECTORIES
include_directories( ${OpenCV_INCLUDE_DIRS})


# add_subdirectory(SampleCode)
# add_subdirectory(ThreeDReconstruction)
add_subdirectory(OpencvCourse1)
