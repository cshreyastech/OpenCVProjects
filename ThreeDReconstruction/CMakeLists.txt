project(ThreeDReconstruction)
cmake_minimum_required(VERSION 3.0.0)

# set(HEADER_DIR
#   include
# )

set(SRC_FILES
  src/three_d_reconstruction.cpp
)

set(project_app ${PROJECT_NAME}_app)

ADD_EXECUTABLE(${project_app}
  ${SRC_FILES}
)

target_precompile_headers(${project_app} PRIVATE ${CMAKE_SOURCE_DIR}/pch/include/pch/pch.h)

target_include_directories(${project_app}
  PRIVATE ${CMAKE_SOURCE_DIR}/pch/include
  PRIVATE ${OpenCV_INCLUDE_DIRS}
  PRIVATE ${HEADER_DIR}
)

target_link_libraries(${project_app}
  PRIVATE ${OpenCV_LIBS}
)

set_target_properties(${project_app} PROPERTIES 
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED YES
)
