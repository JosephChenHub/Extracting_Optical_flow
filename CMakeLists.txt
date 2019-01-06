project(myOpticalFlow)
cmake_minimum_required(VERSION 2.8)

add_definitions("-std=c++11")
set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")

set(rgb_main ./src/extract_rgb.cpp)
set(flow_main ./src/extract_flow.cpp)
set(utils ./src/utils.cpp)

include_directories(./src)
add_executable(extract_rgb ${rgb_main} ${utils})
set_target_properties(extract_rgb PROPERTIES COMPILE_FLAGS -pthread LINK_FLAGS -pthread)


#OpenCV
find_package(OpenCV REQUIRED)

target_link_libraries(extract_rgb ${OpenCV_LIBS})
include_directories(/usr/local/cuda/include)
include_directories(${OpenCV_INCLUDE_DIR})
link_directories(${OpenCV_LIBRARY_DIR})

#MPI
find_package(MPI REQUIRED)
find_package(Boost 1.54 COMPONENTS mpi serialization system REQUIRED)

add_executable(extract_flow ${flow_main} ${utils})

target_compile_options(extract_flow PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_link_libraries(extract_flow ${OpenCV_LIBS} ${MPI_LIBRARIES} ${Boost_LIBRARIES})
target_include_directories(extract_flow PRIVATE ${MPI_INCLUDE_PATH} ${Boost_INCLUDE_DIR})

if(MPI_CXX_COMPILE_FLAGS)
    set_target_properties(extract_flow PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
endif()
if(MPI_CXX_LINK_FLAGS)
    set_target_properties(extract_flow PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()


