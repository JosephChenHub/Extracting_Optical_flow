 

project(myOpticalFlow)   
cmake_minimum_required(VERSION 2.8)                 # CMake工具最低版本要求

aux_source_directory(./src SRC_LIST)                    # 源文件列表
    
include_directories(./src)                     # 头文件目录

#link_directories(/usr/local/cuda/include)           # 库文件的目录

add_executable(${PROJECT_NAME} ${SRC_LIST})         # 工程项目名和我们要编译的文件名

#find_package(OpenCV REQUIRED)                       # 搜索指定的外部依赖库头文件和库文件
 


#OpenMP
FIND_PACKAGE( OpenMP REQUIRED)  
if(OPENMP_FOUND)  
message("OPENMP FOUND")  
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  
endif()  

#OpenCV OpenGL
 
FIND_PACKAGE(OpenCV REQUIRED )  
find_package(OpenGL REQUIRED )  

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})   # 链接到OpenCV库
include_directories(/usr/local/cuda/include) 

include_directories(${OpenCV_INCLUDE_DIR})  
include_directories(${OpenGL_INCLUDE_DIR})  
  
link_directories(${OpenCV_LIBRARY_DIR})  
link_directories(${OpenGL_LIBRARY_DIR}) 

 


