cmake_minimum_required(VERSION 3.10)
project(bmengine VERSION 0.1)

enable_language(C)
enable_language(CXX)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;89;90a")
endif()
if(NOT APPLE)
  enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)


find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUBLAS_LIBRARY cublas ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUBLASLT_LIBRARY cublasLt ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUBRAND_LIBRARY curand_static ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CULIBOS_LIBRARY culibos ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_package(NCCL REQUIRED)


file(GLOB_RECURSE FILES_BMENGINE "*.cpp")
file(GLOB_RECURSE FILES_BMENGINE_CUDA "*.cu")
file(GLOB_RECURSE FILES_BMENGINE_HEADER RELATIVE "include/bmengine" "*.h")

add_library(bmengine STATIC
    ${FILES_BMENGINE}
    ${FILES_BMENGINE_CUDA}
)
if(NCCL_FOUND)
  include_directories(SYSTEM ${NCCL_INCLUDE_DIRS})
else()
  message(WARNING "Not compiling with NCCL support. Suppress this warning with -DUSE_NCCL=OFF.")
  set(USE_NCCL OFF)
endif()
set_property(TARGET bmengine PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET bmengine PROPERTY CMAKE_CXX_VISIBILITY_PRESET hidden)
set_property(TARGET bmengine PROPERTY CMAKE_CUDA_VISIBILITY_PRESET hidden)

target_include_directories(bmengine 
    PUBLIC "include"
    PUBLIC "include/private/3rd/"
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(
    bmengine 
    ${CUDART_LIBRARY}
    ${CUBLAS_LIBRARY}
    ${CUBLASLT_LIBRARY}
    ${CUBRAND_LIBRARY}
    ${CULIBOS_LIBRARY}
    ${CULIBOS_LIBRARY}
    ${NCCL_LIBRARIES}
    "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions"
    "pthread"
)

include(GNUInstallDirs)
install(
    TARGETS bmengine
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(
    DIRECTORY "include/bmengine" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)
