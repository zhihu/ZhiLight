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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)


find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUBLAS_LIBRARY cublas ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUBLASLT_LIBRARY cublasLt ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CURAND_LIBRARY curand ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CULIBOS_LIBRARY culibos ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

# ENABLE_NCCL_TP=on by default.
if(NOT DEFINED ENV{ENABLE_NCCL_TP} OR "$ENV{ENABLE_NCCL_TP}" STREQUAL "on")
    add_definitions(-DENABLE_NCCL_TP=1)
    find_package(NCCL REQUIRED)
    include_directories(SYSTEM ${NCCL_INCLUDE_DIRS})
else()
    message(STATUS "Not compiling with NCCL support.")
endif()


file(GLOB_RECURSE FILES_BMENGINE "*.cpp")
file(GLOB_RECURSE FILES_BMENGINE_CUDA "*.cu")
file(GLOB_RECURSE FILES_BMENGINE_HEADER RELATIVE "include/bmengine" "*.h")

add_library(bmengine STATIC
    ${FILES_BMENGINE}
    ${FILES_BMENGINE_CUDA}
)

set_property(TARGET bmengine PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET bmengine PROPERTY CMAKE_CXX_VISIBILITY_PRESET hidden)
set_property(TARGET bmengine PROPERTY CMAKE_CUDA_VISIBILITY_PRESET hidden)

target_include_directories(bmengine 
    PUBLIC "include"
    PUBLIC "include/private/3rd/"
    PUBLIC "../../zmq"
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# ENABLE_DIST_INFER=off by default.
set(ZMQ_LIBRARY "")
if("$ENV{ENABLE_DIST_INFER}" STREQUAL "on")
    add_definitions(-DENABLE_DIST_INFER=1)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
      set(ZMQ_LIBRARY "zmq_arm")
    else()
      set(ZMQ_LIBRARY "zmq")
    endif()
    message(ZMQ_LIBRARY, ${ZMQ_LIBRARY})
endif()

target_link_directories(
    bmengine
    PUBLIC "../../zmq"
)
target_link_libraries(
    bmengine 
    ${CUDART_LIBRARY}
    ${CUBLAS_LIBRARY}
    ${CUBLASLT_LIBRARY}
    ${CURAND_LIBRARY}
    ${CULIBOS_LIBRARY}
    ${CULIBOS_LIBRARY}
    ${NCCL_LIBRARIES}
    "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions"
    "pthread"
    ${ZMQ_LIBRARY}
)

include(GNUInstallDirs)
install(
    TARGETS bmengine
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(
    DIRECTORY "include/bmengine" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)
