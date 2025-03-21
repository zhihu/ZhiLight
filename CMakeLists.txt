cmake_minimum_required(VERSION 3.20)

project(zhilight)

enable_language(C)
enable_language(CXX)

set(CMAKE_CUDA_ARCHITECTURES $ENV{CMAKE_CUDA_ARCHITECTURES})
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;89;90a")
endif()
message("CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_VISIBILITY_PRESET default)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 0)
set(CMAKE_BUILD_RPATH ${ORIGIN})
set(CMAKE_INSTALL_RPATH ${ORIGIN})
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)
#set(USE_STATIC_NCCL True)

message("CUDA Version: ${CMAKE_CUDA_COMPILER_TOOLKIT_VERSION}")

add_compile_options(
    -Wno-deprecated-declarations
    -Wno-narrowing
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=177,549">)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
set(PYTHON_VERSION ${Python_VERSION})

if(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION VERSION_LESS 12.0.0)
    error("Require CUDA 12")
else()
    message("Will link against CUDA 12 complied libraries")
endif()

option(WITH_TESTING "Compile zhilight with unit testing" ON)

if(WITH_TESTING)
	execute_process(
		COMMAND ${PYTHON_EXECUTABLE} -c "import torch; \
			print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
		OUTPUT_VARIABLE PYTORCH_CXX_ABI
		OUTPUT_STRIP_TRAILING_WHITESPACE
        # COMMAND_ECHO STDERR
        COMMAND_ERROR_IS_FATAL ANY)
    list(APPEND CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${PYTORCH_CXX_ABI}")
    list(APPEND CMAKE_CUDA_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=${PYTORCH_CXX_ABI}")
    message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
endif()

string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string(REPLACE "-DNDEBUG" "" CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO}")

message(STATUS "CMAKE_INSTALL_RPATH: ${CMAKE_INSTALL_RPATH}")

include(bmengine)
add_subdirectory(3rd/bmengine)

execute_process(
	COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; \
		print(pybind11.get_cmake_dir())"
	OUTPUT_VARIABLE pybind11_DIR
	OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 REQUIRED)

find_package(Boost 1.71.0 REQUIRED COMPONENTS serialization)
if(NOT Boost_FOUND OR NOT TARGET Boost::serialization)
    message(FATAL_ERROR "Could not find required component Boost::serialization of the Boost library.")
endif()

file(GLOB_RECURSE BACKEND_SOURCES
    "src/*.cpp"
    "src/*.cu"
)

list(FILTER BACKEND_SOURCES EXCLUDE REGEX "src/py_export/.*\\.cpp$")
list(FILTER BACKEND_SOURCES EXCLUDE REGEX "src/nn/tests/.*$")

add_library(backend STATIC ${BACKEND_SOURCES})
set_property(TARGET backend PROPERTY POSITION_INDEPENDENT_CODE ON)

file(GLOB FLASH_ATTN_LIB "${Python_SITELIB}/flash_attn*.so")
get_filename_component(FLASH_ATTN_SO ${FLASH_ATTN_LIB} NAME)
message("FLASH_ATTN_SO: " ${FLASH_ATTN_SO})

target_include_directories(backend
    PUBLIC "src"
    PUBLIC "3rd/tensorrt_llm/"
    PUBLIC "3rd/flash_decoding"
    PUBLIC "3rd/zmq"
    PUBLIC ${NCCL_INCLUDE_DIRS}
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    PUBLIC ${Python_SITELIB}/torch/include
    PUBLIC ${Boost_INCLUDE_DIRS}
)

target_link_directories(backend
    PUBLIC "3rd/tensorrt_llm/"
    PUBLIC "3rd/zmq"
    PUBLIC ${NCCL_LIBRARIES}
    PUBLIC ${Python_SITELIB}/torch/lib
    PUBLIC ${Python_SITELIB}
    PUBLIC ${Boost_LIBRARY_DIRS}
)

target_link_libraries(backend
    PUBLIC bmengine zmq Boost::serialization
    "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions"
    ":${FLASH_ATTN_SO}"
)

file(GLOB_RECURSE PYBIND_SOURCES "src/py_export/*.cpp")

pybind11_add_module(C ${PYBIND_SOURCES})
target_compile_definitions(C
    PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

target_link_libraries(C PRIVATE
    bmengine
    "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions"
    "pthread"
    "backend"
)

if(WITH_TESTING)
    add_subdirectory(tests/py_export_internal)
    add_subdirectory(src/nn/tests)
endif()