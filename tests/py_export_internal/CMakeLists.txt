enable_language(CUDA)

project(internals_)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_VISIBILITY_PRESET default)
set(CMAKE_CUDA_VISIBILITY_PRESET default)
set(TORCH_CUDA_ARCH_LIST "8.0+PTX;8.9")

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${PYTORCH_CMAKE_PREFIX_PATH}/Torch")
message("PYTORCH_CMAKE_PREFIX_PATH ${PYTORCH_CMAKE_PREFIX_PATH}")
message("CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

file(GLOB_RECURSE TEST_SRC_NN "*.cpp")
# TODO: better way to share this util file
list(APPEND TEST_SRC_NN "../../src/py_export/py_utils.cpp")

pybind11_add_module(${PROJECT_NAME} ${TEST_SRC_NN})
target_compile_definitions(${PROJECT_NAME}
    PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

#target_include_directories(${PROJECT_NAME} PRIVATE ${Python_SITELIB}/torch/include)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/3rd/bmengine/bmengine/include)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/testing)
target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PRIVATE
    "backend"
    "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions"
    "pthread"
    "${TORCH_LIBRARIES}"
    "${TORCH_PYTHON_LIBRARY}"
)