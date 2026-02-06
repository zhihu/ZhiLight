cmake_minimum_required(VERSION 3.10)
project(tests VERSION 0.1)

enable_language(C)
enable_language(CXX)
# enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_VISIBILITY_PRESET default)
set(CMAKE_CUDA_VISIBILITY_PRESET default)
SET ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__ " )
SET ( CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D__CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__ " )
SET ( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -D__CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__  -rdynamic -Wl,-Bsymbolic -Wl,-Bsymbolic-functions" )
SET ( CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS} -D__CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__  -rdynamic -Wl,-Bsymbolic -Wl,-Bsymbolic-functions" )

add_executable(test_allocator test_allocator.cpp)
target_link_libraries(test_allocator LINK_PUBLIC bmengine)

add_executable(test_gemm test_gemm.cpp)
target_link_libraries(test_gemm LINK_PUBLIC bmengine)

add_executable(test_print_tensor test_print_tensor.cpp)
target_link_libraries(test_print_tensor LINK_PUBLIC bmengine)

add_executable(test_bitonic test_bitonic.cpp)
target_link_libraries(test_bitonic LINK_PUBLIC bmengine)

add_executable(test_softmax test_softmax.cpp)
target_link_libraries(test_softmax LINK_PUBLIC bmengine)

add_executable(test_index_select test_index_select.cpp)
target_link_libraries(test_index_select LINK_PUBLIC bmengine)

add_executable(test_nccl test_nccl.cpp)
target_link_libraries(test_nccl LINK_PUBLIC bmengine)

add_executable(test_thread_pool test_thread_pool.cpp)
target_link_libraries(test_thread_pool LINK_PUBLIC bmengine)
