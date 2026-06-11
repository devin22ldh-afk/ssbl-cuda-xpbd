cmake_minimum_required(VERSION 3.25)

project(ssbl_xpbd_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures")

option(SSBL_BUILD_LEGACY "Build the legacy ABI36 CUDA backend" ON)
option(SSBL_BUILD_ABI41 "Build the ABI41 CUDA backend" ON)
set(SSBL_CUDA_OUTPUT_NAME "ssbl_xpbd_cuda_abi36" CACHE STRING "Native SSBL CUDA DLL output name")
set(SSBL_CUDA_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bin" CACHE PATH "Native SSBL CUDA DLL output directory")
set(SSBL_ABI41_OUTPUT_NAME "ssbl_xpbd_cuda_abi41" CACHE STRING "SSBL ABI41 CUDA DLL output name")
set(SSBL_ABI41_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bin" CACHE PATH "SSBL ABI41 CUDA DLL output directory")

if (SSBL_BUILD_LEGACY)
    add_library(ssbl_xpbd_cuda SHARED
        src/ssbl_xpbd_cuda.cu
    )

    target_include_directories(ssbl_xpbd_cuda
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    set_target_properties(ssbl_xpbd_cuda PROPERTIES
        OUTPUT_NAME ${SSBL_CUDA_OUTPUT_NAME}
        CUDA_ARCHITECTURES 89
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RUNTIME_LIBRARY Static
        RUNTIME_OUTPUT_DIRECTORY ${SSBL_CUDA_OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${SSBL_CUDA_OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${SSBL_CUDA_OUTPUT_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${SSBL_CUDA_OUTPUT_DIR}
        LIBRARY_OUTPUT_DIRECTORY_DEBUG ${SSBL_CUDA_OUTPUT_DIR}
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${SSBL_CUDA_OUTPUT_DIR}
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/lib
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/lib
    )

    if (MSVC)
        target_compile_options(ssbl_xpbd_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcompiler=/utf-8>
        )
        target_link_options(ssbl_xpbd_cuda PRIVATE
            $<HOST_LINK:/NODEFAULTLIB:LIBCMT>
        )
    else()
        target_compile_options(ssbl_xpbd_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        )
    endif()
endif()

if (SSBL_BUILD_ABI41)
    add_library(ssbl_xpbd_cuda_abi41 SHARED
        src/ssbl_xpbd_cuda_abi41.cu
    )

    target_include_directories(ssbl_xpbd_cuda_abi41
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    set_target_properties(ssbl_xpbd_cuda_abi41 PROPERTIES
        OUTPUT_NAME ${SSBL_ABI41_OUTPUT_NAME}
        CUDA_ARCHITECTURES 89
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RUNTIME_LIBRARY Static
        RUNTIME_OUTPUT_DIRECTORY ${SSBL_ABI41_OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${SSBL_ABI41_OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${SSBL_ABI41_OUTPUT_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${SSBL_ABI41_OUTPUT_DIR}
        LIBRARY_OUTPUT_DIRECTORY_DEBUG ${SSBL_ABI41_OUTPUT_DIR}
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${SSBL_ABI41_OUTPUT_DIR}
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/lib
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/lib
    )

    add_executable(ssbl_abi41_smoke
        smoke/abi41_smoke.cpp
    )
    target_link_libraries(ssbl_abi41_smoke PRIVATE ssbl_xpbd_cuda_abi41)
    target_include_directories(ssbl_abi41_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(ssbl_abi41_smoke PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${SSBL_ABI41_OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${SSBL_ABI41_OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${SSBL_ABI41_OUTPUT_DIR}
    )

    if (MSVC)
        target_compile_options(ssbl_xpbd_cuda_abi41 PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4>
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcompiler=/utf-8>
        )
        target_compile_options(ssbl_abi41_smoke PRIVATE /W4 /utf-8)
        target_link_options(ssbl_xpbd_cuda_abi41 PRIVATE
            $<HOST_LINK:/NODEFAULTLIB:LIBCMT>
        )
    else()
        target_compile_options(ssbl_xpbd_cuda_abi41 PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        )
        target_compile_options(ssbl_abi41_smoke PRIVATE -Wall -Wextra)
    endif()
endif()
