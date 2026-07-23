cmake_minimum_required(VERSION 3.28)
project(KairoTransformers LANGUAGES CXX)

set(KAIRO_FOUNDATION_DIR "/Users/swayamsingal/Desktop/Programming/Kairo/Foundation" CACHE PATH "Path to Kairo Foundation.")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(KairoTransformers)
add_subdirectory("${KAIRO_FOUNDATION_DIR}/KairoMath" "${CMAKE_BINARY_DIR}/KairoMath")
target_sources(KairoTransformers
    PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
        FILES
            src/KairoTransformers.cppm
            src/KairoTransformerRuntime.cppm
            src/KairoTransformerWeights.cppm
)
target_link_libraries(KairoTransformers PUBLIC KairoMath)

add_executable(KairoTransformersSmoke tests/smoke.cpp)
target_link_libraries(KairoTransformersSmoke PRIVATE KairoTransformers)

enable_testing()
add_test(NAME KairoTransformersSmoke COMMAND KairoTransformersSmoke)
