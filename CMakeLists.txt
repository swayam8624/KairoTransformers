cmake_minimum_required(VERSION 3.28)
project(KairoTransformers LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(KairoTransformers)
target_sources(KairoTransformers
    PUBLIC
        FILE_SET CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src
        FILES src/KairoTransformers.cppm
)

add_executable(KairoTransformersSmoke tests/smoke.cpp)
target_link_libraries(KairoTransformersSmoke PRIVATE KairoTransformers)

enable_testing()
add_test(NAME KairoTransformersSmoke COMMAND KairoTransformersSmoke)
