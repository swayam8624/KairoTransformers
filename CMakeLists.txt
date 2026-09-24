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
            src/KairoTransformerCheckpoint.cppm
            src/KairoTransformerTraining.cppm
)
target_link_libraries(KairoTransformers PUBLIC KairoMath)

add_executable(KairoTransformersSmoke tests/smoke.cpp)
target_link_libraries(KairoTransformersSmoke PRIVATE KairoTransformers)
add_executable(KairoTransformerTrainingSmoke tests/training.cpp)
target_link_libraries(KairoTransformerTrainingSmoke PRIVATE KairoTransformers)
add_executable(KairoTransformerBenchmark tests/benchmark.cpp)
target_link_libraries(KairoTransformerBenchmark PRIVATE KairoTransformers)

enable_testing()
add_test(NAME KairoTransformersSmoke COMMAND KairoTransformersSmoke)
add_test(NAME KairoTransformerTrainingSmoke COMMAND KairoTransformerTrainingSmoke)
add_test(
    NAME KairoTransformerBenchmark
    COMMAND KairoTransformerBenchmark
        "${CMAKE_CURRENT_BINARY_DIR}/transformer_benchmark.json")

# Kairo portfolio warning policy: Kairo-owned source warnings are build failures.
# Catch2's __COUNTER__ registration macro is the only bounded Clang 23 test
# exception; production targets receive no blanket warning suppression.
foreach(_kairo_target IN ITEMS KairoTransformers KairoTransformersSmoke KairoTransformerTrainingSmoke KairoTransformerBenchmark)
    if(TARGET ${_kairo_target})
        set_property(TARGET ${_kairo_target} PROPERTY COMPILE_WARNING_AS_ERROR ON)
    endif()
endforeach()
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 23)
    foreach(_kairo_test_target IN ITEMS KairoTransformersSmoke KairoTransformerTrainingSmoke)
        if(TARGET ${_kairo_test_target})
            target_compile_options(${_kairo_test_target} PRIVATE -Wno-c2y-extensions)
        endif()
    endforeach()
endif()
