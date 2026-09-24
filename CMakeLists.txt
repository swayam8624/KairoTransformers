cmake_minimum_required(VERSION 3.28)
project(KairoTransformers LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(KAIRO_MATH_SOURCE_DIR "" CACHE PATH "Path to a local KairoMath checkout")
set(KAIRO_MATH_REVISION b0e0bf14d81398c85d16011d6c897c782fa6068a)

if(NOT TARGET KairoMath)
    if(KAIRO_MATH_SOURCE_DIR AND EXISTS "${KAIRO_MATH_SOURCE_DIR}/CMakeLists.txt")
        add_subdirectory(
            "${KAIRO_MATH_SOURCE_DIR}"
            "${CMAKE_CURRENT_BINARY_DIR}/_deps/KairoMath")
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../KairoMath/CMakeLists.txt")
        add_subdirectory(
            "${CMAKE_CURRENT_SOURCE_DIR}/../KairoMath"
            "${CMAKE_CURRENT_BINARY_DIR}/_deps/KairoMath")
    else()
        include(FetchContent)
        FetchContent_Declare(
            KairoMath
            GIT_REPOSITORY https://github.com/swayam8624/KairoMath.git
            GIT_TAG ${KAIRO_MATH_REVISION})
        FetchContent_MakeAvailable(KairoMath)
    endif()
endif()

add_library(KairoTransformers)
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

# Clang with libstdc++ 14 diagnoses std::aligned_storage inside the standard
# library as deprecated under C++23 while Tensor templates are instantiated.
# No Kairo transformer source uses the deprecated facility directly.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND UNIX AND NOT APPLE)
    foreach(_kairo_target IN ITEMS
        KairoTransformers
        KairoTransformersSmoke
        KairoTransformerTrainingSmoke
        KairoTransformerBenchmark)
        target_compile_options(${_kairo_target} PRIVATE -Wno-deprecated-declarations)
    endforeach()
endif()
