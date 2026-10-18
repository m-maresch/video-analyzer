cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(video_analyzer_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(
    _video_analyzer_engine
    src/video_analyzer/bindings.cpp
    src/video_analyzer/engine.cpp
    src/video_analyzer/video_pipeline.cpp
)

target_link_libraries(_video_analyzer_engine PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(_video_analyzer_engine PRIVATE src)

target_compile_options(_video_analyzer_engine PRIVATE -O3 -march=native -pthread)

install(TARGETS _video_analyzer_engine LIBRARY DESTINATION .)

option(VIDEO_ANALYZER_BUILD_TESTS "Build C++ tests" OFF)

if(VIDEO_ANALYZER_BUILD_TESTS)
    include(FetchContent)

    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_executable(_video_analyzer_tests
        tests/thread_safe_queue_test.cpp
        tests/engine_test.cpp
        src/video_analyzer/engine.cpp
        src/video_analyzer/video_pipeline.cpp
    )

    target_link_libraries(_video_analyzer_tests PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS} GTest::gtest_main)
    target_include_directories(_video_analyzer_tests PRIVATE src)
    
    include(GoogleTest)
    gtest_discover_tests(_video_analyzer_tests)
endif()
