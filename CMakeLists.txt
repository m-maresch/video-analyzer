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
)

target_link_libraries(_video_analyzer_engine PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(_video_analyzer_engine PRIVATE src)

target_compile_options(_video_analyzer_engine PRIVATE -O3 -march=native -pthread)

install(TARGETS _video_analyzer_engine LIBRARY DESTINATION .)
