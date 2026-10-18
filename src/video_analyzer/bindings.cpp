#include "video_analyzer/engine.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_video_analyzer_engine, module) {
    module.doc() = "Video Analyzer Engine";

    py::class_<video_analyzer::AnalysisResult>(module, "AnalysisResult")
        .def_readwrite(
            "video_path", &video_analyzer::AnalysisResult::video_path)
        .def_readwrite(
            "timestamp", &video_analyzer::AnalysisResult::timestamp)
        .def_readwrite(
            "frame_index", &video_analyzer::AnalysisResult::frame_index)
        .def_readwrite(
            "predicted_class",
            &video_analyzer::AnalysisResult::predicted_class);

    module.def(
        "run_pipeline",
        &video_analyzer::run_pipeline,
        "Run concurrent video ingestion and batched LibTorch inference",
        py::arg("video_paths"),
        py::arg("model_path"),
        py::arg("sample_interval_secs") = 1.0,
        py::arg("batch_size") = 16,
        py::arg("target_width") = 224,
        py::arg("target_height") = 224,
        py::arg("use_cuda") = false);
}
