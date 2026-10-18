#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace video_analyzer {

struct AnalysisResult {
    std::string video_path;
    double timestamp;
    int frame_index;
    std::int64_t predicted_class;
};

std::vector<AnalysisResult> run_pipeline(
    const std::vector<std::string>& video_paths,
    const std::string& model_path,
    double sample_interval_secs,
    int batch_size,
    int target_width,
    int target_height,
    bool use_cuda);

}
