#include "video_analyzer/engine.hpp"
#include "video_analyzer/video_pipeline.hpp"

#include <cmath>
#include <stdexcept>

namespace video_analyzer {

std::vector<AnalysisResult> analyze_videos(
    const std::vector<std::string>& video_paths,
    const std::string& model_path,
    double sample_interval_secs,
    int batch_size,
    int target_width,
    int target_height,
    bool use_cuda) {
    if (!std::isfinite(sample_interval_secs) || sample_interval_secs <= 0) {
        throw std::invalid_argument(
            "sample_interval_secs must be greater than zero");
    }
    if (batch_size <= 0) {
        throw std::invalid_argument("batch_size must be greater than zero");
    }
    if (target_width <= 0 || target_height <= 0) {
        throw std::invalid_argument(
            "target dimensions must be greater than zero");
    }

    VideoPipeline pipeline(
        model_path,
        sample_interval_secs,
        batch_size,
        target_width,
        target_height,
        use_cuda);
    return pipeline.run(video_paths);
}

}
