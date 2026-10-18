#pragma once

#include "video_analyzer/engine.hpp"
#include "video_analyzer/thread_safe_queue.hpp"

#include <torch/script.h>
#include <torch/torch.h>

#include <exception>
#include <mutex>
#include <string>
#include <vector>

namespace video_analyzer {

class VideoPipeline {
public:
    VideoPipeline(
        std::string model_path,
        double sample_interval_secs,
        int batch_size,
        int target_width,
        int target_height,
        bool use_cuda);

    std::vector<AnalysisResult> run(
        const std::vector<std::string>& video_paths);

private:
    struct FrameItem {
        torch::Tensor tensor;
        double timestamp;
        int frame_index;
        std::string video_path;
    };

    void record_failure(std::exception_ptr error);
    void produce_video(const std::string& path);
    void process_batch(
        torch::jit::script::Module& module,
        std::vector<torch::Tensor>& batch_tensors,
        std::vector<FrameItem>& batch_metadata);
    void consume_frames();

    std::string model_path_;
    double sample_interval_secs_;
    int batch_size_;
    int target_width_;
    int target_height_;
    bool use_cuda_;
    ThreadSafeQueue<FrameItem> frame_queue_;
    std::vector<AnalysisResult> results_;
    std::mutex failure_mutex_;
    std::exception_ptr failure_;
};

}
