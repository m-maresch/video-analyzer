#include "video_analyzer/video_pipeline.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <thread>
#include <utility>

namespace video_analyzer {

VideoPipeline::VideoPipeline(
    std::string model_path,
    double sample_interval_secs,
    int batch_size,
    int target_width,
    int target_height,
    bool use_cuda)
    : model_path_(std::move(model_path)),
      sample_interval_secs_(sample_interval_secs),
      batch_size_(batch_size),
      target_width_(target_width),
      target_height_(target_height),
      use_cuda_(use_cuda),
      frame_queue_(batch_size_ * frame_queue_batch_capacity_) {}

void VideoPipeline::record_failure(std::exception_ptr error) {
    {
        std::lock_guard<std::mutex> lock(failure_mutex_);
        if (!failure_) {
            failure_ = std::move(error);
        }
    }
    frame_queue_.close();
}

void VideoPipeline::produce_video(const std::string& path) {
    try {
        cv::VideoCapture cap(path);
        if (!cap.isOpened()) {
            throw std::runtime_error("Failed to open video: " + path);
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) {
            fps = 30.0;
        }
        const int frame_jump =
            std::max(1, static_cast<int>(fps * sample_interval_secs_));

        cv::Mat frame;
        int frame_idx = 0;
        while (cap.read(frame)) {
            if (frame_idx % frame_jump == 0) {
                const double timestamp =
                    cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
                cv::Mat resized;
                cv::Mat rgb;
                cv::resize(
                    frame,
                    resized,
                    cv::Size(target_width_, target_height_));
                cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

                auto tensor = torch::from_blob(
                                  rgb.data,
                                  {rgb.rows, rgb.cols, rgb.channels()},
                                  torch::kUInt8)
                                  .permute({2, 0, 1})
                                  .to(torch::kFloat32)
                                  .div_(255.0)
                                  .clone();

                if (!frame_queue_.push(FrameItem{
                        std::move(tensor), timestamp, frame_idx, path})) {
                    return;
                }
            }
            ++frame_idx;
        }
    } catch (...) {
        record_failure(std::current_exception());
    }
}

void VideoPipeline::process_batch(
    torch::jit::script::Module& module,
    std::vector<torch::Tensor>& batch_tensors,
    std::vector<FrameItem>& batch_metadata) {
    auto batch_tensor = torch::stack(batch_tensors);
    if (use_cuda_ && torch::cuda::is_available()) {
        batch_tensor = batch_tensor.to(torch::kCUDA);
    }

    const auto predictions =
        module.forward({batch_tensor}).toTensor().cpu().argmax(1);

    for (std::size_t i = 0; i < batch_metadata.size(); ++i) {
        results_.push_back({
            batch_metadata[i].video_path,
            batch_metadata[i].timestamp,
            batch_metadata[i].frame_index,
            predictions[i].item<std::int64_t>()});
    }
    batch_tensors.clear();
    batch_metadata.clear();
}

void VideoPipeline::consume_frames() {
    try {
        torch::jit::script::Module module;
        try {
            module = torch::jit::load(model_path_);
        } catch (const std::exception& error) {
            throw std::runtime_error(
                "Failed to load model '" + model_path_ +
                "': " + error.what());
        }
        module.eval();
        if (use_cuda_ && torch::cuda::is_available()) {
            module.to(torch::kCUDA);
        }
        torch::InferenceMode inference_mode;

        std::vector<torch::Tensor> batch_tensors;
        std::vector<FrameItem> batch_metadata;
        batch_tensors.reserve(static_cast<std::size_t>(batch_size_));
        batch_metadata.reserve(static_cast<std::size_t>(batch_size_));

        FrameItem item;
        while (frame_queue_.pop(item)) {
            batch_tensors.push_back(item.tensor);
            batch_metadata.push_back(std::move(item));

            if (batch_tensors.size() == static_cast<std::size_t>(batch_size_)) {
                process_batch(module, batch_tensors, batch_metadata);
            }
        }

        if (!batch_tensors.empty()) {
            process_batch(module, batch_tensors, batch_metadata);
        }
    } catch (...) {
        record_failure(std::current_exception());
    }
}

std::vector<AnalysisResult> VideoPipeline::run(
    const std::vector<std::string>& video_paths) {
    std::vector<std::thread> producers;
    producers.reserve(video_paths.size());

    std::thread consumer(&VideoPipeline::consume_frames, this);
    try {
        for (const auto& path : video_paths) {
            producers.emplace_back(&VideoPipeline::produce_video, this, path);
        }
    } catch (...) {
        record_failure(std::current_exception());
    }

    for (auto& producer : producers) {
        producer.join();
    }
    frame_queue_.close();
    consumer.join();

    if (failure_) {
        std::rethrow_exception(failure_);
    }
    return std::move(results_);
}

}
