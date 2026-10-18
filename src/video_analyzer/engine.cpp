#include "video_analyzer/engine.hpp"
#include "video_analyzer/thread_safe_queue.hpp"

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace video_analyzer {

struct FrameItem {
    torch::Tensor tensor;
    double timestamp;
    int frame_index;
    std::string video_path;
};

std::vector<AnalysisResult> run_pipeline(
    const std::vector<std::string>& video_paths,
    const std::string& model_path,
    double sample_interval_secs,
    int batch_size,
    int target_width,
    int target_height,
    bool use_cuda) {
    ThreadSafeQueue<FrameItem> frame_queue(
        static_cast<std::size_t>(batch_size) * 4);
    std::vector<AnalysisResult> results;
    std::mutex results_mutex;

    std::vector<std::thread> producers;
    producers.reserve(video_paths.size());

    for (const auto& path_str : video_paths) {
        producers.emplace_back([&, path_str] {
            cv::VideoCapture cap(path_str);
            if (!cap.isOpened()) {
                return;
            }

            double fps = cap.get(cv::CAP_PROP_FPS);
            if (fps <= 0) {
                fps = 30.0;
            }
            const int frame_jump = std::max(
                1, static_cast<int>(fps * sample_interval_secs));

            cv::Mat frame;
            int frame_idx = 0;
            while (cap.read(frame)) {
                if (frame_idx % frame_jump == 0) {
                    const double timestamp =
                        cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
                    cv::Mat resized;
                    cv::Mat rgb;
                    cv::resize(
                        frame, resized, cv::Size(target_width, target_height));
                    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

                    auto tensor = torch::from_blob(
                                      rgb.data,
                                      {rgb.rows, rgb.cols, rgb.channels()},
                                      torch::kUInt8)
                                      .permute({2, 0, 1})
                                      .to(torch::kFloat32)
                                      .div_(255.0)
                                      .clone();

                    if (!frame_queue.push(FrameItem{
                            std::move(tensor), timestamp, frame_idx, path_str})) {
                        break;
                    }
                }
                ++frame_idx;
            }
            cap.release();
        });
    }

    std::thread producer_closer([&] {
        for (auto& producer : producers) {
            producer.join();
        }
        frame_queue.set_finished();
    });

    std::thread consumer([&] {
        torch::jit::script::Module module;
        try {
            module = torch::jit::load(model_path);
            module.eval();
            if (use_cuda && torch::cuda::is_available()) {
                module.to(torch::kCUDA);
            }
        } catch (const c10::Error&) {
            return;
        }

        std::vector<torch::Tensor> batch_tensors;
        std::vector<FrameItem> batch_metadata;
        batch_tensors.reserve(batch_size);
        batch_metadata.reserve(batch_size);

        const auto process_batch = [&] {
            auto batch_tensor = torch::stack(batch_tensors);
            if (use_cuda && torch::cuda::is_available()) {
                batch_tensor = batch_tensor.to(torch::kCUDA);
            }

            const auto predictions =
                module.forward({batch_tensor}).toTensor().cpu().argmax(1);

            std::lock_guard<std::mutex> lock(results_mutex);
            for (std::size_t i = 0; i < batch_metadata.size(); ++i) {
                results.push_back({
                    batch_metadata[i].video_path,
                    batch_metadata[i].timestamp,
                    batch_metadata[i].frame_index,
                    predictions[i].item<std::int64_t>()});
            }
            batch_tensors.clear();
            batch_metadata.clear();
        };

        FrameItem item;
        while (frame_queue.pop(item)) {
            batch_tensors.push_back(item.tensor);
            batch_metadata.push_back(std::move(item));

            if (batch_tensors.size() == static_cast<std::size_t>(batch_size)) {
                process_batch();
            }
        }

        if (!batch_tensors.empty()) {
            process_batch();
        }
    });

    producer_closer.join();
    consumer.join();

    return results;
}

}
