#include "video_analyzer/engine.hpp"

#include <gtest/gtest.h>
#include <torch/script.h>

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace {

std::filesystem::path temporary_model_path() {
    const auto nonce = std::chrono::steady_clock::now()
                           .time_since_epoch()
                           .count();
    return std::filesystem::temp_directory_path() /
           ("video-analyzer-test-" + std::to_string(nonce) + ".pt");
}

}

TEST(AnalyzeVideosTest, RejectsInvalidSampleInterval) {
    EXPECT_THROW(
        video_analyzer::analyze_videos({}, "model.pt", 0.0, 1, 8, 8, false),
        std::invalid_argument);
}

TEST(AnalyzeVideosTest, ReportsModelLoadFailure) {
    const auto missing_model = temporary_model_path();

    try {
        video_analyzer::analyze_videos(
            {}, missing_model.string(), 1.0, 1, 8, 8, false);
        FAIL() << "Expected a model loading error";
    } catch (const std::runtime_error& error) {
        const std::string expected_prefix =
            "Failed to load model '" + missing_model.string() + "':";
        const std::string message = error.what();
        EXPECT_EQ(message.substr(0, expected_prefix.size()), expected_prefix);
    }
}

TEST(AnalyzeVideosTest, ReportsUnreadableVideo) {
    const auto model_path = temporary_model_path();
    const auto missing_video =
        std::filesystem::temp_directory_path() /
        ("video-analyzer-missing-" + model_path.stem().string() + ".mp4");

    torch::jit::Module model("IdentityModel");
    model.define("def forward(self, input):\n    return input\n");
    model.save(model_path.string());

    try {
        video_analyzer::analyze_videos(
            {missing_video.string()}, model_path.string(), 1.0, 1, 8, 8, false);
        FAIL() << "Expected an unreadable video error";
    } catch (const std::runtime_error& error) {
        const std::string expected_message =
            "Failed to open video: " + missing_video.string();
        EXPECT_EQ(error.what(), expected_message);
    }

    std::filesystem::remove(model_path);
}
