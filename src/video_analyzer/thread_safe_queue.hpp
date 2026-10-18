#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <utility>

namespace video_analyzer {

template <typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(std::size_t max_size) : max_size_(max_size) {}

    bool push(T&& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] {
            return queue_.size() < max_size_ || finished_;
        });
        if (finished_) {
            return false;
        }

        queue_.push(std::move(item));
        condition_.notify_one();
        return true;
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] {
            return !queue_.empty() || finished_;
        });
        if (queue_.empty()) {
            return false;
        }

        item = std::move(queue_.front());
        queue_.pop();
        condition_.notify_one();
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        finished_ = true;
        condition_.notify_all();
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::size_t max_size_;
    bool finished_ = false;
};

}
