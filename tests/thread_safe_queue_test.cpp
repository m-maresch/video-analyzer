#include "video_analyzer/thread_safe_queue.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <future>

TEST(ThreadSafeQueueTest, PushAndPopPreserveFifoOrder) {
    video_analyzer::ThreadSafeQueue<int> queue(3);

    ASSERT_TRUE(queue.push(10));
    ASSERT_TRUE(queue.push(20));
    ASSERT_TRUE(queue.push(30));

    int item = 0;
    ASSERT_TRUE(queue.pop(item));
    EXPECT_EQ(item, 10);
    ASSERT_TRUE(queue.pop(item));
    EXPECT_EQ(item, 20);
    ASSERT_TRUE(queue.pop(item));
    EXPECT_EQ(item, 30);
}

TEST(ThreadSafeQueueTest, ConsumerWaitsUntilItemIsPushed) {
    video_analyzer::ThreadSafeQueue<int> queue(1);
    auto pending_pop = std::async(std::launch::async, [&queue] {
        int item = 0;
        if (!queue.pop(item)) {
            return -1;
        }
        return item;
    });

    EXPECT_EQ(
        pending_pop.wait_for(std::chrono::milliseconds(50)),
        std::future_status::timeout);
    ASSERT_TRUE(queue.push(42));

    ASSERT_EQ(
        pending_pop.wait_for(std::chrono::seconds(1)),
        std::future_status::ready);
    EXPECT_EQ(pending_pop.get(), 42);
}

TEST(ThreadSafeQueueTest, FullQueueBlocksProducerUntilItemIsPopped) {
    video_analyzer::ThreadSafeQueue<int> queue(1);
    ASSERT_TRUE(queue.push(1));

    auto pending_push = std::async(std::launch::async, [&queue] {
        return queue.push(2);
    });
    EXPECT_EQ(
        pending_push.wait_for(std::chrono::milliseconds(50)),
        std::future_status::timeout);

    int item = 0;
    ASSERT_TRUE(queue.pop(item));
    EXPECT_EQ(item, 1);

    ASSERT_EQ(
        pending_push.wait_for(std::chrono::seconds(1)),
        std::future_status::ready);
    EXPECT_TRUE(pending_push.get());
    ASSERT_TRUE(queue.pop(item));
    EXPECT_EQ(item, 2);
}

TEST(ThreadSafeQueueTest, ClosedQueueDrainsItemsBeforePopReturnsFalse) {
    video_analyzer::ThreadSafeQueue<int> queue(2);
    ASSERT_TRUE(queue.push(1));
    ASSERT_TRUE(queue.push(2));

    queue.close();

    int item = 0;
    ASSERT_TRUE(queue.pop(item));
    EXPECT_EQ(item, 1);
    ASSERT_TRUE(queue.pop(item));
    EXPECT_EQ(item, 2);
    EXPECT_FALSE(queue.pop(item));
}

TEST(ThreadSafeQueueTest, ClosingUnblocksProducerAndDrainsQueuedItems) {
    video_analyzer::ThreadSafeQueue<int> queue(1);
    ASSERT_TRUE(queue.push(1));

    auto pending_push = std::async(std::launch::async, [&queue] {
        return queue.push(2);
    });
    EXPECT_EQ(
        pending_push.wait_for(std::chrono::milliseconds(50)),
        std::future_status::timeout);

    queue.close();
    ASSERT_EQ(
        pending_push.wait_for(std::chrono::seconds(1)),
        std::future_status::ready);
    EXPECT_FALSE(pending_push.get());

    int item = 0;
    ASSERT_TRUE(queue.pop(item));
    EXPECT_EQ(item, 1);
    EXPECT_FALSE(queue.pop(item));
}
