#include <gtest/gtest.h>
#include "io/swf_reader.h"

static const std::string SWF_PATH = "../tests/test.swf";

// ── Header parsing ────────────────────────────────────────────────────────────

TEST(SwfReaderTest, OpensWithoutThrowing) {
    EXPECT_NO_THROW(SwfReader reader(SWF_PATH); reader.read(););
}

TEST(SwfReaderTest, MaxNodesFromHeader) {
    SwfReader reader(SWF_PATH);
    reader.read();
    EXPECT_EQ(reader.max_nodes(), 4360);
}

TEST(SwfReaderTest, MaxProcsFromHeader) {
    SwfReader reader(SWF_PATH);
    reader.read();
    EXPECT_EQ(reader.max_procs(), 4360);
}

TEST(SwfReaderTest, UnixStartTimeFromHeader) {
    SwfReader reader(SWF_PATH);
    reader.read();
    EXPECT_EQ(reader.unix_start_time(), 0);
}

// ── Job count ─────────────────────────────────────────────────────────────────

TEST(SwfReaderTest, ReturnsCorrectJobCount) {
    SwfReader reader(SWF_PATH);
    auto jobs = reader.read();
    EXPECT_EQ(static_cast<int>(jobs.size()), 500);
}

// ── First job fields ──────────────────────────────────────────────────────────

TEST(SwfReaderTest, FirstJobFields) {
    SwfReader reader(SWF_PATH);
    auto jobs = reader.read();
    ASSERT_FALSE(jobs.empty());

    const SimJob& j = jobs[0];
    EXPECT_EQ(j.id,          1);
    EXPECT_EQ(j.submit_time, 1641021254);
    EXPECT_EQ(j.walltime,    10800);
    EXPECT_EQ(j.run_time,    10849);
    EXPECT_EQ(j.requested.procs, 128);
}

// ── Sanity checks across all jobs ─────────────────────────────────────────────

TEST(SwfReaderTest, AllJobsStartQueued) {
    SwfReader reader(SWF_PATH);
    auto jobs = reader.read();
    for (const auto& j : jobs) {
        EXPECT_EQ(j.state, JobState::Queued) << "job id=" << j.id;
    }
}

TEST(SwfReaderTest, NoJobHasZeroProcs) {
    SwfReader reader(SWF_PATH);
    auto jobs = reader.read();
    for (const auto& j : jobs) {
        EXPECT_GT(j.requested.procs, 0) << "job id=" << j.id;
    }
}

TEST(SwfReaderTest, ThrowsOnBadPath) {
    SwfReader reader("nonexistent.swf");
    EXPECT_THROW(reader.read(), std::runtime_error);
}
