// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <deque>
#include <vector>

#include "common/common_types.h"
#include "video_core/nx2_gpu_work.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class MasterSemaphore;

/// Host GPU time by kind of work (nx2_gpu_work.h), from timestamps the scheduler writes at the
/// start and end of each command buffer pair and wherever the recorded kind changes. Results are
/// read without waiting once a command buffer's tick has passed, and published every profiler
/// window as the "host GPU" report (Nx2Profile::Snapshots). Vulkan worker thread only.
class GpuTiming {
public:
    explicit GpuTiming(const Device& device, MasterSemaphore& master_semaphore);
    ~GpuTiming();

    GpuTiming(const GpuTiming&) = delete;
    GpuTiming& operator=(const GpuTiming&) = delete;

    /// After both command buffers began: times them when timestamps are on and the ring has room.
    void Begin(vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf);

    /// The commands recorded after this point are `work`.
    void Mark(vk::CommandBuffer cmdbuf, VideoCommon::Nx2GpuWork::Work work);

    /// Before both command buffers end; `tick` is the submission's semaphore value.
    void End(vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf, u64 tick);

private:
    using Work = VideoCommon::Nx2GpuWork::Work;
    static constexpr size_t NumWork = static_cast<size_t>(Work::Count);

    /// One command buffer pair's queries, in ring order: upload begin, main begin, the marks,
    /// upload end, main end. works[i] is the kind of the main segment that starts at query i + 1.
    struct Range {
        u64 first{};
        u32 count{};
        u64 tick{};
        bool after_timed{}; ///< The command buffer before it was timed too (gap measurable)
        std::vector<Work> works;
    };

    struct Window {
        std::array<u64, NumWork> ns{};
        std::array<u64, NumWork> segments{};
        u64 gap_ns{};
        u64 gaps{};
        u64 command_buffers{};
        u64 untimed{};
        u64 timestamps{};
        u64 dropped{};
        u64 start{};
        u32 generation{};
    };

    void Write(vk::CommandBuffer cmdbuf);
    void Collect();
    void Account(const Range& range, const u64* values);
    void Report();

    const Device& device;
    MasterSemaphore& master_semaphore;
    vk::QueryPool pool;
    bool supported{};
    double ns_per_tick{};
    u64 valid_mask{};

    u64 head{}; ///< Next query to write (absolute; the pool index is head % size)
    u64 tail{}; ///< Oldest query not yet read
    bool active{};
    Range current;
    std::deque<Range> pending;
    std::vector<u64> results;

    u64 last_main_end{}; ///< Ticks: the last read command buffer's end
    Window window;
    bool published{};
};

} // namespace Vulkan
