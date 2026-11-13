// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <mutex>
#include <string>
#include <utility>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/nx2_profile.h"
#include "video_core/renderer_vulkan/vk_gpu_timing.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

namespace {

namespace GpuWork = VideoCommon::Nx2GpuWork;
namespace Profile = VideoCommon::Nx2Profile;

constexpr u32 POOL_SIZE = 1u << 16;
/// Room a command buffer pair needs to be timed: its four fixed queries and some marks.
constexpr u64 MIN_FREE = 64;
constexpr const char* THREAD_NAME = "host GPU";

} // Anonymous namespace

GpuTiming::GpuTiming(const Device& device_, MasterSemaphore& master_semaphore_)
    : device{device_}, master_semaphore{master_semaphore_} {
    const auto families = device.GetPhysical().GetQueueFamilyProperties();
    const u32 family = device.GetGraphicsFamily();
    const u32 bits = family < families.size() ? families[family].timestampValidBits : 0;
    supported = bits != 0 && device.SupportsGraphicsTimestamps() &&
                device.IsHostQueryResetSupported();
    if (!supported) {
        if (GpuWork::Enabled()) {
            LOG_WARNING(Render_Vulkan,
                        "GPU timestamps unavailable (valid bits {}, host query reset {})", bits,
                        device.IsHostQueryResetSupported());
        }
        return;
    }
    valid_mask = bits >= 64 ? ~u64{0} : (u64{1} << bits) - 1;
    ns_per_tick = static_cast<double>(device.GetTimestampPeriod());
}

GpuTiming::~GpuTiming() = default;

void GpuTiming::Begin(vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf) {
    Collect();
    Report();
    const bool previous_timed = std::exchange(active, false);
    if (!supported || !GpuWork::Enabled()) {
        return;
    }
    if (!pool) {
        pool = device.GetLogical().CreateQueryPool({
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = POOL_SIZE,
            .pipelineStatistics = 0,
        });
        device.GetLogical().ResetQueryPool(*pool, 0, POOL_SIZE);
    }
    if (POOL_SIZE - (head - tail) < MIN_FREE) {
        ++window.untimed;
        return;
    }
    active = true;
    current.first = head;
    current.count = 0;
    current.tick = 0;
    current.after_timed = previous_timed;
    current.works.clear();
    Write(upload_cmdbuf);
    Write(cmdbuf);
    current.works.push_back(Work::Untagged);
}

void GpuTiming::Mark(vk::CommandBuffer cmdbuf, Work work) {
    if (!active) {
        return;
    }
    // Two queries stay free for End.
    if (POOL_SIZE - (head - tail) <= 2) {
        ++window.dropped;
        return;
    }
    Write(cmdbuf);
    current.works.push_back(work);
}

void GpuTiming::End(vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf, u64 tick) {
    if (!active) {
        return;
    }
    Write(upload_cmdbuf);
    Write(cmdbuf);
    current.tick = tick;
    pending.push_back(std::move(current));
    current = {};
}

void GpuTiming::Write(vk::CommandBuffer cmdbuf) {
    cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *pool,
                          static_cast<u32>(head % POOL_SIZE));
    ++head;
    ++current.count;
}

void GpuTiming::Collect() {
    if (pending.empty()) {
        return;
    }
    master_semaphore.Refresh();
    while (!pending.empty()) {
        const Range& range = pending.front();
        if (!master_semaphore.IsFree(range.tick)) {
            break;
        }
        results.resize(range.count);
        const u32 index = static_cast<u32>(range.first % POOL_SIZE);
        const u32 first_part = std::min(range.count, POOL_SIZE - index);
        const u32 second_part = range.count - first_part;
        const auto& logical = device.GetLogical();
        VkResult result = logical.GetQueryResults(*pool, index, first_part,
                                                  first_part * sizeof(u64), results.data(),
                                                  sizeof(u64), VK_QUERY_RESULT_64_BIT);
        if (result == VK_SUCCESS && second_part != 0) {
            result = logical.GetQueryResults(*pool, 0, second_part, second_part * sizeof(u64),
                                             results.data() + first_part, sizeof(u64),
                                             VK_QUERY_RESULT_64_BIT);
        }
        if (result == VK_NOT_READY) {
            break;
        }
        if (result == VK_SUCCESS) {
            Account(range, results.data());
        }
        logical.ResetQueryPool(*pool, index, first_part);
        if (second_part != 0) {
            logical.ResetQueryPool(*pool, 0, second_part);
        }
        tail = range.first + range.count;
        pending.pop_front();
    }
}

void GpuTiming::Account(const Range& range, const u64* values) {
    const auto span_ns = [this](u64 from, u64 to) {
        return static_cast<u64>(static_cast<double>((to - from) & valid_mask) * ns_per_tick);
    };
    const u32 n = range.count;
    // Upload begin, main begin, the marks, upload end, main end.
    const size_t upload = static_cast<size_t>(Work::UploadBuffer);
    window.ns[upload] += span_ns(values[0], values[n - 2]);
    ++window.segments[upload];
    const size_t marks = range.works.size() - 1;
    for (size_t i = 0; i < range.works.size(); ++i) {
        const u64 from = values[1 + i];
        const u64 to = i < marks ? values[2 + i] : values[n - 1];
        const auto work = static_cast<size_t>(range.works[i]);
        window.ns[work] += span_ns(from, to);
        ++window.segments[work];
    }
    if (range.after_timed && last_main_end != 0) {
        // Present submissions, other queues' work and idle time all land here.
        const u64 gap = (values[0] - last_main_end) & valid_mask;
        if (gap < valid_mask / 2) {
            window.gap_ns += static_cast<u64>(static_cast<double>(gap) * ns_per_tick);
            ++window.gaps;
        }
    }
    last_main_end = values[n - 1];
    ++window.command_buffers;
    window.timestamps += n;
}

void GpuTiming::Report() {
    if (!GpuWork::Enabled()) {
        if (published) {
            std::scoped_lock lock{Profile::SnapshotMutex()};
            Profile::Snapshots().erase(THREAD_NAME);
            published = false;
        }
        window = {};
        last_main_end = 0;
        return;
    }
    const u64 now = Profile::Now();
    const u32 generation = Profile::g_generation.load(std::memory_order_relaxed);
    if (window.start == 0 || window.generation != generation) {
        window = {};
        window.start = now;
        window.generation = generation;
        return;
    }
    const u64 span = now - window.start;
    if (span < Profile::g_window_ns.load(std::memory_order_relaxed)) {
        return;
    }
    const double per_s = 1e9 / static_cast<double>(span);
    Profile::Snapshot snap{.thread = THREAD_NAME, .when_ns = now, .window_ns = span};
    std::string line;
    const auto add = [&](const char* name, u64 ns, u64 count) {
        if (ns == 0 && count == 0) {
            return;
        }
        const double ms = static_cast<double>(ns) / 1e6 * per_s;
        const double calls = static_cast<double>(count) * per_s;
        snap.zones.push_back({name, ms, calls});
        line += fmt::format(" | {} {:.1f} ms ({:.0f}/s)", name, ms, calls);
    };
    for (size_t i = 0; i < NumWork; ++i) {
        add(GpuWork::work_names[i], window.ns[i], window.segments[i]);
    }
    add("between command buffers", window.gap_ns, window.gaps);
    snap.counters = {
        {"command_buffers", static_cast<double>(window.command_buffers) * per_s},
        {"untimed_command_buffers", static_cast<double>(window.untimed) * per_s},
        {"timestamps", static_cast<double>(window.timestamps) * per_s},
        {"marks_dropped", static_cast<double>(window.dropped) * per_s},
    };
    LOG_INFO(Render_Vulkan, "{} per second:{} | {:.0f} command buffers, {:.0f} timestamps",
             THREAD_NAME, line, static_cast<double>(window.command_buffers) * per_s,
             static_cast<double>(window.timestamps) * per_s);
    {
        std::scoped_lock lock{Profile::SnapshotMutex()};
        Profile::Snapshots()[THREAD_NAME] = std::move(snap);
    }
    published = true;
    window = {};
    window.start = now;
    window.generation = generation;
}

} // namespace Vulkan
