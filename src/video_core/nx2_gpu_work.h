// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstdlib>

#include "common/common_types.h"

// Host GPU time by kind of work (EDEN_NX2_GPU_TIMESTAMPS=1, or gpu_profile gpu_timestamps). The
// thread recording Vulkan commands names what it records with a Tag; the scheduler writes a GPU
// timestamp whenever that kind changes, so every command in a timed command buffer counts
// towards one kind (Vulkan::GpuTiming).
namespace VideoCommon::Nx2GpuWork {

enum class Work : u8 {
    Untagged,        ///< Recorded outside every Tag
    Draw,            ///< Game draws, their state and descriptor binds
    Clear,           ///< Game clears
    Compute,         ///< Game compute dispatches
    RenderPassBegin, ///< vkCmdBeginRenderPass (load ops, clears) and the barrier after compute
    RenderPassEnd,   ///< vkCmdEndRenderPass (store ops) and its barrier: one per pass break
    BufferUpload,    ///< Guest buffer data copied to host buffers in the main command buffer
    BufferCopy,      ///< Buffer to buffer copies, fills, and their barriers
    BufferDownload,  ///< Buffer data copied back for the CPU
    ImageUpload,     ///< Guest texture data copied to images
    ImageCopy,       ///< Image copies, blits, resolves and format conversions
    ImageDownload,   ///< Image data copied back for the CPU
    AstcDecode,      ///< ASTC decode (and BC recompression) compute passes
    Upscale,         ///< DLSS replacement: FSR 3.1, the temporal stand-in, upscale blits
    Present,         ///< Frame output to the window (blit_screen, filters)
    Queries,         ///< Query resets and result copies
    ListCopy,        ///< Command list copies the GPU thread reads back
    SubmitEnd,       ///< The barrier closing each command buffer
    UploadBuffer,    ///< The upload command buffer submitted ahead of each main one
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Work::Count)> work_names{
    "untagged",       "draw",           "clear",          "compute",
    "render pass begin", "render pass end", "buffer upload", "buffer copy",
    "buffer download", "image upload",  "image copy",     "image download",
    "astc decode",    "upscale",        "present",        "queries",
    "list copy",      "submit end",     "upload cmdbuf",
};

inline std::atomic<bool> g_enabled{[] {
    const char* value = std::getenv("EDEN_NX2_GPU_TIMESTAMPS");
    return value != nullptr && value[0] != '0';
}()};

inline bool Enabled() {
    return g_enabled.load(std::memory_order_relaxed);
}

inline thread_local Work t_work{Work::Untagged};

inline Work Current() {
    return t_work;
}

/// Names the GPU work recorded in its scope (nests).
class Tag {
public:
    explicit Tag(Work work) : previous{t_work} {
        t_work = work;
    }
    ~Tag() {
        t_work = previous;
    }

    Tag(const Tag&) = delete;
    Tag& operator=(const Tag&) = delete;

private:
    Work previous;
};

/// A Tag that applies only inside a broad kind (draws, clears, dispatches, untagged): a copy made
/// for a draw counts as a copy, one made for a download keeps the outer download tag.
class InnerTag {
public:
    explicit InnerTag(Work work) : previous{t_work} {
        switch (previous) {
        case Work::Untagged:
        case Work::Draw:
        case Work::Clear:
        case Work::Compute:
        case Work::RenderPassBegin:
        case Work::RenderPassEnd:
            t_work = work;
            break;
        default:
            break;
        }
    }
    ~InnerTag() {
        t_work = previous;
    }

    InnerTag(const InnerTag&) = delete;
    InnerTag& operator=(const InnerTag&) = delete;

private:
    Work previous;
};

} // namespace VideoCommon::Nx2GpuWork
