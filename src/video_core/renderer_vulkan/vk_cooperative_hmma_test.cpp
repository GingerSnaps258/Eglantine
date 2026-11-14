// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "video_core/renderer_vulkan/vk_cooperative_hmma_test.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
namespace {

using Shader::Backend::SPIRV::COOPERATIVE_HMMA_TEST_LANE_WORDS;

constexpr u32 LANES = GuestWarpSize;
constexpr u32 BUFFER_WORDS = LANES * COOPERATIVE_HMMA_TEST_LANE_WORDS;

/// The half-precision encoding of a small integer. Products and sums of the test's integers stay
/// exact in half precision, so the results compare bit for bit, without any rounding involved.
u16 IntegerToHalf(s32 value) {
    if (value == 0) {
        return 0;
    }
    const u32 magnitude = static_cast<u32>(value < 0 ? -value : value);
    const u32 exponent = static_cast<u32>(std::bit_width(magnitude)) - 1;
    const u32 mantissa = (magnitude << (10 - exponent)) & 0x3ff;
    return static_cast<u16>((value < 0 ? 0x8000U : 0U) | ((exponent + 15) << 10) | mantissa);
}

u32 PackHalves(s32 low, s32 high) {
    return IntegerToHalf(low) | (static_cast<u32>(IntegerToHalf(high)) << 16);
}

struct Matrices {
    std::array<std::array<s32, 16>, 16> a{};
    std::array<std::array<s32, 8>, 16> b{};
    std::array<std::array<s32, 8>, 16> c{};
    std::array<std::array<s32, 8>, 16> d{};
};

Matrices MakeMatrices() {
    // Every element differs from most others, so that any misplaced element changes the results.
    u32 state = 0x2545f491;
    const auto next = [&state] {
        state = state * 1664525U + 1013904223U;
        return static_cast<s32>((state >> 16) % 7) - 3;
    };
    Matrices m;
    for (auto& row : m.a) {
        for (s32& value : row) {
            value = next();
        }
    }
    for (auto& row : m.b) {
        for (s32& value : row) {
            value = next();
        }
    }
    for (auto& row : m.c) {
        for (s32& value : row) {
            value = next();
        }
    }
    for (size_t row = 0; row < 16; ++row) {
        for (size_t column = 0; column < 8; ++column) {
            s32 sum = m.c[row][column];
            for (size_t k = 0; k < 16; ++k) {
                sum += m.a[row][k] * m.b[k][column];
            }
            m.d[row][column] = sum;
        }
    }
    return m;
}

/// Writes one lane's guest registers, in the layout of mma.m16n8k16: with g = lane / 4 and
/// t = lane % 4, a lane holds rows g and g + 8 of A, C and D, and column g of B.
void WriteLane(const Matrices& m, u32 lane, std::span<u32> words) {
    const size_t g = lane >> 2;
    const size_t t = lane & 3;
    const size_t k = t * 2;
    words[0] = PackHalves(m.a[g][k], m.a[g][k + 1]);
    words[1] = PackHalves(m.a[g + 8][k], m.a[g + 8][k + 1]);
    words[2] = PackHalves(m.a[g][k + 8], m.a[g][k + 9]);
    words[3] = PackHalves(m.a[g + 8][k + 8], m.a[g + 8][k + 9]);
    words[4] = PackHalves(m.b[k][g], m.b[k + 1][g]);
    words[5] = PackHalves(m.b[k + 8][g], m.b[k + 9][g]);
    words[6] = PackHalves(m.c[g][k], m.c[g][k + 1]);
    words[7] = PackHalves(m.c[g + 8][k], m.c[g + 8][k + 1]);
    words[8] = 0;
    words[9] = 0;
}

bool SameHalf(u32 got, s32 expected) {
    const u32 want = IntegerToHalf(expected);
    // A sum that cancels to zero may have either sign.
    return got == want || (expected == 0 && (got & 0x7fff) == 0);
}

u32 CountWrongLanes(const Matrices& m, std::span<const u32> words) {
    u32 wrong = 0;
    for (u32 lane = 0; lane < LANES; ++lane) {
        const size_t g = lane >> 2;
        const size_t k = (lane & 3) * 2;
        const auto lane_words = words.subspan(lane * COOPERATIVE_HMMA_TEST_LANE_WORDS);
        const u32 d0 = lane_words[8];
        const u32 d1 = lane_words[9];
        if (!SameHalf(d0 & 0xffff, m.d[g][k]) || !SameHalf(d0 >> 16, m.d[g][k + 1]) ||
            !SameHalf(d1 & 0xffff, m.d[g + 8][k]) || !SameHalf(d1 >> 16, m.d[g + 8][k + 1])) {
            ++wrong;
        }
    }
    return wrong;
}

bool RunTest(const Device& device) {
    const auto& logical = device.GetLogical();
    const std::vector<u32> code =
        Shader::Backend::SPIRV::EmitCooperativeHmmaTest(device.SupportedSpirvVersion());
    const vk::ShaderModule module = logical.CreateShaderModule(VkShaderModuleCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = code.size() * sizeof(u32),
        .pCode = code.data(),
    });
    const VkDescriptorSetLayoutBinding binding{
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = nullptr,
    };
    const vk::DescriptorSetLayout set_layout =
        logical.CreateDescriptorSetLayout(VkDescriptorSetLayoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = 1,
            .pBindings = &binding,
        });
    const VkDescriptorSetLayout set_layout_handle = *set_layout;
    const vk::PipelineLayout layout = logical.CreatePipelineLayout(VkPipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout_handle,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr,
    });
    const vk::Pipeline pipeline = logical.CreateComputePipeline(VkComputePipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = *module,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
        .layout = *layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0,
    });
    const VkDescriptorPoolSize pool_size{
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
    };
    const vk::DescriptorPool pool = logical.CreateDescriptorPool(VkDescriptorPoolCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    });
    const vk::DescriptorSets sets = pool.Allocate(VkDescriptorSetAllocateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = *pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &set_layout_handle,
    });
    const VkDescriptorSet set = sets[0];

    const MemoryAllocator allocator{device};
    vk::Buffer buffer = allocator.CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = BUFFER_WORDS * sizeof(u32),
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        },
        MemoryUsage::Download);

    const Matrices matrices = MakeMatrices();
    std::array<u32, BUFFER_WORDS> words{};
    for (u32 lane = 0; lane < LANES; ++lane) {
        WriteLane(matrices, lane,
                  std::span{words}.subspan(lane * COOPERATIVE_HMMA_TEST_LANE_WORDS,
                                           COOPERATIVE_HMMA_TEST_LANE_WORDS));
    }
    std::memcpy(buffer.Mapped().data(), words.data(), sizeof(words));
    buffer.Flush();

    const VkDescriptorBufferInfo buffer_info{
        .buffer = *buffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    const VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pImageInfo = nullptr,
        .pBufferInfo = &buffer_info,
        .pTexelBufferView = nullptr,
    };
    logical.UpdateDescriptorSets(write, {});

    // Its own submission: this runs while the renderer is being built, before the scheduler can
    // flush (it needs the query cache) and before anything else submits to the queue.
    const vk::CommandPool command_pool = logical.CreateCommandPool(VkCommandPoolCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = device.GetGraphicsFamily(),
    });
    const vk::CommandBuffers command_buffers = command_pool.Allocate(1);
    const vk::CommandBuffer cmdbuf{command_buffers[0], device.GetDispatchLoader()};
    cmdbuf.Begin(VkCommandBufferBeginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
    cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, set, {});
    cmdbuf.Dispatch(1, 1, 1);
    const VkMemoryBarrier read_results{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                           read_results);
    cmdbuf.End();
    const vk::Fence fence = logical.CreateFence(VkFenceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    });
    const VkCommandBuffer submitted = *cmdbuf;
    const VkSubmitInfo submit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = nullptr,
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers = &submitted,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr,
    };
    vk::Check(device.GetGraphicsQueue().Submit(submit, *fence));
    vk::Check(fence.Wait());

    buffer.Invalidate();
    std::memcpy(words.data(), buffer.Mapped().data(), sizeof(words));
    const u32 wrong = CountWrongLanes(matrices, words);
    if (wrong != 0) {
        LOG_WARNING(Render_Vulkan,
                    "Cooperative matrices lay out elements differently on this driver ({} of {} "
                    "lanes wrong): guest tensor operations stay in shader emulation",
                    wrong, LANES);
        return false;
    }
    LOG_INFO(Render_Vulkan, "Guest tensor operations run on cooperative matrices");
    return true;
}

// Transposed path (EDEN_NX2_COOP_HMMA=transposed).

using Shader::Backend::SPIRV::TRANSPOSED_HMMA_TEST_INVOCATIONS;
using Shader::Backend::SPIRV::TRANSPOSED_HMMA_TEST_LANE_WORDS;

/// The shader's own workgroup memory beside the tiles: as much as the Ocarina of Time network
/// pass this path was measured on, and not a multiple of the tile size.
constexpr u32 TRANSPOSED_TEST_SHARED_BYTES = 6912;
constexpr u32 TRANSPOSED_TEST_WORDS = TRANSPOSED_HMMA_TEST_INVOCATIONS * TRANSPOSED_HMMA_TEST_LANE_WORDS;

using Tile16x8 = std::array<std::array<u16, 8>, 16>;

struct HalfMatrices {
    std::array<std::array<u16, 16>, 16> a{};
    Tile16x8 b{};
    Tile16x8 c{};
};

double HalfToDouble(u16 half) {
    const double sign = (half & 0x8000) != 0 ? -1.0 : 1.0;
    const u32 exponent = (half >> 10) & 0x1f;
    const u32 mantissa = half & 0x3ff;
    if (exponent == 0) {
        return sign * std::ldexp(static_cast<double>(mantissa), -24);
    }
    if (exponent == 31) {
        return mantissa != 0 ? std::nan("") : sign * HUGE_VAL;
    }
    return sign * std::ldexp(static_cast<double>(mantissa | 0x400), static_cast<int>(exponent) - 25);
}

/// Round to nearest, ties to even, overflowing to infinity, as the guest's HMMA.F16 rounds.
u16 RoundToHalf(double value) {
    const u16 sign = std::signbit(value) ? 0x8000 : 0;
    const double magnitude = std::fabs(value);
    if (magnitude >= 65520.0) {
        return sign | 0x7c00;
    }
    if (magnitude < std::ldexp(1.0, -14)) {
        return sign | static_cast<u16>(std::nearbyint(magnitude * 16777216.0));
    }
    int exponent;
    std::frexp(magnitude, &exponent);
    double mantissa = std::nearbyint(std::ldexp(magnitude, 11 - exponent));
    if (mantissa >= 2048.0) {
        mantissa /= 2;
        ++exponent;
    }
    if (exponent + 14 > 30) {
        return sign | 0x7c00;
    }
    return sign | static_cast<u16>(((exponent + 14) << 10) | (static_cast<u32>(mantissa) & 0x3ff));
}

/// D = A * B + C rounded after each K=8 group (per_group) or once. The test inputs keep every
/// sum exact in double, so this is also what f32 accumulation in any order gives.
Tile16x8 Multiply(const HalfMatrices& m, const Tile16x8& c, bool per_group) {
    Tile16x8 d{};
    for (size_t row = 0; row < 16; ++row) {
        for (size_t column = 0; column < 8; ++column) {
            double sum = HalfToDouble(c[row][column]);
            for (size_t k = 0; k < 16; ++k) {
                if (per_group && k == 8) {
                    sum = HalfToDouble(RoundToHalf(sum));
                }
                sum += HalfToDouble(m.a[row][k]) * HalfToDouble(m.b[k][column]);
            }
            d[row][column] = RoundToHalf(sum);
        }
    }
    return d;
}

HalfMatrices IntegerHalfMatrices(u32 seed) {
    u32 state = seed;
    const auto next = [&state] {
        state = state * 1664525U + 1013904223U;
        return IntegerToHalf(static_cast<s32>((state >> 16) % 7) - 3);
    };
    HalfMatrices m;
    for (auto& row : m.a) {
        for (u16& value : row) {
            value = next();
        }
    }
    for (auto* tile : {&m.b, &m.c}) {
        for (auto& row : *tile) {
            for (u16& value : row) {
                value = next();
            }
        }
    }
    return m;
}

/// Each K half adds 0.5 to values near +-1024, where f16 steps by 1: the guest's rounding after
/// the first half lands on a tie and goes to the even neighbour, rounding once does not.
HalfMatrices RoundingHalfMatrices(bool negative) {
    constexpr u16 HALF_ONE = 0x3c00;
    constexpr u16 HALF_POINT_FIVE = 0x3800;
    HalfMatrices m;
    for (size_t row = 0; row < 16; ++row) {
        m.a[row][row % 8] = HALF_POINT_FIVE;
        m.a[row][8 + (row * 3) % 8] = HALF_POINT_FIVE;
        for (size_t column = 0; column < 8; ++column) {
            const double value = (negative ? -1.0 : 1.0) * static_cast<double>(1024 + row + column);
            m.c[row][column] = RoundToHalf(value);
        }
    }
    for (auto& row : m.b) {
        row.fill(HALF_ONE);
    }
    return m;
}

u32 Pack(u16 low, u16 high) {
    return low | (static_cast<u32>(high) << 16);
}

/// The guest's mma.m16n8k16 registers of lane t of a warp (g = t / 4, k = 2 * (t % 4)).
void WriteTransposedLane(const HalfMatrices& m, u32 t, std::span<u32> words) {
    const size_t g = t >> 2;
    const size_t k = (t & 3) * 2;
    words[0] = Pack(m.a[g][k], m.a[g][k + 1]);
    words[1] = Pack(m.a[g + 8][k], m.a[g + 8][k + 1]);
    words[2] = Pack(m.a[g][k + 8], m.a[g][k + 9]);
    words[3] = Pack(m.a[g + 8][k + 8], m.a[g + 8][k + 9]);
    words[4] = Pack(m.b[k][g], m.b[k + 1][g]);
    words[5] = Pack(m.b[k + 8][g], m.b[k + 9][g]);
    words[6] = Pack(m.c[g][k], m.c[g][k + 1]);
    words[7] = Pack(m.c[g + 8][k], m.c[g + 8][k + 1]);
}

bool SameHalves(u32 got, u16 low, u16 high) {
    const auto same = [](u16 a, u16 b) { return a == b || ((a | b) & 0x7fff) == 0; };
    return same(static_cast<u16>(got), low) && same(static_cast<u16>(got >> 16), high);
}

/// Runs a single-workgroup compute shader on `words` (set 0, binding 0) with 32-lane, full
/// subgroups, as the emulator creates pipelines with cooperative matrices.
void RunTransposedShader(const Device& device, std::span<const u32> code, std::span<u32> words) {
    const auto& logical = device.GetLogical();
    const vk::ShaderModule module = logical.CreateShaderModule(VkShaderModuleCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = code.size() * sizeof(u32),
        .pCode = code.data(),
    });
    const VkDescriptorSetLayoutBinding binding{
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = nullptr,
    };
    const vk::DescriptorSetLayout set_layout =
        logical.CreateDescriptorSetLayout(VkDescriptorSetLayoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = 1,
            .pBindings = &binding,
        });
    const VkDescriptorSetLayout set_layout_handle = *set_layout;
    const vk::PipelineLayout layout = logical.CreatePipelineLayout(VkPipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout_handle,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr,
    });
    const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .pNext = nullptr,
        .requiredSubgroupSize = GuestWarpSize,
    };
    const vk::Pipeline pipeline = logical.CreateComputePipeline(VkComputePipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = &subgroup_size,
            .flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = *module,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
        .layout = *layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0,
    });
    // Intel's driver has returned VK_SUCCESS with no pipeline for shaders it could not compile.
    if (*pipeline == VK_NULL_HANDLE) {
        throw std::runtime_error("the driver returned no pipeline");
    }
    const VkDescriptorPoolSize pool_size{
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
    };
    const vk::DescriptorPool pool = logical.CreateDescriptorPool(VkDescriptorPoolCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    });
    const vk::DescriptorSets sets = pool.Allocate(VkDescriptorSetAllocateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = *pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &set_layout_handle,
    });
    const VkDescriptorSet set = sets[0];

    const MemoryAllocator allocator{device};
    vk::Buffer buffer = allocator.CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = words.size_bytes(),
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        },
        MemoryUsage::Download);
    std::memcpy(buffer.Mapped().data(), words.data(), words.size_bytes());
    buffer.Flush();
    const VkDescriptorBufferInfo buffer_info{
        .buffer = *buffer,
        .offset = 0,
        .range = VK_WHOLE_SIZE,
    };
    const VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pImageInfo = nullptr,
        .pBufferInfo = &buffer_info,
        .pTexelBufferView = nullptr,
    };
    logical.UpdateDescriptorSets(write, {});

    // Its own submission, like RunTest: this runs while the renderer is being built.
    const vk::CommandPool command_pool = logical.CreateCommandPool(VkCommandPoolCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = device.GetGraphicsFamily(),
    });
    const vk::CommandBuffers command_buffers = command_pool.Allocate(1);
    const vk::CommandBuffer cmdbuf{command_buffers[0], device.GetDispatchLoader()};
    cmdbuf.Begin(VkCommandBufferBeginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *pipeline);
    cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, *layout, 0, set, {});
    cmdbuf.Dispatch(1, 1, 1);
    const VkMemoryBarrier read_results{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                           read_results);
    cmdbuf.End();
    const vk::Fence fence = logical.CreateFence(VkFenceCreateInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    });
    const VkCommandBuffer submitted = *cmdbuf;
    const VkSubmitInfo submit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = nullptr,
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers = &submitted,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr,
    };
    vk::Check(device.GetGraphicsQueue().Submit(submit, *fence));
    vk::Check(fence.Wait());
    buffer.Invalidate();
    std::memcpy(words.data(), buffer.Mapped().data(), words.size_bytes());
}

/// One layout of the shader's own workgroup memory. Returns an empty string on success.
std::string RunTransposedTest(const Device& device, bool explicit_layout) {
    // Four warps, each with its own matrices, so a tile mixed up between warps shows.
    const std::array<HalfMatrices, 4> warps{IntegerHalfMatrices(0x2545f491), IntegerHalfMatrices(0x9e3779b9),
                                             RoundingHalfMatrices(false), RoundingHalfMatrices(true)};
    u32 rounding_cases = 0;
    for (const HalfMatrices& m : warps) {
        const Tile16x8 per_group = Multiply(m, m.c, true);
        const Tile16x8 once = Multiply(m, m.c, false);
        for (size_t row = 0; row < 16; ++row) {
            for (size_t column = 0; column < 8; ++column) {
                rounding_cases += per_group[row][column] != once[row][column];
            }
        }
    }
    if (rounding_cases == 0) {
        return "no test case tells the guest's rounding apart";
    }
    std::vector<u32> words(TRANSPOSED_TEST_WORDS);
    for (u32 invocation = 0; invocation < TRANSPOSED_HMMA_TEST_INVOCATIONS; ++invocation) {
        WriteTransposedLane(warps[invocation / LANES], invocation % LANES,
                            std::span{words}.subspan(invocation * TRANSPOSED_HMMA_TEST_LANE_WORDS,
                                                     TRANSPOSED_HMMA_TEST_LANE_WORDS));
    }
    const std::vector<u32> code = Shader::Backend::SPIRV::EmitTransposedCooperativeHmmaTest(
        device.SupportedSpirvVersion(), TRANSPOSED_TEST_SHARED_BYTES, explicit_layout);
    RunTransposedShader(device, code, words);

    u32 wrong_first = 0;
    u32 wrong_chained = 0;
    u32 clobbered = 0;
    for (u32 invocation = 0; invocation < TRANSPOSED_HMMA_TEST_INVOCATIONS; ++invocation) {
        const HalfMatrices& m = warps[invocation / LANES];
        const Tile16x8 d = Multiply(m, m.c, true);
        const Tile16x8 e = Multiply(m, d, true);
        const u32 t = invocation % LANES;
        const size_t g = t >> 2;
        const size_t k = (t & 3) * 2;
        const u32* const out = words.data() + invocation * TRANSPOSED_HMMA_TEST_LANE_WORDS;
        wrong_first += !SameHalves(out[8], d[g][k], d[g][k + 1]) ||
                       !SameHalves(out[9], d[g + 8][k], d[g + 8][k + 1]);
        wrong_chained += !SameHalves(out[10], e[g][k], e[g][k + 1]) ||
                         !SameHalves(out[11], e[g + 8][k], e[g + 8][k + 1]);
        clobbered += out[12] != 1;
    }
    if (wrong_first != 0 || wrong_chained != 0 || clobbered != 0) {
        return fmt::format("{} layout: {} of {} lanes wrong, {} wrong after chaining, {} lanes saw "
                           "the shader's workgroup memory overwritten",
                           explicit_layout ? "explicit" : "plain", wrong_first,
                           TRANSPOSED_HMMA_TEST_INVOCATIONS, wrong_chained, clobbered);
    }
    return {};
}

} // Anonymous namespace

bool TestTransposedCooperativeHmma(const Device& device) {
    try {
        std::string failure = RunTransposedTest(device, false);
        const bool explicit_layout = device.IsKhrWorkgroupMemoryExplicitLayoutSupported() &&
                                     device.IsWorkgroupMemoryExplicitLayout16BitAccessSupported() &&
                                     device.SupportedSpirvVersion() >= 0x00010400;
        if (failure.empty() && explicit_layout) {
            failure = RunTransposedTest(device, true);
        }
        if (!failure.empty()) {
            LOG_WARNING(Render_Vulkan,
                        "Transposed cooperative HMMA self-test failed ({}): guest tensor "
                        "operations stay in shader emulation",
                        failure);
            return false;
        }
        LOG_INFO(Render_Vulkan,
                 "Transposed cooperative HMMA self-test passed (lane layout, rounding after each "
                 "K=8 group, chained HMMA, shader workgroup memory kept; {}): guest tensor "
                 "operations run on M8 N16 K16 cooperative matrices",
                 explicit_layout ? "plain and explicit layouts" : "plain layout");
        return true;
    } catch (const std::exception& exception) {
        LOG_WARNING(Render_Vulkan,
                    "Transposed cooperative HMMA self-test failed ({}): guest tensor operations "
                    "stay in shader emulation",
                    exception.what());
        return false;
    }
}

bool TestCooperativeHmma(const Device& device) {
    try {
        return RunTest(device);
    } catch (const std::exception& exception) {
        LOG_WARNING(Render_Vulkan,
                    "Cooperative matrix self-test failed ({}): guest tensor operations stay in "
                    "shader emulation",
                    exception.what());
        return false;
    }
}

} // namespace Vulkan
