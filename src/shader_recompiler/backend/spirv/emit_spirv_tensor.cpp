// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <optional>
#include <vector>

#include "common/alignment.h"

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "common/common_types.h"
#include "shader_recompiler/frontend/ir/program.h"

namespace Shader::Backend::SPIRV {
namespace {

constexpr u32 GUEST_WARP_LANES = 32;

// A cooperative matrix needs its whole subgroup: every row of the workgroup fills whole warps and
// the pipeline is created with full subgroups. Cooperative matrices also need the Vulkan memory
// model, which has no Coherent decoration, so a program that needs coherent memory keeps the
// emulation.
bool SupportsCooperativeHmma(const IR::Program& program) {
    return program.stage == Stage::Compute && program.workgroup_size[0] != 0 &&
           program.workgroup_size[0] % GUEST_WARP_LANES == 0 &&
           !program.info.requires_coherent_global_memory &&
           !program.info.requires_coherent_image_memory;
}

// Sirit lacks the two cooperative matrix instructions. They are emitted as an OpTypeStruct and an
// OpSelect, which have the same word layout, and PatchCooperativeHmma replaces the opcodes after
// assembly. The operands are constants that no real structure has as members.
CooperativeHmmaTypes DefineCooperativeHmmaTypes(Sirit::Module& module, Id u32_type,
                                                u32 spirv_version) {
    module.AddExtension("SPV_KHR_cooperative_matrix");
    module.AddCapability(spv::Capability::CooperativeMatrixKHR);
    module.AddCapability(spv::Capability::Float16);
    if (spirv_version < 0x00010500) {
        module.AddExtension("SPV_KHR_vulkan_memory_model");
    }
    module.AddCapability(spv::Capability::VulkanMemoryModel);
    module.AddCapability(spv::Capability::VulkanMemoryModelDeviceScope);
    module.SetMemoryModel(spv::AddressingModel::Logical, spv::MemoryModel::Vulkan);
    const Id float16{module.TypeFloat(16)};
    const auto constant{[&](u32 value) { return module.Constant(u32_type, value); }};
    const Id scope{constant(static_cast<u32>(spv::Scope::Subgroup))};
    const auto matrix{[&](u32 columns, spv::CooperativeMatrixUse use) {
        return module.TypeStruct(float16, scope, constant(16), constant(columns),
                                 constant(static_cast<u32>(use)));
    }};
    return CooperativeHmmaTypes{
        .float16 = float16,
        .float16x2 = module.TypeVector(float16, 2),
        .matrix_a = matrix(16, spv::CooperativeMatrixUse::MatrixAKHR),
        .matrix_b = matrix(8, spv::CooperativeMatrixUse::MatrixBKHR),
        .matrix_c = matrix(8, spv::CooperativeMatrixUse::MatrixAccumulatorKHR),
    };
}

// HMMA.16816.F16 on a cooperative matrix. Each lane already holds its share of the matrices in
// registers, so no data moves between lanes or through memory: the lane's halves become the
// matrix's local elements. Vulkan leaves their order to the driver; NVIDIA's is the guest's except
// that A's middle registers swap (see the self-test in vk_cooperative_hmma_test.cpp).
Id EmitCooperativeHmma(Sirit::Module& module, const CooperativeHmmaTypes& types, Id u32_type,
                       Id u32x2_type, Id a, Id b, Id c) {
    const auto insert_word{[&](Id matrix_type, Id matrix, Id vector, u32 word, u32 slot) {
        const Id halves{module.OpBitcast(types.float16x2,
                                         module.OpCompositeExtract(u32_type, vector, word))};
        for (u32 half = 0; half < 2; ++half) {
            matrix = module.OpCompositeInsert(
                matrix_type, module.OpCompositeExtract(types.float16, halves, half), matrix,
                slot * 2 + half);
        }
        return matrix;
    }};
    constexpr std::array<u32, 4> a_words{0, 2, 1, 3};
    Id matrix_a{module.OpUndef(types.matrix_a)};
    for (u32 slot = 0; slot < a_words.size(); ++slot) {
        matrix_a = insert_word(types.matrix_a, matrix_a, a, a_words[slot], slot);
    }
    Id matrix_b{module.OpUndef(types.matrix_b)};
    Id matrix_c{module.OpUndef(types.matrix_c)};
    for (u32 slot = 0; slot < 2; ++slot) {
        matrix_b = insert_word(types.matrix_b, matrix_b, b, slot, slot);
        matrix_c = insert_word(types.matrix_c, matrix_c, c, slot, slot);
    }
    // OpCooperativeMatrixMulAddKHR after PatchCooperativeHmma.
    const Id matrix_d{module.OpSelect(types.matrix_c, matrix_a, matrix_b, matrix_c)};
    const auto extract_word{[&](u32 word) {
        return module.OpBitcast(
            u32_type,
            module.OpCompositeConstruct(
                types.float16x2, module.OpCompositeExtract(types.float16, matrix_d, word * 2),
                module.OpCompositeExtract(types.float16, matrix_d, word * 2 + 1)));
    }};
    return module.OpCompositeConstruct(u32x2_type, extract_word(0), extract_word(1));
}

// The transposed path, for devices whose only f16 shape is M8 N16 K16 (Intel's): the guest's
// D(16x8) = A(16x16) * B(16x8) + C(16x8) is computed as D^T = B^T * A^T + C^T. Every lane stores
// its register fragments into its warp's tile in workgroup memory, column-major loads of the
// row-major tiles give the transposes, and the result is stored back and read by the lanes. The
// guest rounds to f16 after each K=8 group, so the product is two K=16 products over B tiles with
// the other half zeroed, with an f16 accumulator (one K=16 product rounds once; a round trip
// through f32 is optimised away by Intel's driver).
constexpr u32 TRANSPOSED_TILE_HALVES = 640; // A 16x16, B low and high 16x8, C/D 16x8: 1,280 bytes
constexpr u32 TRANSPOSED_TILE_A = 0;
constexpr u32 TRANSPOSED_TILE_B_LOW = 256;
constexpr u32 TRANSPOSED_TILE_B_HIGH = 384;
constexpr u32 TRANSPOSED_TILE_C = 512;

u32 TransposedStagingBase(const IR::Program& program) {
    // Cooperative matrix loads and stores need 16-byte aligned pointers and strides.
    return Common::AlignUp(program.shared_memory_size, 16U);
}

u32 TransposedWarps(const IR::Program& program) {
    return program.workgroup_size[0] * program.workgroup_size[1] * program.workgroup_size[2] /
           GUEST_WARP_LANES;
}

// Whether a program's tiles fit beside its own workgroup memory. With explicit layouts every
// Block-decorated workgroup variable aliases the others from byte 0, so the tiles start past the
// program's bytes; otherwise they are a separate variable, and the same budget is kept.
bool SupportsTransposedHmma(const EmitContext& ctx, const IR::Program& program) {
    if (!SupportsCooperativeHmma(program) || !ctx.SupportsSubgroups()) {
        return false;
    }
    const bool explicit_layout = ctx.uses_explicit_workgroup_layout && program.shared_memory_size != 0;
    if (explicit_layout && !ctx.profile.support_workgroup_layout_16bit_access) {
        return false;
    }
    const u64 bytes = u64{TransposedStagingBase(program)} +
                      u64{TransposedWarps(program)} * TRANSPOSED_TILE_HALVES * 2;
    return bytes <= ctx.profile.max_compute_shared_memory_size;
}

CooperativeHmmaTypes DefineTransposedHmmaTypes(Sirit::Module& module, Id u32_type,
                                               u32 spirv_version) {
    module.AddExtension("SPV_KHR_cooperative_matrix");
    module.AddCapability(spv::Capability::CooperativeMatrixKHR);
    module.AddCapability(spv::Capability::Float16);
    module.AddCapability(spv::Capability::GroupNonUniform);
    if (spirv_version < 0x00010500) {
        module.AddExtension("SPV_KHR_vulkan_memory_model");
    }
    module.AddCapability(spv::Capability::VulkanMemoryModel);
    module.AddCapability(spv::Capability::VulkanMemoryModelDeviceScope);
    module.SetMemoryModel(spv::AddressingModel::Logical, spv::MemoryModel::Vulkan);
    const Id float16{module.TypeFloat(16)};
    const auto constant{[&](u32 value) { return module.Constant(u32_type, value); }};
    const Id scope{constant(static_cast<u32>(spv::Scope::Subgroup))};
    const auto matrix{[&](u32 rows, u32 columns, spv::CooperativeMatrixUse use) {
        return module.TypeStruct(float16, scope, constant(rows), constant(columns),
                                 constant(static_cast<u32>(use)));
    }};
    return CooperativeHmmaTypes{
        .float16 = float16,
        .float16x2 = module.TypeVector(float16, 2),
        .transposed_a = matrix(8, 16, spv::CooperativeMatrixUse::MatrixAKHR),
        .transposed_b = matrix(16, 16, spv::CooperativeMatrixUse::MatrixBKHR),
        .transposed_c = matrix(8, 16, spv::CooperativeMatrixUse::MatrixAccumulatorKHR),
        .column_major =
            constant(static_cast<u32>(spv::CooperativeMatrixLayout::ColumnMajorKHR)),
        .workgroup_scope = constant(static_cast<u32>(spv::Scope::Workgroup)),
    };
}

struct TransposedStaging {
    Id variable;        ///< The warps' tiles, TRANSPOSED_TILE_HALVES f16 per warp
    Id element_pointer; ///< Pointer to one f16 in workgroup memory
    bool in_block;      ///< Member 0 of a Block struct (explicit layout)
};

TransposedStaging DefineTransposedStaging(Sirit::Module& module, const CooperativeHmmaTypes& types,
                                          Id u32_type, u32 warps, u32 base_bytes,
                                          bool explicit_layout) {
    const Id length{module.Constant(u32_type, warps * TRANSPOSED_TILE_HALVES)};
    const Id element_pointer{module.TypePointer(spv::StorageClass::Workgroup, types.float16)};
    if (!explicit_layout) {
        const Id array{module.TypeArray(types.float16, length)};
        const Id variable{module.AddGlobalVariable(
            module.TypePointer(spv::StorageClass::Workgroup, array), spv::StorageClass::Workgroup)};
        return {variable, element_pointer, false};
    }
    module.AddCapability(spv::Capability::WorkgroupMemoryExplicitLayout16BitAccessKHR);
    const Id array{module.TypeArray(types.float16, length)};
    module.Decorate(array, spv::Decoration::ArrayStride, 2U);
    const Id block{module.TypeStruct(array)};
    module.MemberDecorate(block, 0U, spv::Decoration::Offset, base_bytes);
    module.Decorate(block, spv::Decoration::Block);
    const Id variable{module.AddGlobalVariable(
        module.TypePointer(spv::StorageClass::Workgroup, block), spv::StorageClass::Workgroup)};
    module.Decorate(variable, spv::Decoration::Aliased);
    return {variable, element_pointer, true};
}

struct TransposedInputs {
    Id u32_type;
    Id u32x2_type;
    Id u32x4_type;
    Id lane;     ///< SubgroupLocalInvocationId input
    Id subgroup; ///< SubgroupId input
    TransposedStaging staging;
};

// The helper every transposed HMMA calls: (a0..a3, b0 b1, c0 c1) -> (d0 d1) in the mma.m16n8k16
// register layout (lane t: g = t / 4 and k = 2 * (t % 4)).
Id DefineTransposedHmmaFunction(Sirit::Module& module, CooperativeHmmaTypes& types,
                                const TransposedInputs& in) {
    const Id u32_type{in.u32_type};
    const auto constant{[&](u32 value) { return module.Constant(u32_type, value); }};
    const Id function_type{
        module.TypeFunction(in.u32x2_type, in.u32x4_type, in.u32x2_type, in.u32x2_type)};
    const Id function{
        module.OpFunction(in.u32x2_type, spv::FunctionControlMask::DontInline, function_type)};
    const Id a{module.OpFunctionParameter(in.u32x4_type)};
    const Id b{module.OpFunctionParameter(in.u32x2_type)};
    const Id c{module.OpFunctionParameter(in.u32x2_type)};
    module.AddLabel();

    const Id lane{module.OpLoad(u32_type, in.lane)};
    const Id warp{module.OpLoad(u32_type, in.subgroup)};
    const Id g{module.OpShiftRightLogical(u32_type, lane, constant(2))};
    const Id k{module.OpShiftLeftLogical(u32_type, module.OpBitwiseAnd(u32_type, lane, constant(3)),
                                         constant(1))};
    const Id tile{module.OpIMul(u32_type, warp, constant(TRANSPOSED_TILE_HALVES))};
    const auto barrier{[&] {
        const u32 semantics{static_cast<u32>(spv::MemorySemanticsMask::AcquireRelease) |
                            static_cast<u32>(spv::MemorySemanticsMask::WorkgroupMemory)};
        module.OpControlBarrier(constant(static_cast<u32>(spv::Scope::Subgroup)),
                                constant(static_cast<u32>(spv::Scope::Subgroup)),
                                constant(semantics));
    }};
    const auto pointer{[&](Id index) {
        return in.staging.in_block
                   ? module.OpAccessChain(in.staging.element_pointer, in.staging.variable,
                                          constant(0), index)
                   : module.OpAccessChain(in.staging.element_pointer, in.staging.variable, index);
    }};
    // Element (row + row_add, column + column_add) of a row-major tile region.
    const auto element{[&](u32 region, Id row, u32 row_add, u32 stride, Id column, u32 column_add) {
        const Id row_offset{
            module.OpIMul(u32_type, module.OpIAdd(u32_type, row, constant(row_add)), constant(stride))};
        const Id offset{module.OpIAdd(u32_type, row_offset,
                                      module.OpIAdd(u32_type, column, constant(column_add)))};
        return module.OpIAdd(u32_type, module.OpIAdd(u32_type, tile, constant(region)), offset);
    }};
    // The lanes share the tiles: non-private accesses, made available and visible at workgroup
    // scope (the memory model leaves private ones unordered between invocations).
    const auto store{[&](Id index, Id value) {
        module.OpStore(pointer(index), value,
                       spv::MemoryAccessMask::NonPrivatePointer |
                           spv::MemoryAccessMask::MakePointerAvailable,
                       types.workgroup_scope.value);
    }};
    const auto load{[&](Id index) {
        return module.OpLoad(types.float16, pointer(index),
                             spv::MemoryAccessMask::NonPrivatePointer |
                                 spv::MemoryAccessMask::MakePointerVisible,
                             types.workgroup_scope.value);
    }};
    const auto halves{[&](Id vector, u32 word) {
        const Id pair{
            module.OpBitcast(types.float16x2, module.OpCompositeExtract(u32_type, vector, word))};
        return std::array{module.OpCompositeExtract(types.float16, pair, 0U),
                          module.OpCompositeExtract(types.float16, pair, 1U)};
    }};

    // The previous HMMA's lanes may still read the C tile.
    barrier();
    for (u32 word = 0; word < 4; ++word) {
        // a0: row g, k; a1: row g + 8, k; a2: row g, k + 8; a3: row g + 8, k + 8.
        const u32 row_add{(word & 1) != 0 ? 8U : 0U};
        const u32 column_add{(word & 2) != 0 ? 8U : 0U};
        const auto values{halves(a, word)};
        store(element(TRANSPOSED_TILE_A, g, row_add, 16, k, column_add), values[0]);
        store(element(TRANSPOSED_TILE_A, g, row_add, 16, k, column_add + 1), values[1]);
    }
    // b0: column g, rows k and k + 1 (the low K half); b1: rows k + 8 and k + 9 (the high half).
    // Each half has its own tile with the other half zeroed.
    const Id zero{module.ConstantNull(types.float16)};
    const auto b_low{halves(b, 0)};
    const auto b_high{halves(b, 1)};
    for (u32 row = 0; row < 2; ++row) {
        store(element(TRANSPOSED_TILE_B_LOW, k, row, 8, g, 0), b_low[row]);
        store(element(TRANSPOSED_TILE_B_LOW, k, row + 8, 8, g, 0), zero);
        store(element(TRANSPOSED_TILE_B_HIGH, k, row, 8, g, 0), zero);
        store(element(TRANSPOSED_TILE_B_HIGH, k, row + 8, 8, g, 0), b_high[row]);
    }
    // c0: row g, k and k + 1; c1: row g + 8.
    for (u32 word = 0; word < 2; ++word) {
        const auto values{halves(c, word)};
        store(element(TRANSPOSED_TILE_C, g, word * 8, 8, k, 0), values[0]);
        store(element(TRANSPOSED_TILE_C, g, word * 8, 8, k, 1), values[1]);
    }
    barrier();

    const auto tile_pointer{[&](u32 region) {
        return pointer(module.OpIAdd(u32_type, tile, constant(region)));
    }};
    // OpCooperativeMatrixLoadKHR after PatchCooperativeHmma: (pointer, layout, stride).
    const auto matrix_load{[&](Id type, u32 region, u32 stride) {
        return module.OpBitFieldUExtract(type, tile_pointer(region), types.column_major,
                                         constant(stride));
    }};
    const Id b_low_t{matrix_load(types.transposed_a, TRANSPOSED_TILE_B_LOW, 8)};
    const Id b_high_t{matrix_load(types.transposed_a, TRANSPOSED_TILE_B_HIGH, 8)};
    const Id a_t{matrix_load(types.transposed_b, TRANSPOSED_TILE_A, 16)};
    Id accumulator{matrix_load(types.transposed_c, TRANSPOSED_TILE_C, 8)};
    // OpCooperativeMatrixMulAddKHR after PatchCooperativeHmma; rounds to f16 after each half.
    accumulator = module.OpSelect(types.transposed_c, b_low_t, a_t, accumulator);
    accumulator = module.OpSelect(types.transposed_c, b_high_t, a_t, accumulator);
    barrier();
    const Id result_pointer{tile_pointer(TRANSPOSED_TILE_C)};
    module.OpStore(result_pointer, accumulator); // OpCooperativeMatrixStoreKHR
    types.transposed_stores.push_back({result_pointer.value, accumulator.value, constant(8).value});
    barrier();

    std::array<Id, 2> d;
    for (u32 word = 0; word < 2; ++word) {
        const Id low{load(element(TRANSPOSED_TILE_C, g, word * 8, 8, k, 0))};
        const Id high{load(element(TRANSPOSED_TILE_C, g, word * 8, 8, k, 1))};
        d[word] = module.OpBitcast(u32_type, module.OpCompositeConstruct(types.float16x2, low, high));
    }
    module.OpReturnValue(module.OpCompositeConstruct(in.u32x2_type, d[0], d[1]));
    module.OpFunctionEnd();
    return function;
}

Id DefineBuiltinInput(Sirit::Module& module, Id u32_type, spv::BuiltIn builtin) {
    const Id variable{module.AddGlobalVariable(
        module.TypePointer(spv::StorageClass::Input, u32_type), spv::StorageClass::Input)};
    module.Decorate(variable, spv::Decoration::BuiltIn, builtin);
    return variable;
}

void DefineTransposedHmma(EmitContext& ctx, const IR::Program& program) {
    ctx.cooperative_hmma =
        DefineTransposedHmmaTypes(ctx, ctx.U32[1], ctx.profile.supported_spirv);
    const bool explicit_layout =
        ctx.uses_explicit_workgroup_layout && program.shared_memory_size != 0;
    const TransposedStaging staging{DefineTransposedStaging(
        ctx, *ctx.cooperative_hmma, ctx.U32[1], TransposedWarps(program),
        TransposedStagingBase(program), explicit_layout)};
    ctx.interfaces.push_back(staging.variable);
    Id lane{ctx.subgroup_local_invocation_id};
    if (!Sirit::ValidId(lane)) {
        lane = DefineBuiltinInput(ctx, ctx.U32[1], spv::BuiltIn::SubgroupLocalInvocationId);
        ctx.interfaces.push_back(lane);
    }
    const Id subgroup{DefineBuiltinInput(ctx, ctx.U32[1], spv::BuiltIn::SubgroupId)};
    ctx.interfaces.push_back(subgroup);
    ctx.transposed_hmma_function = DefineTransposedHmmaFunction(
        ctx, *ctx.cooperative_hmma,
        TransposedInputs{ctx.U32[1], ctx.U32[2], ctx.U32[4], lane, subgroup, staging});
}

} // Anonymous namespace

// One path per program for HMMA.16816 (K=16): the direct cooperative path where the device has
// the guest's 16x8x16 shape, else the transposed one (opt-in), else the software helper. A program
// can still mix: its K=8 HMMAs always take the software helper, and a guarded (predicated) HMMA was
// already expanded by the lowering into shuffles and FFMAs, so neither reaches the cooperative
// paths. Both need 32-lane warps, which every pipeline with a cooperative matrix gets.
void DefineSoftwareMatrixFunctions(EmitContext& ctx, const IR::Program& program) {
    const bool cooperative{ctx.profile.support_cooperative_hmma && SupportsCooperativeHmma(program)};
    const bool transposed{!cooperative && ctx.profile.support_transposed_cooperative_hmma &&
                          SupportsTransposedHmma(ctx, program)};
    std::array<bool, 2> needed{};
    for (const auto* block : program.blocks) {
        for (const auto& inst : block->Instructions()) {
            if (inst.GetOpcode() != IR::Opcode::TensorMulAdd || inst.Flags<u32>() == 0) continue;
            const u32 depth = inst.Flags<u32>();
            if (depth != 8 && depth != 16) throw LogicError("Invalid software HMMA depth");
            if (depth == 16 && (cooperative || transposed)) {
                if (!ctx.cooperative_hmma) {
                    if (transposed) {
                        DefineTransposedHmma(ctx, program);
                    } else {
                        ctx.cooperative_hmma = DefineCooperativeHmmaTypes(
                            ctx, ctx.U32[1], ctx.profile.supported_spirv);
                    }
                }
                continue;
            }
            needed[depth == 16] = true;
        }
    }
    for (u32 variant = 0; variant < 2; ++variant) {
        if (!needed[variant]) continue;
        ctx.AddCapability(spv::Capability::GroupNonUniform);
        ctx.AddCapability(spv::Capability::GroupNonUniformShuffle);
        if (!ctx.SupportsSubgroups() ||
            ctx.profile.has_broken_spirv_subgroup_shuffle)
            throw NotImplementedException("Software HMMA requires subgroup shuffle support");
        const Id u32ty = ctx.U32[1], f32ty = ctx.F32[1];
        const auto imm = [&](u32 v) { return ctx.Const(v); };
        const auto add = [&](Id a, Id b) { return ctx.OpIAdd(u32ty, a, b); };
        const auto and_ = [&](Id a, u32 b) { return ctx.OpBitwiseAnd(u32ty, a, imm(b)); };
        const auto unpack = [&](Id v) { return ctx.OpUnpackHalf2x16(ctx.F32[2], v); };
        const auto round_half = [&](Id value) {
            // Do not express the intermediate rounding as unpack(pack(value)): drivers may
            // eliminate that round trip. Round normal significands explicitly, ties to even;
            // half subnormals lie on the exactly representable 2^-24 grid.
            const Id bits = ctx.OpBitcast(u32ty, value);
            const Id magnitude = and_(bits, 0x7fffffffU);
            const Id tie = and_(ctx.OpShiftRightLogical(u32ty, magnitude, imm(13)), 1);
            const Id rounded = and_(add(add(magnitude, imm(0xfff)), tie), ~0x1fffU);
            const Id normal = ctx.OpSelect(u32ty,
                ctx.OpUGreaterThanEqual(ctx.U1, rounded, imm(0x47800000)), imm(0x7f800000), rounded);
            const Id scaled = ctx.OpFMul(f32ty, ctx.OpBitcast(f32ty, magnitude), ctx.Const(16777216.0f));
            const Id subnormal = ctx.OpBitcast(u32ty,
                ctx.OpFMul(f32ty, ctx.OpRoundEven(f32ty, scaled), ctx.Const(0x1p-24f)));
            const Id finite = ctx.OpSelect(u32ty,
                ctx.OpUGreaterThanEqual(ctx.U1, magnitude, imm(0x38800000)), normal, subnormal);
            const Id result = ctx.OpSelect(u32ty,
                ctx.OpUGreaterThanEqual(ctx.U1, magnitude, imm(0x7f800000)), magnitude, finite);
            return ctx.OpBitcast(f32ty, ctx.OpBitwiseOr(u32ty, result, and_(bits, 0x80000000)));
        };
        const Id type = ctx.TypeFunction(ctx.U32[2], ctx.U32[4], ctx.U32[2], ctx.U32[2]);
        const Id fn = ctx.OpFunction(ctx.U32[2], spv::FunctionControlMask::DontInline, type);
        ctx.software_matrix_functions[variant] = fn;
        const Id matrix_a = ctx.OpFunctionParameter(ctx.U32[4]);
        const Id matrix_b = ctx.OpFunctionParameter(ctx.U32[2]);
        const Id matrix_c = ctx.OpFunctionParameter(ctx.U32[2]);
        ctx.AddLabel();
        const Id counter = ctx.AddLocalVariable(ctx.TypePointer(spv::StorageClass::Function, u32ty),
                                           spv::StorageClass::Function);
        std::array<Id, 4> accumulators;
        for (auto& acc : accumulators)
            acc = ctx.AddLocalVariable(ctx.TypePointer(spv::StorageClass::Function, f32ty),
                                 spv::StorageClass::Function);
        std::array<Id, 8> inputs;
        for (u32 i = 0; i < 4; ++i)
            inputs[i] = ctx.OpCompositeExtract(u32ty, matrix_a, i);
        for (u32 i = 0; i < 2; ++i) {
            inputs[4 + i] = ctx.OpCompositeExtract(u32ty, matrix_b, i);
            inputs[6 + i] = ctx.OpCompositeExtract(u32ty, matrix_c, i);
        }
        for (u32 row = 0; row < 2; ++row) {
            const Id c = unpack(inputs[6 + row]);
            for (u32 col = 0; col < 2; ++col)
                ctx.OpStore(accumulators[row * 2 + col], ctx.OpCompositeExtract(f32ty, c, col));
        }
        ctx.OpStore(counter, imm(0));
        const Id lane = ctx.OpLoad(u32ty, ctx.subgroup_local_invocation_id);
        const Id warp_base = and_(lane, ~31U);
        const Id row_base = and_(lane, 28);
        const Id col_base = ctx.OpShiftLeftLogical(u32ty, and_(lane, 3), imm(3));
        const Id a_base = add(warp_base, row_base);
        const Id b_base = add(warp_base, col_base);
        // Unrolling a K=8 group lets its shuffles overlap, which latency-bound small dispatches
        // need; unrolling all of K=16 raises register pressure and slows large dispatches.
        const auto emit_group = [&](auto pick) {
            for (u32 component = 0; component < 4; ++component) {
                const Id b = pick(inputs[4], inputs[5]);
                std::array<Id, 2> bs;
                for (u32 col = 0; col < 2; ++col)
                    bs[col] = unpack(ctx.OpGroupNonUniformShuffle(
                        u32ty, imm(3), b, add(b_base, imm(component + col * 4))));
                for (u32 row = 0; row < 2; ++row) {
                    const Id as = unpack(ctx.OpGroupNonUniformShuffle(
                        u32ty, imm(3), pick(inputs[row], inputs[row + 2]),
                        add(a_base, imm(component))));
                    for (u32 col = 0; col < 2; ++col) {
                        Id value = ctx.OpLoad(f32ty, accumulators[row * 2 + col]);
                        for (u32 half = 0; half < 2; ++half) {
                            value = ctx.OpFma(f32ty, ctx.OpCompositeExtract(f32ty, as, half),
                                              ctx.OpCompositeExtract(f32ty, bs[col], half), value);
                            ctx.Decorate(value, spv::Decoration::NoContraction);
                        }
                        ctx.OpStore(accumulators[row * 2 + col], value);
                    }
                }
            }
            // HMMA.F16 rounds after each K=8 group, including the midpoint of K=16.
            for (const Id accumulator : accumulators)
                ctx.OpStore(accumulator, round_half(ctx.OpLoad(f32ty, accumulator)));
        };
        if (variant == 0) {
            emit_group([](Id low, Id) { return low; });
        } else {
            const Id header = ctx.OpLabel(), body = ctx.OpLabel(), next = ctx.OpLabel(),
                     end = ctx.OpLabel();
            ctx.OpBranch(header);
            ctx.AddLabel(header);
            const Id group = ctx.OpLoad(u32ty, counter);
            const Id more = ctx.OpULessThan(ctx.U1, group, imm(2));
            ctx.OpLoopMerge(end, next, spv::LoopControlMask::DontUnroll);
            ctx.OpBranchConditional(more, body, end);
            ctx.AddLabel(body);
            const Id upper = ctx.OpIEqual(ctx.U1, group, imm(1));
            emit_group([&](Id low, Id high) { return ctx.OpSelect(u32ty, upper, high, low); });
            ctx.OpBranch(next);
            ctx.AddLabel(next);
            ctx.OpStore(counter, add(group, imm(1)));
            ctx.OpBranch(header);
            ctx.AddLabel(end);
        }
        std::array<Id, 2> packed;
        for (u32 row = 0; row < 2; ++row)
            packed[row] = ctx.OpPackHalf2x16(u32ty, ctx.OpCompositeConstruct(ctx.F32[2],
                ctx.OpLoad(f32ty, accumulators[row * 2]), ctx.OpLoad(f32ty, accumulators[row * 2 + 1])));
        ctx.OpReturnValue(ctx.OpCompositeConstruct(ctx.U32[2], packed[0], packed[1]));
        ctx.OpFunctionEnd();
    }
}

Id EmitTensorMulAdd(EmitContext& ctx, IR::Inst* inst, Id a, Id b, Id c) {
    const u32 depth = inst->Flags<u32>();
    if (depth != 8 && depth != 16) {
        throw NotImplementedException("Unsupported matrix multiply depth");
    }
    if (depth == 16 && ctx.cooperative_hmma) {
        if (Sirit::ValidId(ctx.transposed_hmma_function)) {
            return ctx.OpFunctionCall(ctx.U32[2], ctx.transposed_hmma_function, a, b, c);
        }
        return EmitCooperativeHmma(ctx, *ctx.cooperative_hmma, ctx.U32[1], ctx.U32[2], a, b, c);
    }
    return ctx.OpFunctionCall(ctx.U32[2], ctx.software_matrix_functions[depth == 16], a, b, c);
}

void PatchCooperativeHmma(std::vector<u32>& code, const CooperativeHmmaTypes& types) {
    const auto is{[](u32 word, Id id) { return Sirit::ValidId(id) && word == id.value; }};
    const auto is_matrix{[&](u32 word) {
        return is(word, types.matrix_a) || is(word, types.matrix_b) || is(word, types.matrix_c) ||
               is(word, types.transposed_a) || is(word, types.transposed_b) ||
               is(word, types.transposed_c);
    }};
    constexpr u32 visible{static_cast<u32>(spv::MemoryAccessMask::NonPrivatePointer) |
                          static_cast<u32>(spv::MemoryAccessMask::MakePointerVisible)};
    constexpr u32 available{static_cast<u32>(spv::MemoryAccessMask::NonPrivatePointer) |
                            static_cast<u32>(spv::MemoryAccessMask::MakePointerAvailable)};
    const bool rewrites_memory{!types.transposed_stores.empty()};
    std::vector<u32> patched;
    if (rewrites_memory) {
        patched.reserve(code.size() + types.transposed_stores.size() * 4 + 64);
        patched.insert(patched.end(), code.begin(), code.begin() + 5);
    }
    for (size_t offset = 5; offset < code.size();) {
        const u32 word_count{code[offset] >> 16};
        if (word_count == 0) {
            break;
        }
        const auto opcode{static_cast<spv::Op>(code[offset] & 0xFFFFu)};
        std::optional<spv::Op> replacement;
        if (opcode == spv::Op::OpTypeStruct) {
            if (is_matrix(code[offset + 1])) {
                replacement = spv::Op::OpTypeCooperativeMatrixKHR;
            }
        } else if (opcode == spv::Op::OpSelect &&
                   (is(code[offset + 1], types.matrix_c) ||
                    is(code[offset + 1], types.transposed_c))) {
            replacement = spv::Op::OpCooperativeMatrixMulAddKHR;
        } else if (rewrites_memory && opcode == spv::Op::OpBitFieldUExtract &&
                   is_matrix(code[offset + 1])) {
            // (type, result, pointer, layout, stride) plus the memory operands.
            patched.push_back((8U << 16) | static_cast<u32>(spv::Op::OpCooperativeMatrixLoadKHR));
            patched.insert(patched.end(), code.begin() + offset + 1, code.begin() + offset + 6);
            patched.push_back(visible);
            patched.push_back(types.workgroup_scope.value);
            offset += word_count;
            continue;
        } else if (rewrites_memory && opcode == spv::Op::OpStore && word_count == 3) {
            const auto store{std::ranges::find_if(types.transposed_stores, [&](const auto& marker) {
                return marker.pointer == code[offset + 1] && marker.object == code[offset + 2];
            })};
            if (store != types.transposed_stores.end()) {
                patched.insert(patched.end(),
                               {(7U << 16) | static_cast<u32>(spv::Op::OpCooperativeMatrixStoreKHR),
                                store->pointer, store->object, types.column_major.value,
                                store->stride, available, types.workgroup_scope.value});
                offset += word_count;
                continue;
            }
        }
        if (replacement) {
            code[offset] = (code[offset] & 0xFFFF0000u) | static_cast<u32>(*replacement);
        }
        if (rewrites_memory) {
            patched.insert(patched.end(), code.begin() + offset, code.begin() + offset + word_count);
        }
        offset += word_count;
    }
    if (rewrites_memory) {
        code = std::move(patched);
    }
}

std::vector<u32> EmitCooperativeHmmaTest(u32 spirv_version) {
    Sirit::Module module{spirv_version};
    module.AddCapability(spv::Capability::Shader);
    module.AddCapability(spv::Capability::GroupNonUniform);

    const Id void_type{module.TypeVoid()};
    const Id u32_type{module.TypeUInt(32)};
    const Id u32x2_type{module.TypeVector(u32_type, 2)};
    const Id u32x4_type{module.TypeVector(u32_type, 4)};
    const CooperativeHmmaTypes types{DefineCooperativeHmmaTypes(module, u32_type, spirv_version)};

    const Id words_type{module.TypeRuntimeArray(u32_type)};
    module.Decorate(words_type, spv::Decoration::ArrayStride, 4U);
    const Id block_type{module.TypeStruct(words_type)};
    module.Decorate(block_type, spv::Decoration::Block);
    module.MemberDecorate(block_type, 0U, spv::Decoration::Offset, 0U);
    const Id buffer{module.AddGlobalVariable(
        module.TypePointer(spv::StorageClass::StorageBuffer, block_type),
        spv::StorageClass::StorageBuffer)};
    module.Decorate(buffer, spv::Decoration::DescriptorSet, 0U);
    module.Decorate(buffer, spv::Decoration::Binding, 0U);
    const Id lane_variable{module.AddGlobalVariable(
        module.TypePointer(spv::StorageClass::Input, u32_type), spv::StorageClass::Input)};
    module.Decorate(lane_variable, spv::Decoration::BuiltIn,
                    spv::BuiltIn::SubgroupLocalInvocationId);
    const Id word_pointer{module.TypePointer(spv::StorageClass::StorageBuffer, u32_type)};
    const Id zero{module.Constant(u32_type, 0U)};

    const Id main{module.OpFunction(void_type, spv::FunctionControlMask::MaskNone,
                                    module.TypeFunction(void_type))};
    module.AddLabel();
    const Id base{module.OpIMul(u32_type, module.OpLoad(u32_type, lane_variable),
                                module.Constant(u32_type, COOPERATIVE_HMMA_TEST_LANE_WORDS))};
    const auto word_at{[&](u32 index) {
        return module.OpAccessChain(word_pointer, buffer, zero,
                                    module.OpIAdd(u32_type, base,
                                                  module.Constant(u32_type, index)));
    }};
    const auto load{[&](u32 index) { return module.OpLoad(u32_type, word_at(index)); }};
    const Id a{module.OpCompositeConstruct(u32x4_type, load(0), load(1), load(2), load(3))};
    const Id b{module.OpCompositeConstruct(u32x2_type, load(4), load(5))};
    const Id c{module.OpCompositeConstruct(u32x2_type, load(6), load(7))};
    const Id d{EmitCooperativeHmma(module, types, u32_type, u32x2_type, a, b, c)};
    module.OpStore(word_at(8), module.OpCompositeExtract(u32_type, d, 0U));
    module.OpStore(word_at(9), module.OpCompositeExtract(u32_type, d, 1U));
    module.OpReturn();
    module.OpFunctionEnd();

    // Since SPIR-V 1.4 the interface lists every global the entry point uses.
    std::vector<Id> interfaces{lane_variable};
    if (spirv_version >= 0x00010400U) {
        interfaces.push_back(buffer);
    }
    module.AddEntryPoint(spv::ExecutionModel::GLCompute, main, "main", interfaces);
    module.AddExecutionMode(main, spv::ExecutionMode::LocalSize, GUEST_WARP_LANES, 1U, 1U);

    std::vector<u32> code{module.Assemble()};
    PatchCooperativeHmma(code, types);
    return code;
}

std::vector<u32> EmitTransposedCooperativeHmmaTest(u32 spirv_version, u32 shared_bytes,
                                                   bool explicit_layout) {
    Sirit::Module module{spirv_version};
    module.AddCapability(spv::Capability::Shader);
    module.AddCapability(spv::Capability::GroupNonUniform);

    const Id void_type{module.TypeVoid()};
    const Id bool_type{module.TypeBool()};
    const Id u32_type{module.TypeUInt(32)};
    const Id u32x2_type{module.TypeVector(u32_type, 2)};
    const Id u32x4_type{module.TypeVector(u32_type, 4)};
    const auto constant{[&](u32 value) { return module.Constant(u32_type, value); }};
    CooperativeHmmaTypes types{DefineTransposedHmmaTypes(module, u32_type, spirv_version)};

    const Id words_type{module.TypeRuntimeArray(u32_type)};
    module.Decorate(words_type, spv::Decoration::ArrayStride, 4U);
    const Id block_type{module.TypeStruct(words_type)};
    module.Decorate(block_type, spv::Decoration::Block);
    module.MemberDecorate(block_type, 0U, spv::Decoration::Offset, 0U);
    const Id buffer{module.AddGlobalVariable(
        module.TypePointer(spv::StorageClass::StorageBuffer, block_type),
        spv::StorageClass::StorageBuffer)};
    module.Decorate(buffer, spv::Decoration::DescriptorSet, 0U);
    module.Decorate(buffer, spv::Decoration::Binding, 0U);
    const Id word_pointer{module.TypePointer(spv::StorageClass::StorageBuffer, u32_type)};

    const Id lane{DefineBuiltinInput(module, u32_type, spv::BuiltIn::SubgroupLocalInvocationId)};
    const Id subgroup{DefineBuiltinInput(module, u32_type, spv::BuiltIn::SubgroupId)};
    const Id invocation{DefineBuiltinInput(module, u32_type, spv::BuiltIn::LocalInvocationIndex)};
    std::vector<Id> interfaces{lane, subgroup, invocation};

    // The shader's own workgroup memory, declared the way EmitContext::DefineSharedMemory does.
    const u32 shared_words{shared_bytes / 4};
    const bool block_layout{explicit_layout && shared_words != 0};
    if (block_layout && spirv_version < 0x00010400) {
        throw LogicError("SPV_KHR_workgroup_memory_explicit_layout needs SPIR-V 1.4");
    }
    Id shared{};
    const Id shared_word_pointer{module.TypePointer(spv::StorageClass::Workgroup, u32_type)};
    if (shared_words != 0) {
        const Id array{module.TypeArray(u32_type, constant(shared_words))};
        if (block_layout) {
            module.AddExtension("SPV_KHR_workgroup_memory_explicit_layout");
            module.AddCapability(spv::Capability::WorkgroupMemoryExplicitLayoutKHR);
            module.Decorate(array, spv::Decoration::ArrayStride, 4U);
            const Id block{module.TypeStruct(array)};
            module.MemberDecorate(block, 0U, spv::Decoration::Offset, 0U);
            module.Decorate(block, spv::Decoration::Block);
            shared = module.AddGlobalVariable(
                module.TypePointer(spv::StorageClass::Workgroup, block), spv::StorageClass::Workgroup);
            module.Decorate(shared, spv::Decoration::Aliased);
        } else {
            shared = module.AddGlobalVariable(
                module.TypePointer(spv::StorageClass::Workgroup, array), spv::StorageClass::Workgroup);
        }
    }
    const TransposedStaging staging{DefineTransposedStaging(
        module, types, u32_type, TRANSPOSED_HMMA_TEST_INVOCATIONS / GUEST_WARP_LANES,
        Common::AlignUp(shared_bytes, 16U), block_layout)};
    const Id function{DefineTransposedHmmaFunction(
        module, types,
        TransposedInputs{u32_type, u32x2_type, u32x4_type, lane, subgroup, staging})};

    const Id main{module.OpFunction(void_type, spv::FunctionControlMask::MaskNone,
                                    module.TypeFunction(void_type))};
    module.AddLabel();
    const Id index{module.OpLoad(u32_type, invocation)};
    const Id base{module.OpIMul(u32_type, index, constant(TRANSPOSED_HMMA_TEST_LANE_WORDS))};
    const auto word_at{[&](u32 word) {
        return module.OpAccessChain(word_pointer, buffer, constant(0),
                                    module.OpIAdd(u32_type, base, constant(word)));
    }};
    const auto load{[&](u32 word) { return module.OpLoad(u32_type, word_at(word)); }};
    const auto shared_at{[&](Id word) {
        return block_layout ? module.OpAccessChain(shared_word_pointer, shared, constant(0), word)
                            : module.OpAccessChain(shared_word_pointer, shared, word);
    }};
    // Each invocation fills words index, index + 128, ... of the shader's memory.
    const u32 pattern_words{std::min(shared_words / TRANSPOSED_HMMA_TEST_INVOCATIONS, 16U)};
    std::vector<std::pair<Id, Id>> pattern;
    for (u32 slot = 0; slot < pattern_words; ++slot) {
        const Id word{module.OpIAdd(u32_type, index,
                                    constant(slot * TRANSPOSED_HMMA_TEST_INVOCATIONS))};
        const Id value{module.OpBitwiseOr(u32_type, word, constant(0xa5000000U))};
        module.OpStore(shared_at(word), value);
        pattern.emplace_back(word, value);
    }
    const Id a{module.OpCompositeConstruct(u32x4_type, load(0), load(1), load(2), load(3))};
    const Id b{module.OpCompositeConstruct(u32x2_type, load(4), load(5))};
    const Id c{module.OpCompositeConstruct(u32x2_type, load(6), load(7))};
    const Id d{module.OpFunctionCall(u32x2_type, function, a, b, c)};
    const Id e{module.OpFunctionCall(u32x2_type, function, a, b, d)};
    for (u32 word = 0; word < 2; ++word) {
        module.OpStore(word_at(8 + word), module.OpCompositeExtract(u32_type, d, word));
        module.OpStore(word_at(10 + word), module.OpCompositeExtract(u32_type, e, word));
    }
    // Every warp's staging is done: the shader's own words must be unchanged.
    module.OpControlBarrier(constant(static_cast<u32>(spv::Scope::Workgroup)),
                            constant(static_cast<u32>(spv::Scope::Workgroup)),
                            constant(static_cast<u32>(spv::MemorySemanticsMask::AcquireRelease) |
                                     static_cast<u32>(spv::MemorySemanticsMask::WorkgroupMemory)));
    Id intact{module.ConstantTrue(bool_type)};
    for (const auto& [word, value] : pattern) {
        intact = module.OpLogicalAnd(bool_type, intact,
                                     module.OpIEqual(bool_type, module.OpLoad(u32_type, shared_at(word)),
                                                     value));
    }
    module.OpStore(word_at(12), module.OpSelect(u32_type, intact, constant(1), constant(0)));
    module.OpReturn();
    module.OpFunctionEnd();

    // Since SPIR-V 1.4 the interface lists every global the entry point uses.
    if (spirv_version >= 0x00010400U) {
        interfaces.push_back(buffer);
        interfaces.push_back(staging.variable);
        if (shared_words != 0) {
            interfaces.push_back(shared);
        }
    }
    module.AddEntryPoint(spv::ExecutionModel::GLCompute, main, "main", interfaces);
    module.AddExecutionMode(main, spv::ExecutionMode::LocalSize, TRANSPOSED_HMMA_TEST_INVOCATIONS,
                            1U, 1U);

    std::vector<u32> code{module.Assemble()};
    PatchCooperativeHmma(code, types);
    return code;
}

} // namespace Shader::Backend::SPIRV
