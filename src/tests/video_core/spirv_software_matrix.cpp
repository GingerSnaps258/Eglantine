// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <span>
#include <string>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/maxwell/decode.h"

namespace {
using namespace Shader;

struct TensorProgram {
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::Program program;
    Profile profile;

    explicit TensorProgram(u32 count = 1, u32 software_depth = 0) {
        program.stage = Stage::Compute;
        program.workgroup_size = {64, 1, 1};
        program.blocks = {&block};
        program.post_order_blocks = {&block};
        program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                               {{}, IR::AbstractSyntaxNode::Type::Return}};
        program.info.uses_subgroup_shuffles = count != 0 && software_depth != 0;
        program.info.uses_subgroup_invocation_id = count != 0;
        profile.supported_spirv = 0x00010500;
        profile.supported_subgroup_stages = 1U << static_cast<u32>(Stage::Compute);
        profile.max_subgroup_size = 32;
        IR::IREmitter ir{block};
        if (count == 0) {
            return;
        }
        const auto lane = ir.LaneId();
        std::array<IR::U32, 8> inputs;
        for (u32 i = 0; i < inputs.size(); ++i) {
            inputs[i] = software_depth ? ir.IAdd(ir.Imm32(0x3c003c00U + i * 0x01000100U),
                                                 ir.ShiftLeftLogical(lane, ir.Imm32(3)))
                                       : ir.IAdd(ir.Imm32((i + 1) * 0x01000100U), lane);
        }
        for (u32 i = 0; i < count; ++i) {
            const auto result =
                ir.TensorMulAdd(ir.CompositeConstruct(inputs[0], inputs[1], inputs[2], inputs[3]),
                                ir.CompositeConstruct(inputs[4], inputs[5]),
                                ir.CompositeConstruct(inputs[6], inputs[7]), software_depth);
            inputs[6] = IR::U32{ir.CompositeExtract(result, 0)};
            inputs[7] = IR::U32{ir.CompositeExtract(result, 1)};
        }
        if (software_depth) {
            program.info.used_storage_buffer_types = IR::Type::U32;
            program.info.storage_buffers_descriptors = {{0, 0, 1, true}};
            program.info.uses_local_invocation_id = true;
            const auto offset = ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(3));
            block.AppendNewInst(IR::Opcode::WriteStorage32, {ir.Imm32(0), offset, inputs[6]});
            block.AppendNewInst(IR::Opcode::WriteStorage32,
                                {ir.Imm32(0), ir.IAdd(offset, ir.Imm32(4)), inputs[7]});
        }
    }

    std::vector<u32> Emit(std::string_view name = {}) {
        auto code = Backend::SPIRV::EmitSPIRV(profile, program);
        if (const char* directory = std::getenv("EDEN_TENSOR_TEST_OUTPUT");
            directory && !name.empty()) {
            std::filesystem::create_directories(directory);
            std::ofstream file{std::filesystem::path{directory} / (std::string{name} + ".spv"),
                               std::ios::binary};
            file.write(reinterpret_cast<const char*>(code.data()),
                       static_cast<std::streamsize>(code.size() * sizeof(u32)));
            REQUIRE(file.good());
        }
        return code;
    }
};

std::vector<std::vector<u32>> Instructions(const std::vector<u32>& code, u32 opcode) {
    std::vector<std::vector<u32>> result;
    for (size_t offset = 5; offset < code.size();) {
        const u32 size = code[offset] >> 16;
        REQUIRE(size != 0);
        REQUIRE(offset + size <= code.size());
        if ((code[offset] & 0xffffU) == opcode) {
            result.emplace_back(code.begin() + offset + 1, code.begin() + offset + size);
        }
        offset += size;
    }
    return result;
}

bool HasCapability(const std::vector<u32>& code, u32 capability) {
    const auto capabilities = Instructions(code, 17);
    return std::ranges::any_of(capabilities,
                               [=](const auto& words) { return words[0] == capability; });
}

} // namespace

TEST_CASE("Software HMMA shares one helper per depth without tensor extensions",
          "[video_core][tensor][ampere]") {
    for (const u32 depth : {8U, 16U}) {
        TensorProgram single{1, depth};
        const auto first = single.Emit(depth == 8 ? "software-k8" : "software-k16");
        REQUIRE_FALSE(HasCapability(first, 5357));
        REQUIRE(Instructions(first, 57).size() == 1); // OpFunctionCall
        TensorProgram repeated{64, depth};
        const auto many = repeated.Emit(depth == 8 ? "software-k8-many" : "software-k16-many");
        REQUIRE(Instructions(many, 57).size() == 64);
        // Three fragment-vector constructions, the call and two result extracts
        // take at most 40 words per site; the matrix body itself is never duplicated.
        REQUIRE(many.size() < first.size() + 64 * 40);
    }
}

TEST_CASE("Software HMMA rounds once per uniform K8 group", "[video_core][tensor][ampere]") {
    for (const u32 depth : {8U, 16U}) {
        TensorProgram fixture{1, depth};
        const auto code = fixture.Emit();
        std::map<u32, u32> constants;
        for (const auto& words : Instructions(code, 43)) { // OpConstant
            constants.emplace(words[1], words[2]);
        }
        // One K=8 group is emitted once; K=16 runs it twice from a uniform counter.
        const auto loops = Instructions(code, 246); // OpLoopMerge
        if (depth == 8) {
            REQUIRE(loops.empty());
        } else {
            REQUIRE(loops.size() == 1);
            const auto limits = Instructions(code, 176); // OpULessThan
            REQUIRE(limits.size() == 1);
            REQUIRE(constants.at(limits.front()[3]) == 2);
            const auto loads = Instructions(code, 61); // OpLoad
            const auto counter_load = std::ranges::find_if(
                loads, [&](const auto& words) { return words[1] == limits.front()[2]; });
            REQUIRE(counter_load != loads.end());
            const u32 counter = (*counter_load)[2];
            u32 counter_stores = 0;
            for (const auto& store : Instructions(code, 62)) { // OpStore
                if (store[0] != counter) {
                    continue;
                }
                ++counter_stores;
                if (constants.contains(store[1])) {
                    REQUIRE(constants.at(store[1]) == 0);
                }
            }
            REQUIRE(counter_stores == 2); // Zero, then the increment: no lane input.
        }
        u32 rounding_operations = 0;
        u32 fma_operations = 0;
        for (const auto& words : Instructions(code, 12)) { // OpExtInst
            rounding_operations += words[3] == 2;          // GLSL.std.450 RoundEven
            fma_operations += words[3] == 50;              // GLSL.std.450 Fma
        }
        REQUIRE(rounding_operations == 4); // One per accumulator, after each K=8 group.
        REQUIRE(fma_operations == 32);
        REQUIRE(Instructions(code, 345).size() == 16); // OpGroupNonUniformShuffle
        REQUIRE(Instructions(code, 247).empty());      // No per-pair rounding branch.
    }
}

TEST_CASE("Software matrix helpers also emit for 64-lane host subgroups",
          "[video_core][tensor][ampere]") {
    for (const u32 depth : {8U, 16U}) {
        TensorProgram fixture{1, depth};
        fixture.profile.max_subgroup_size = 64;
        const auto code = fixture.Emit(depth == 8 ? "software-k8-wave64" : "software-k16-wave64");
        REQUIRE_FALSE(HasCapability(code, 5357));
        REQUIRE(Instructions(code, 345).size() == 16); // OpGroupNonUniformShuffle
    }
}

namespace {

constexpr u32 OP_COMPOSITE_EXTRACT = 81;
constexpr u32 OP_COMPOSITE_INSERT = 82;
constexpr u32 OP_BITCAST = 124;
constexpr u32 OP_TYPE_COOPERATIVE_MATRIX = 4456;
constexpr u32 OP_COOPERATIVE_MATRIX_MUL_ADD = 4459;
constexpr u32 CAPABILITY_COOPERATIVE_MATRIX = 6022;
constexpr u32 CAPABILITY_FLOAT16 = 9;
constexpr u32 CAPABILITY_VULKAN_MEMORY_MODEL = 5345;
constexpr u32 CAPABILITY_VULKAN_MEMORY_MODEL_DEVICE_SCOPE = 5346;

std::map<u32, u32> Constants(const std::vector<u32>& code) {
    std::map<u32, u32> constants;
    for (const auto& words : Instructions(code, 43)) { // OpConstant
        constants.emplace(words[1], words[2]);
    }
    return constants;
}

void RequireHmmaMatrixTypes(const std::vector<u32>& code) {
    const auto constants = Constants(code);
    const auto types = Instructions(code, OP_TYPE_COOPERATIVE_MATRIX);
    REQUIRE(types.size() == 3);
    std::array<u32, 3> columns_by_use{};
    for (const auto& words : types) {
        REQUIRE(words.size() == 6);
        REQUIRE(constants.at(words[2]) == 3); // Subgroup scope
        REQUIRE(constants.at(words[3]) == 16);
        columns_by_use.at(constants.at(words[5])) = constants.at(words[4]);
    }
    REQUIRE(columns_by_use == std::array<u32, 3>{16, 8, 8}); // A, B, accumulator
}

// SPV_KHR_cooperative_matrix requires the Vulkan memory model, which spirv-val does not check.
void RequireVulkanMemoryModel(const std::vector<u32>& code) {
    REQUIRE(HasCapability(code, CAPABILITY_VULKAN_MEMORY_MODEL));
    REQUIRE(HasCapability(code, CAPABILITY_VULKAN_MEMORY_MODEL_DEVICE_SCOPE));
    const auto models = Instructions(code, 14); // OpMemoryModel
    REQUIRE(models.size() == 1);
    REQUIRE(models[0][1] == 3); // Vulkan
    bool extension = false;
    for (const auto& words : Instructions(code, 10)) { // OpExtension
        extension |= std::string{reinterpret_cast<const char*>(words.data())} ==
                     "SPV_KHR_vulkan_memory_model";
    }
    REQUIRE(extension == (code[1] < 0x00010500));
}

} // namespace

TEST_CASE("Cooperative HMMA replaces the K16 software helper", "[video_core][tensor][ampere]") {
    TensorProgram fixture{3, 16};
    fixture.profile.support_cooperative_hmma = true;
    const auto code = fixture.Emit("cooperative-k16");
    REQUIRE(HasCapability(code, CAPABILITY_COOPERATIVE_MATRIX));
    REQUIRE(HasCapability(code, CAPABILITY_FLOAT16));
    RequireVulkanMemoryModel(code);
    REQUIRE(fixture.program.info.uses_cooperative_matrix);
    RequireHmmaMatrixTypes(code);
    REQUIRE(Instructions(code, OP_COOPERATIVE_MATRIX_MUL_ADD).size() == 3);
    REQUIRE(Instructions(code, 246).empty()); // OpLoopMerge
    REQUIRE(Instructions(code, 57).empty());  // OpFunctionCall
    REQUIRE(Instructions(code, 345).empty()); // OpGroupNonUniformShuffle
    for (const auto& words : Instructions(code, 169)) { // OpSelect
        for (const auto& type : Instructions(code, OP_TYPE_COOPERATIVE_MATRIX)) {
            REQUIRE(words[0] != type[0]);
        }
    }
}

TEST_CASE("Cooperative HMMA keeps barriers valid under the Vulkan memory model",
          "[video_core][tensor][ampere]") {
    for (const u32 version : {0x00010400U, 0x00010600U}) {
        TensorProgram fixture{1, 16};
        fixture.profile.supported_spirv = version;
        fixture.profile.support_cooperative_hmma = true;
        IR::IREmitter ir{fixture.block};
        ir.Barrier();
        ir.WorkgroupMemoryBarrier();
        ir.DeviceMemoryBarrier();
        const auto code = fixture.Emit("cooperative-k16-barriers-" + std::to_string(version));
        REQUIRE(code[1] == version);
        RequireVulkanMemoryModel(code);
        REQUIRE(Instructions(code, OP_COOPERATIVE_MATRIX_MUL_ADD).size() == 1);
    }
}

TEST_CASE("Cooperative HMMA keeps the driver's order of A's registers",
          "[video_core][tensor][ampere]") {
    TensorProgram fixture{1, 16};
    fixture.profile.support_cooperative_hmma = true;
    const auto code = fixture.Emit();
    const auto extracts = Instructions(code, OP_COMPOSITE_EXTRACT);
    const auto bitcasts = Instructions(code, OP_BITCAST);
    const auto by_result = [](const auto& instructions, u32 result) {
        const auto it = std::ranges::find_if(
            instructions, [=](const auto& words) { return words[1] == result; });
        REQUIRE(it != instructions.end());
        return *it;
    };
    const auto types = Instructions(code, OP_TYPE_COOPERATIVE_MATRIX);
    const auto constants = Constants(code);
    const auto matrix_a = std::ranges::find_if(
        types, [&](const auto& words) { return constants.at(words[5]) == 0; });
    REQUIRE(matrix_a != types.end());

    std::array<u32, 8> register_of_element{};
    u32 inserts = 0;
    for (const auto& words : Instructions(code, OP_COMPOSITE_INSERT)) {
        if (words[0] != (*matrix_a)[0]) {
            continue;
        }
        const auto half = by_result(extracts, words[2]);
        const auto halves = by_result(bitcasts, half[2]);
        const auto guest_register = by_result(extracts, halves[2]);
        REQUIRE(half[3] == (words[4] & 1));
        register_of_element.at(words[4]) = guest_register[3];
        ++inserts;
    }
    REQUIRE(inserts == 8);
    REQUIRE(register_of_element == std::array<u32, 8>{0, 0, 2, 2, 1, 1, 3, 3});
}

TEST_CASE("Cooperative HMMA falls back to the software helper", "[video_core][tensor][ampere]") {
    const auto require_software = [](TensorProgram& fixture) {
        const auto code = fixture.Emit();
        REQUIRE_FALSE(HasCapability(code, CAPABILITY_COOPERATIVE_MATRIX));
        REQUIRE(Instructions(code, OP_TYPE_COOPERATIVE_MATRIX).empty());
        REQUIRE(Instructions(code, 57).size() == 1); // OpFunctionCall
        REQUIRE_FALSE(fixture.program.info.uses_cooperative_matrix);
    };
    SECTION("unsupported by the device") {
        TensorProgram fixture{1, 16};
        require_software(fixture);
    }
    SECTION("the K8 form") {
        TensorProgram fixture{1, 8};
        fixture.profile.support_cooperative_hmma = true;
        require_software(fixture);
    }
    SECTION("a workgroup that leaves a warp unfilled") {
        TensorProgram fixture{1, 16};
        fixture.profile.support_cooperative_hmma = true;
        fixture.program.workgroup_size = {48, 1, 1};
        require_software(fixture);
    }
    SECTION("a workgroup row that leaves a warp unfilled") {
        TensorProgram fixture{1, 16};
        fixture.profile.support_cooperative_hmma = true;
        fixture.program.workgroup_size = {16, 2, 1};
        require_software(fixture);
    }
    SECTION("coherent memory, which the Vulkan memory model has no decoration for") {
        TensorProgram fixture{1, 16};
        fixture.profile.support_cooperative_hmma = true;
        fixture.program.info.requires_coherent_global_memory = true;
        require_software(fixture);
    }
}

TEST_CASE("Cooperative HMMA self-test shader runs one lowering on one warp",
          "[video_core][tensor][ampere]") {
    for (const u32 version : {0x00010300U, 0x00010600U}) {
        const auto code = Backend::SPIRV::EmitCooperativeHmmaTest(version);
        REQUIRE(code[1] == version);
        REQUIRE(HasCapability(code, CAPABILITY_COOPERATIVE_MATRIX));
        RequireVulkanMemoryModel(code);
        RequireHmmaMatrixTypes(code);
        REQUIRE(Instructions(code, OP_COOPERATIVE_MATRIX_MUL_ADD).size() == 1);
        const auto modes = Instructions(code, 16); // OpExecutionMode
        REQUIRE(modes.size() == 1);
        REQUIRE(modes[0][1] == 17); // LocalSize
        REQUIRE(modes[0][2] == 32);
        if (const char* directory = std::getenv("EDEN_TENSOR_TEST_OUTPUT")) {
            std::filesystem::create_directories(directory);
            std::ofstream file{std::filesystem::path{directory} /
                                   ("cooperative-self-test-" + std::to_string(version) + ".spv"),
                               std::ios::binary};
            file.write(reinterpret_cast<const char*>(code.data()),
                       static_cast<std::streamsize>(code.size() * sizeof(u32)));
        }
    }
}

TEST_CASE("Compact HMMA lowering emits one helper call per matrix multiply",
          "[video_core][tensor][ampere]") {
    // HMMA.16816.F16 R44, R16, R48, RZ (Donkey Kong Bananza capture) and EXIT.
    constexpr Ampere::Instruction Hmma{{0x102c723c, 0x30, 0x8ff, 0x04aff000}};
    constexpr Ampere::Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};
    const std::array code{Hmma, Exit};
    const auto is_tensor = [](u64 word) { return (word >> 51) == (0x5d30000000000000ULL >> 51); };

    const auto expanded = Ampere::LowerToMaxwell(code, {.tensor = true});
    INFO(expanded.error);
    REQUIRE(static_cast<bool>(expanded));
    REQUIRE(std::ranges::none_of(expanded.instructions, is_tensor));

    const auto compact = Ampere::LowerToMaxwell(code, {.tensor = true, .compact_tensor = true});
    INFO(compact.error);
    REQUIRE(static_cast<bool>(compact));
    REQUIRE(std::ranges::count_if(compact.instructions, is_tensor) == 1);
    const u64 word = *std::ranges::find_if(compact.instructions, is_tensor);
    CHECK((word & 0xff) == 0x2c);
    CHECK(((word >> 8) & 0xff) == 0x10);
    CHECK(((word >> 20) & 0xff) == 0x30);
    CHECK(((word >> 39) & 0xff) == 0xff);
    CHECK(((word >> 48) & 1) == 1);
    CHECK(((word >> 49) & 1) == 0);
    CHECK(compact.instructions.size() * 4 < expanded.instructions.size());

    // A guarded HMMA keeps the expansion: the helper shuffles across the whole warp.
    auto guarded = code;
    guarded[0].words[0] = (guarded[0].words[0] & ~0xf000U) | (1U << 12);
    const auto guarded_lowered =
        Ampere::LowerToMaxwell(guarded, {.tensor = true, .compact_tensor = true});
    INFO(guarded_lowered.error);
    REQUIRE(static_cast<bool>(guarded_lowered));
    CHECK(std::ranges::none_of(guarded_lowered.instructions, is_tensor));
}

namespace {

constexpr u32 OP_FUNCTION_CALL = 57;
constexpr u32 OP_TYPE_ARRAY = 28;
constexpr u32 OP_DECORATE = 71;
constexpr u32 OP_MEMBER_DECORATE = 72;
constexpr u32 OP_COOPERATIVE_MATRIX_LOAD = 4457;
constexpr u32 OP_COOPERATIVE_MATRIX_STORE = 4458;
constexpr u32 DECORATION_ARRAY_STRIDE = 6;
constexpr u32 DECORATION_OFFSET = 35;
constexpr u32 TILE_HALVES = 640; // per warp: A, B low, B high, C/D

void UseTransposedPath(TensorProgram& fixture, u32 shared_bytes = 0, bool explicit_layout = false,
                       bool layout_16bit = true) {
    fixture.profile.support_transposed_cooperative_hmma = true;
    fixture.profile.max_compute_shared_memory_size = 48 * 1024;
    fixture.profile.support_explicit_workgroup_layout = explicit_layout;
    fixture.profile.support_workgroup_layout_16bit_access = layout_16bit;
    fixture.program.shared_memory_size = shared_bytes;
}

// Writes the module (EDEN_TENSOR_TEST_OUTPUT, else a temporary folder) and, when EDEN_SPIRV_VAL
// names spirv-val, requires it to accept the module for Vulkan 1.3.
bool ValidateModule(const std::vector<u32>& code, const std::string& name) {
    const char* const output = std::getenv("EDEN_TENSOR_TEST_OUTPUT");
    const char* const validator = std::getenv("EDEN_SPIRV_VAL");
    if (output == nullptr && validator == nullptr) {
        return false;
    }
    const std::filesystem::path directory =
        output ? std::filesystem::path{output}
               : std::filesystem::temp_directory_path() / "eden-tensor-test";
    std::filesystem::create_directories(directory);
    const std::filesystem::path path = directory / (name + ".spv");
    {
        std::ofstream file{path, std::ios::binary};
        file.write(reinterpret_cast<const char*>(code.data()),
                   static_cast<std::streamsize>(code.size() * sizeof(u32)));
        REQUIRE(file.good());
    }
    if (validator == nullptr) {
        return false;
    }
    std::string command = "\"" + std::string{validator} + "\" --target-env vulkan1.3 \"" +
                          path.string() + "\"";
#ifdef _WIN32
    command = "\"" + command + "\""; // cmd.exe strips the outer quotes
#endif
    INFO(command);
    REQUIRE(std::system(command.c_str()) == 0);
    return true;
}

void RequireTransposedTypes(const std::vector<u32>& code) {
    const auto constants = Constants(code);
    const auto types = Instructions(code, OP_TYPE_COOPERATIVE_MATRIX);
    REQUIRE(types.size() == 3);
    std::array<std::pair<u32, u32>, 3> shape_by_use{};
    for (const auto& words : types) {
        REQUIRE(constants.at(words[2]) == 3); // Subgroup scope
        shape_by_use.at(constants.at(words[5])) = {constants.at(words[3]), constants.at(words[4])};
    }
    // A = B^T (8x16), B = A^T (16x16), accumulator = C^T (8x16): Intel's M8 N16 K16.
    REQUIRE(shape_by_use == std::array<std::pair<u32, u32>, 3>{{{8, 16}, {16, 16}, {8, 16}}});
}

// One helper: four column-major loads, two products (one per K half), one store, with the
// memory operands the lanes' sharing needs under the Vulkan memory model.
void RequireTransposedHelper(const std::vector<u32>& code) {
    const auto constants = Constants(code);
    RequireTransposedTypes(code);
    REQUIRE(HasCapability(code, CAPABILITY_COOPERATIVE_MATRIX));
    RequireVulkanMemoryModel(code);
    const auto loads = Instructions(code, OP_COOPERATIVE_MATRIX_LOAD);
    REQUIRE(loads.size() == 4);
    for (const auto& words : loads) {
        REQUIRE(words.size() == 7);                // type, result, pointer, layout, stride, mask, scope
        REQUIRE(constants.at(words[3]) == 1);      // ColumnMajorKHR
        const u32 stride = constants.at(words[4]); // in halves
        REQUIRE((stride == 8 || stride == 16));
        REQUIRE((stride * 2) % 16 == 0);           // VUID-RuntimeSpirv-OpCooperativeMatrixLoadKHR-08986
        REQUIRE(words[5] == 0x30);                 // NonPrivatePointer | MakePointerVisible
        REQUIRE(constants.at(words[6]) == 2);      // Workgroup
    }
    const auto stores = Instructions(code, OP_COOPERATIVE_MATRIX_STORE);
    REQUIRE(stores.size() == 1);
    REQUIRE(stores[0].size() == 6);
    REQUIRE(constants.at(stores[0][2]) == 1);
    REQUIRE(constants.at(stores[0][3]) == 8);
    REQUIRE(stores[0][4] == 0x28); // NonPrivatePointer | MakePointerAvailable
    REQUIRE(Instructions(code, OP_COOPERATIVE_MATRIX_MUL_ADD).size() == 2);
    // No placeholder survives the patch.
    for (const auto& words : Instructions(code, 169)) { // OpSelect
        for (const auto& type : Instructions(code, OP_TYPE_COOPERATIVE_MATRIX)) {
            REQUIRE(words[0] != type[0]);
        }
    }
}

// The staging array: TILE_HALVES halves per warp. Returns its OpTypeArray result id.
u32 StagingArray(const std::vector<u32>& code, u32 warps) {
    const auto constants = Constants(code);
    for (const auto& words : Instructions(code, OP_TYPE_ARRAY)) {
        const auto length = constants.find(words[2]);
        if (length != constants.end() && length->second == warps * TILE_HALVES) {
            return words[0];
        }
    }
    FAIL("no staging array");
    return 0;
}

} // namespace

TEST_CASE("Transposed cooperative HMMA is opt-in and never replaces the direct path",
          "[video_core][tensor][ampere]") {
    SECTION("default: shader emulation") {
        TensorProgram fixture{3, 16};
        fixture.profile.max_compute_shared_memory_size = 48 * 1024;
        const auto code = fixture.Emit();
        REQUIRE_FALSE(HasCapability(code, CAPABILITY_COOPERATIVE_MATRIX));
        REQUIRE_FALSE(fixture.program.info.uses_transposed_cooperative_hmma);
    }
    SECTION("transposed only") {
        TensorProgram fixture{3, 16};
        UseTransposedPath(fixture);
        const auto code = fixture.Emit("transposed-k16");
        RequireTransposedHelper(code);
        REQUIRE(fixture.program.info.uses_cooperative_matrix);
        REQUIRE(fixture.program.info.uses_transposed_cooperative_hmma);
        REQUIRE(Instructions(code, OP_FUNCTION_CALL).size() == 3); // one helper, three calls
        REQUIRE(Instructions(code, 345).empty());                  // no software helper
        ValidateModule(code, "transposed-k16");
    }
    SECTION("both: the direct path wins") {
        TensorProgram fixture{3, 16};
        UseTransposedPath(fixture);
        fixture.profile.support_cooperative_hmma = true;
        const auto code = fixture.Emit();
        RequireHmmaMatrixTypes(code);
        REQUIRE(Instructions(code, OP_COOPERATIVE_MATRIX_LOAD).empty());
        REQUIRE(Instructions(code, OP_COOPERATIVE_MATRIX_MUL_ADD).size() == 3);
        REQUIRE(fixture.program.info.uses_cooperative_matrix);
        REQUIRE_FALSE(fixture.program.info.uses_transposed_cooperative_hmma);
    }
}

TEST_CASE("Transposed cooperative HMMA stages past the program's workgroup memory",
          "[video_core][tensor][ampere]") {
    // 6,900 bytes: not 16-byte aligned, so the tiles must start at 6,912.
    constexpr u32 shared_bytes = 6900;
    constexpr u32 staging_base = 6912;
    constexpr u32 warps = 2; // the fixture's 64 invocations
    STATIC_REQUIRE(TILE_HALVES * 2 % 16 == 0);
    for (const bool explicit_layout : {false, true}) {
        DYNAMIC_SECTION((explicit_layout ? "explicit" : "plain") << " layout") {
            TensorProgram fixture{1, 16};
            UseTransposedPath(fixture, shared_bytes, explicit_layout);
            const std::string name = explicit_layout ? "transposed-explicit" : "transposed-plain";
            const auto code = fixture.Emit(name);
            RequireTransposedHelper(code);
            const u32 array = StagingArray(code, warps);
            bool strided = false;
            for (const auto& words : Instructions(code, OP_DECORATE)) {
                strided |= words[0] == array && words[1] == DECORATION_ARRAY_STRIDE && words[2] == 2;
            }
            u32 staging_offsets = 0;
            for (const auto& words : Instructions(code, OP_MEMBER_DECORATE)) {
                if (words[2] == DECORATION_OFFSET && words[3] != 0) {
                    REQUIRE(words[3] == staging_base);
                    REQUIRE(words[3] % 16 == 0);
                    ++staging_offsets;
                }
            }
            if (explicit_layout) {
                // Block members alias from byte 0: the tiles' member starts past the program's.
                REQUIRE(strided);
                REQUIRE(staging_offsets == 1);
                REQUIRE(HasCapability(code, 4430)); // WorkgroupMemoryExplicitLayout16BitAccessKHR
            } else {
                // A separate, unaliased variable.
                REQUIRE_FALSE(strided);
                REQUIRE(staging_offsets == 0);
            }
            ValidateModule(code, name);
        }
    }
}

TEST_CASE("Transposed cooperative HMMA falls back to shader emulation",
          "[video_core][tensor][ampere]") {
    const auto require_software = [](TensorProgram& fixture) {
        const auto code = fixture.Emit();
        REQUIRE_FALSE(HasCapability(code, CAPABILITY_COOPERATIVE_MATRIX));
        REQUIRE(Instructions(code, OP_TYPE_COOPERATIVE_MATRIX).empty());
        REQUIRE(Instructions(code, OP_FUNCTION_CALL).size() == 1);
        REQUIRE_FALSE(fixture.program.info.uses_cooperative_matrix);
        REQUIRE_FALSE(fixture.program.info.uses_transposed_cooperative_hmma);
    };
    SECTION("workgroup memory one byte short of the tiles") {
        TensorProgram fixture{1, 16};
        UseTransposedPath(fixture, 6912);
        fixture.profile.max_compute_shared_memory_size = 6912 + 2 * TILE_HALVES * 2 - 1;
        require_software(fixture);
    }
    SECTION("workgroup memory exactly enough") {
        TensorProgram fixture{1, 16};
        UseTransposedPath(fixture, 6912);
        fixture.profile.max_compute_shared_memory_size = 6912 + 2 * TILE_HALVES * 2;
        const auto code = fixture.Emit();
        REQUIRE(fixture.program.info.uses_transposed_cooperative_hmma);
    }
    SECTION("explicit layout without 16-bit workgroup access") {
        TensorProgram fixture{1, 16};
        UseTransposedPath(fixture, 6912, true, false);
        require_software(fixture);
    }
    SECTION("a workgroup that leaves a warp unfilled") {
        TensorProgram fixture{1, 16};
        UseTransposedPath(fixture);
        fixture.program.workgroup_size = {48, 1, 1};
        require_software(fixture);
    }
    SECTION("coherent memory") {
        TensorProgram fixture{1, 16};
        UseTransposedPath(fixture);
        fixture.program.info.requires_coherent_global_memory = true;
        require_software(fixture);
    }
}

// A program can mix paths: its K=16 HMMAs take the transposed helper while its K=8 ones keep the
// software helper (and guarded HMMAs never reach the backend, see the lowering test below).
TEST_CASE("Transposed cooperative HMMA mixes with the K8 software helper",
          "[video_core][tensor][ampere]") {
    TensorProgram fixture{1, 16};
    UseTransposedPath(fixture, 2048, true);
    IR::IREmitter ir{fixture.block};
    const auto one = ir.Imm32(0x3c003c00U);
    const auto k8 = ir.TensorMulAdd(ir.CompositeConstruct(one, one, one, one),
                                    ir.CompositeConstruct(one, one),
                                    ir.CompositeConstruct(ir.Imm32(0), ir.Imm32(0)), 8);
    fixture.block.AppendNewInst(IR::Opcode::WriteStorage32,
                                {ir.Imm32(0), ir.Imm32(1024), ir.CompositeExtract(k8, 0)});
    const auto code = fixture.Emit("transposed-mixed-k8");
    RequireTransposedHelper(code);
    REQUIRE(fixture.program.info.uses_transposed_cooperative_hmma);
    REQUIRE(Instructions(code, OP_FUNCTION_CALL).size() == 2); // transposed helper, K8 helper
    REQUIRE(Instructions(code, 345).size() == 16);             // the K8 helper's shuffles
    ValidateModule(code, "transposed-mixed-k8");
}

TEST_CASE("Compact HMMA lowering keeps a guarded HMMA expanded beside a compact one",
          "[video_core][tensor][ampere]") {
    constexpr Ampere::Instruction Hmma{{0x102c723c, 0x30, 0x8ff, 0x04aff000}};
    constexpr Ampere::Instruction Exit{{0x0000794d, 0, 0x03800000, 0}};
    Ampere::Instruction guarded = Hmma;
    guarded.words[0] = (guarded.words[0] & ~0xf000U) | (1U << 12);
    const std::array code{Hmma, guarded, Exit};
    const auto is_tensor = [](u64 word) { return (word >> 51) == (0x5d30000000000000ULL >> 51); };
    const auto lowered = Ampere::LowerToMaxwell(code, {.tensor = true, .compact_tensor = true});
    INFO(lowered.error);
    REQUIRE(static_cast<bool>(lowered));
    REQUIRE(std::ranges::count_if(lowered.instructions, is_tensor) == 1);
    const std::array single_code{Hmma, Exit};
    const auto single = Ampere::LowerToMaxwell(single_code, {.tensor = true, .compact_tensor = true});
    REQUIRE(static_cast<bool>(single));
    CHECK(lowered.instructions.size() > single.instructions.size() * 4); // the expansion
}

TEST_CASE("Transposed cooperative HMMA self-test shader runs the emitted helper",
          "[video_core][tensor][ampere]") {
    for (const u32 version : {0x00010300U, 0x00010400U, 0x00010600U}) {
        for (const bool explicit_layout : {false, true}) {
            if (explicit_layout && version < 0x00010400U) {
                // SPV_KHR_workgroup_memory_explicit_layout needs SPIR-V 1.4.
                REQUIRE_THROWS(Backend::SPIRV::EmitTransposedCooperativeHmmaTest(version, 6912, true));
                continue;
            }
            const auto code =
                Backend::SPIRV::EmitTransposedCooperativeHmmaTest(version, 6912, explicit_layout);
            REQUIRE(code[1] == version);
            RequireTransposedHelper(code);
            REQUIRE(Instructions(code, OP_FUNCTION_CALL).size() == 2); // two chained HMMAs
            const auto modes = Instructions(code, 16); // OpExecutionMode
            REQUIRE(modes.size() == 1);
            REQUIRE(modes[0][2] == Backend::SPIRV::TRANSPOSED_HMMA_TEST_INVOCATIONS);
            ValidateModule(code, "transposed-self-test-" + std::to_string(version) +
                                     (explicit_layout ? "-explicit" : "-plain"));
        }
    }
}

TEST_CASE("Transposed cooperative HMMA modules pass spirv-val", "[video_core][tensor][ampere]") {
    if (std::getenv("EDEN_SPIRV_VAL") == nullptr) {
        SKIP("set EDEN_SPIRV_VAL to spirv-val to validate the transposed modules");
    }
    for (const bool explicit_layout : {false, true}) {
        TensorProgram fixture{2, 16};
        UseTransposedPath(fixture, 6912, explicit_layout);
        const auto code = fixture.Emit();
        REQUIRE(ValidateModule(code, explicit_layout ? "transposed-val-explicit"
                                                     : "transposed-val-plain"));
    }
}
