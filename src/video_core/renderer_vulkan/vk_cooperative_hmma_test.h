// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace Vulkan {

class Device;

/// Runs the HMMA.16816.F16 cooperative matrix lowering once on known matrices and compares every
/// lane's result. Vulkan leaves the per-lane order of a cooperative matrix's elements to the driver
/// and the lowering relies on it, so a driver with another order keeps the shader emulation.
[[nodiscard]] bool TestCooperativeHmma(const Device& device);

/// Runs the transposed M8 N16 K16 path (the helper the emulator emits) on four warps beside the
/// shader's own workgroup memory, with integer matrices (lane layout), cases where the guest's
/// rounding after each K=8 group differs from rounding once, and a second HMMA on the first's
/// result; checks that the shader's workgroup memory is untouched. Plain and, where supported,
/// explicit workgroup layouts.
[[nodiscard]] bool TestTransposedCooperativeHmma(const Device& device);

} // namespace Vulkan
