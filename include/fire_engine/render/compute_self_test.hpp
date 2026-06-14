#pragma once

#include <fire_engine/render/compute_pipeline.hpp>

namespace fire_engine
{

class Device;

// The ComputePipelineConfig for the self-test kernel (compute_selftest.comp):
// one storage-buffer binding + a push-constant range. Exposed so it can be
// validated GPU-free in tests, mirroring the graphics pipeline-config factories.
[[nodiscard]] ComputePipelineConfig computeSelfTestConfig();

// Exercises the compute path end-to-end: builds a ComputePipeline + SSBO, runs
// two chained dispatches separated by a synchronization2 buffer barrier, then
// reads the buffer back and asserts the result. Logs "compute path OK" on
// success; throws std::runtime_error on mismatch. Debug-only smoke test.
void runComputeSelfTest(const Device& device);

} // namespace fire_engine
