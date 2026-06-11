// SSBL ABI41 CUDA type skeleton used by the ABI41 backend.
// Keep these helper structs layout-stable with the ABI41 CUDA implementation.
#pragma once

#include <cstdint>

namespace ssbl_abi41 {

using u32 = std::uint32_t;

struct float3 {
    float x;
    float y;
    float z;
};

struct CudaSpringPBD {
    u32 id0;
    u32 id1;
    float rest_length;
};

struct CudaTriangle {
    u32 v0;
    u32 v1;
    u32 v2;
};

struct symMatCuda {
    float m11;
    float m12;
    float m13;
    float m22;
    float m23;
    float m33;
};

constexpr u32 kPinnedOrKinematicFlag = 0x4;
constexpr u32 kProbableMaxSelfCollisionNeighbors = 32;

static_assert(sizeof(float3) == 12, "SSBL ABI41 float3 must stay 12 bytes.");
static_assert(sizeof(CudaSpringPBD) == 12, "CudaSpringPBD must stay 12 bytes.");
static_assert(sizeof(CudaTriangle) == 12, "CudaTriangle must stay 12 bytes.");
static_assert(sizeof(symMatCuda) == 24, "symMatCuda must stay 24 bytes.");

} // namespace ssbl_abi41
