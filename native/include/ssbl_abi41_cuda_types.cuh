// Reconstructed SSBL CUDA type skeleton.
// Source priority: sm_89 instructions/demangled symbols from cuda_extract.
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
    float m00;
    float m11;
    float m22;
    float m01;
    float m02;
    float m12;
};

constexpr u32 kPinnedOrKinematicFlag = 0x4;
constexpr u32 kProbableMaxSelfCollisionNeighbors = 32;

static_assert(sizeof(float3) == 12, "SSBL reconstructed float3 must stay 12 bytes.");
static_assert(sizeof(CudaSpringPBD) == 12, "CudaSpringPBD must stay 12 bytes.");
static_assert(sizeof(CudaTriangle) == 12, "CudaTriangle must stay 12 bytes.");
static_assert(sizeof(symMatCuda) == 24, "symMatCuda must stay 24 bytes.");

} // namespace ssbl_abi41
