#include "ssbl_xpbd_cuda.h"
#include "ssbl_abi41_cuda_types.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using ReconSpring = ssbl_abi41::CudaSpringPBD;
using ReconSymMat = ssbl_abi41::symMatCuda;
using ReconTriangle = ssbl_abi41::CudaTriangle;

constexpr int kThreads = 256;
constexpr float kEps = 1.0e-7f;
constexpr int kAbi41ParticleHashMinCount = 128;
constexpr int kAbi41ParticleHashBucketSlots = 16;
constexpr int kAbi41MinParticleHashBuckets = 1024;
constexpr int kAbi41SelfHashMinCount = 128;
constexpr int kAbi41SelfHashBucketSlots = 32;
constexpr int kAbi41MinSelfHashBuckets = 1024;
constexpr int kAbi41SelfTriangleHashMinCount = 64;
constexpr int kAbi41SelfTriangleHashBucketSlots = 32;
constexpr int kAbi41MinSelfTriangleHashBuckets = 1024;
constexpr int kAbi41SelfEdgeHashMinCount = 64;
constexpr int kAbi41SelfEdgeHashBucketSlots = 32;
constexpr int kAbi41MinSelfEdgeHashBuckets = 1024;
constexpr int kAbi41SelfCollisionNeighborSlots = 32;
constexpr int kAbi41TriangleHashMinCount = 64;
constexpr int kAbi41TriangleHashBucketSlots = 32;
constexpr int kAbi41MinTriangleHashBuckets = 1024;
constexpr int kAbi41CountSoftContacts = 0;
constexpr int kAbi41CountExactImpulseContacts = 1;
constexpr int kAbi41CountEdgeEdgeContacts = 2;
constexpr int kAbi41CountHardFallbacks = 3;
constexpr int kAbi41CountDynamicParticleCandidates = 4;
constexpr int kAbi41CountDynamicParticleContacts = 5;
constexpr int kAbi41CountDynamicParticleOverflow = 6;
constexpr int kAbi41CountTrianglePairs = 7;
constexpr int kAbi41CountTrianglePairOverflow = 8;
constexpr int kAbi41CountSelfCandidates = 9;
constexpr int kAbi41CountSelfOverflow = 10;
constexpr int kAbi41CountStaticSdfContacts = 11;
constexpr int kAbi41CountSlots = 12;
constexpr float kAbi41SpringRelaxation = 0.18f;
constexpr float kAbi41SelfAveragingClampScale = 0.35f;
constexpr int kAbi41PcgMaxIterations = 1;
constexpr int kAbi41PcgReductionDAD = 0;
constexpr int kAbi41PcgReductionRZ = 1;
constexpr int kAbi41PcgReductionRZNext = 2;
constexpr int kAbi41PcgReductionStatus = 3;
constexpr int kAbi41PcgReductionSlots = 4;
constexpr float kAbi41PcgStatusOk = 0.0f;
constexpr float kAbi41PcgStatusBadResidual = 1.0f;
constexpr float kAbi41PcgStatusZeroResidual = 2.0f;
constexpr float kAbi41PcgStatusBadDAD = 3.0f;
constexpr float kAbi41PcgStatusBadAlpha = 4.0f;
constexpr int kAbi41MaxForceFields = 64;
constexpr int kAbi41ForceFieldWind = 1;
constexpr int kAbi41ForceFieldForce = 2;
constexpr int kAbi41ForceFieldVortex = 3;
constexpr int kAbi41ForceFieldTurbulence = 4;
constexpr int kAbi41ForceFieldCharge = 5;
constexpr int kAbi41ForceFieldHarmonic = 6;
constexpr int kAbi41ForceFieldLennardJ = 7;
constexpr int kAbi41ForceFieldMagnet = 8;
constexpr int kAbi41ForceFieldDrag = 9;
constexpr int kAbi41ForceFieldTexture = 10;
constexpr float kAbi41MaxForceFieldAcceleration = 5000.0f;

std::string g_last_error;

bool abi41_env_flag(const char* name, bool default_value) {
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') {
        return default_value;
    }
    return std::strcmp(raw, "0") != 0
        && std::strcmp(raw, "false") != 0
        && std::strcmp(raw, "FALSE") != 0
        && std::strcmp(raw, "off") != 0
        && std::strcmp(raw, "OFF") != 0
        && std::strcmp(raw, "no") != 0
        && std::strcmp(raw, "NO") != 0;
}

bool abi41_pcg_device_scalar_enabled() {
    static const bool enabled = abi41_env_flag("SSBL_ABI41_PCG_DEVICE_SCALAR", true);
    return enabled;
}

struct Vec3 {
    float x;
    float y;
    float z;
};

static_assert(sizeof(Vec3) == 12, "Recon Vec3 must match CUDA float3 stride.");

struct TriangleProximityPair {
    int vertex;
    int triangle;
    int source;
    int reserved;
    Vec3 delta;
};

struct ReconCSRTextureObject {
    cudaTextureObject_t tex = 0;

    __device__ __forceinline__ ReconSymMat getMatrixBlock(unsigned int index) const {
        const float4 data1 = tex1Dfetch<float4>(tex, static_cast<int>(index * 2u));
        const float4 data2 = tex1Dfetch<float4>(tex, static_cast<int>(index * 2u + 1u));
        ReconSymMat mat{};
        mat.m11 = data1.x;
        mat.m12 = data1.y;
        mat.m13 = data1.z;
        mat.m22 = data1.w;
        mat.m23 = data2.x;
        mat.m33 = data2.y;
        return mat;
    }
};

struct Abi41Solver {
    SsblXpbdConfig cfg{};
    Vec3* pos = nullptr;
    Vec3* prev = nullptr;
    Vec3* vel = nullptr;
    Vec3* rest = nullptr;
    float* inv_mass = nullptr;
    unsigned int* state_flags = nullptr;
    ReconSpring* springs = nullptr;
    ReconTriangle* triangles = nullptr;
    int* surface_vertex_offsets = nullptr;
    int* surface_vertex_triangles = nullptr;
    int surface_vertex_triangle_count = 0;
    int* self_bucket_counts = nullptr;
    int* self_bucket_indices = nullptr;
    int* self_cell_coords = nullptr;
    int self_cell_capacity = 0;
    int self_hash_bucket_count = 0;
    int self_hash_ready = 0;
    float self_hash_cell_size = 0.0f;
    int* self_triangle_bucket_counts = nullptr;
    int* self_triangle_bucket_indices = nullptr;
    int* self_triangle_cell_coords = nullptr;
    int self_triangle_cell_capacity = 0;
    int self_triangle_hash_bucket_count = 0;
    int self_triangle_hash_ready = 0;
    float self_triangle_hash_cell_size = 0.0f;
    int* self_edge_bucket_counts = nullptr;
    int* self_edge_bucket_indices = nullptr;
    int* self_edge_cell_coords = nullptr;
    int self_edge_cell_capacity = 0;
    int self_edge_hash_bucket_count = 0;
    int self_edge_hash_ready = 0;
    float self_edge_hash_cell_size = 0.0f;
    unsigned int* self_collision_counts = nullptr;
    unsigned int* self_collision_indices = nullptr;
    float* self_collision_radii = nullptr;
    Vec3* self_accumulated_delta = nullptr;
    float* self_accumulated_weight = nullptr;
    Vec3* self_averaged_delta = nullptr;
    float* self_max_smoothed_delta_device = nullptr;
    Vec3* static_triangles = nullptr;
    int static_triangle_count = 0;
    int static_triangle_capacity = 0;
    float* static_sdf = nullptr;
    int static_sdf_capacity = 0;
    int static_sdf_ready = 0;
    int static_sdf_dim_x = 0;
    int static_sdf_dim_y = 0;
    int static_sdf_dim_z = 0;
    Vec3 static_sdf_origin{};
    float static_sdf_voxel_size = 0.0f;
    int static_sdf_signed = 0;
    long long static_sdf_rebuild_count = 0;
    long long static_sdf_unsigned_fallback_count = 0;
    float static_sdf_build_ms = 0.0f;
    Vec3* dynamic_triangles = nullptr;
    int* dynamic_triangle_bucket_counts = nullptr;
    int* dynamic_triangle_bucket_indices = nullptr;
    int* dynamic_triangle_cell_coords = nullptr;
    int dynamic_triangle_count = 0;
    int dynamic_triangle_capacity = 0;
    int dynamic_triangle_cell_capacity = 0;
    int dynamic_triangle_hash_bucket_count = 0;
    int dynamic_triangle_hash_ready = 0;
    float dynamic_triangle_hash_cell_size = 0.0f;
    Vec3* dynamic_particle_positions = nullptr;
    float* dynamic_particle_radii = nullptr;
    int* dynamic_particle_bucket_counts = nullptr;
    int* dynamic_particle_bucket_indices = nullptr;
    int* dynamic_particle_cell_coords = nullptr;
    int dynamic_particle_count = 0;
    int dynamic_particle_capacity = 0;
    int dynamic_particle_cell_capacity = 0;
    int dynamic_particle_hash_bucket_count = 0;
    int dynamic_particle_hash_ready = 0;
    float dynamic_particle_hash_cell_size = 0.0f;
    TriangleProximityPair* triangle_pairs = nullptr;
    int* triangle_pair_count = nullptr;
    int triangle_pair_capacity = 0;
    SsblXpbdForceField* force_fields = nullptr;
    int force_field_capacity = 0;
    int force_field_count = 0;
    int unsupported_force_field_count = 0;
    int* pin_indices = nullptr;
    Vec3* pin_targets = nullptr;
    int pin_count = 0;
    int pin_capacity = 0;
    unsigned int* pcg_row_offsets = nullptr;
    unsigned int* pcg_col_indices = nullptr;
    int* pcg_edge_entry_ij = nullptr;
    int* pcg_edge_entry_ji = nullptr;
    ReconSymMat* pcg_diag_values = nullptr;
    ReconSymMat* pcg_preconditioner_inv = nullptr;
    float4* pcg_offdiag_texels = nullptr;
    Vec3* pcg_rhs = nullptr;
    Vec3* pcg_solution = nullptr;
    Vec3* pcg_residual = nullptr;
    Vec3* pcg_z = nullptr;
    Vec3* pcg_search_dir = nullptr;
    Vec3* pcg_adir = nullptr;
    float* pcg_reductions = nullptr;
    float* pcg_max_delta_device = nullptr;
    unsigned long long* pcg_guard_count = nullptr;
    cudaTextureObject_t pcg_offdiag_texture = 0;
    int pcg_csr_nnz = 0;
    int pcg_texture_ready = 0;
    SsblXpbdRuntimeColliders runtime_colliders{};
    unsigned long long* abi41_counts = nullptr;
    SsblXpbdDiagnostics diag{};
};

bool set_error(const char* message) {
    g_last_error = message ? message : "unknown ABI39 recon CUDA error";
    return false;
}

bool set_cuda_error(cudaError_t err, const char* prefix) {
    if (err == cudaSuccess) {
        return true;
    }
    g_last_error = std::string(prefix) + ": " + cudaGetErrorString(err);
    return false;
}

int block_count(int count) {
    return (std::max(count, 0) + kThreads - 1) / kThreads;
}

float elapsed_ms_since(const std::chrono::high_resolution_clock::time_point& start) {
    const auto now = std::chrono::high_resolution_clock::now();
    return static_cast<float>(std::chrono::duration<double, std::milli>(now - start).count());
}

__host__ __device__ Vec3 make_vec3(float x, float y, float z) {
    Vec3 v{x, y, z};
    return v;
}

__host__ __device__ Vec3 add(Vec3 a, Vec3 b) {
    return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ Vec3 sub(Vec3 a, Vec3 b) {
    return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ Vec3 mul(Vec3 a, float s) {
    return make_vec3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ float fma_rn(float a, float b, float c) {
#if defined(__CUDA_ARCH__)
    return __fmaf_rn(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

__host__ __device__ Vec3 fma_vec(Vec3 a, float s, Vec3 b) {
    return make_vec3(
        fma_rn(a.x, s, b.x),
        fma_rn(a.y, s, b.y),
        fma_rn(a.z, s, b.z)
    );
}

__host__ __device__ float dot(Vec3 a, Vec3 b) {
    return fma_rn(a.z, b.z, fma_rn(a.y, b.y, fma_rn(a.x, b.x, 0.0f)));
}

__host__ __device__ Vec3 cross(Vec3 a, Vec3 b) {
    return make_vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__host__ __device__ float clamp01(float value);

__host__ __device__ int cell_coord(float value, float cell_size) {
    return static_cast<int>(floorf(value / fmaxf(cell_size, kEps)));
}

__host__ __device__ unsigned int hash_cell(int x, int y, int z, int bucket_count) {
    const unsigned int ux = static_cast<unsigned int>(x);
    const unsigned int uy = static_cast<unsigned int>(y);
    const unsigned int uz = static_cast<unsigned int>(z);
    const unsigned int hash = ux * 73856093u ^ uy * 19349663u ^ uz * 83492791u;
    return bucket_count > 0 ? hash & static_cast<unsigned int>(bucket_count - 1) : 0u;
}

__host__ __device__ int clamp_int_value(int value, int lo, int hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

__device__ void atomic_add(Vec3* dst, Vec3 value) {
    atomicAdd(&dst->x, value.x);
    atomicAdd(&dst->y, value.y);
    atomicAdd(&dst->z, value.z);
}

__device__ float length(Vec3 value) {
    return sqrtf(fmaxf(dot(value, value), 0.0f));
}

__device__ bool finite_vec(Vec3 value) {
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

__host__ __device__ ReconSymMat make_sym_mat(
    float m11,
    float m12,
    float m13,
    float m22,
    float m23,
    float m33
) {
    ReconSymMat mat{m11, m12, m13, m22, m23, m33};
    return mat;
}

__host__ __device__ ReconSymMat zero_sym_mat() {
    return make_sym_mat(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
}

__host__ __device__ ReconSymMat identity_sym_mat(float scale) {
    return make_sym_mat(scale, 0.0f, 0.0f, scale, 0.0f, scale);
}

__host__ __device__ Vec3 sym_madd(ReconSymMat mat, Vec3 value, Vec3 acc) {
    return make_vec3(
        fma_rn(mat.m13, value.z, fma_rn(mat.m12, value.y, fma_rn(mat.m11, value.x, acc.x))),
        fma_rn(mat.m23, value.z, fma_rn(mat.m22, value.y, fma_rn(mat.m12, value.x, acc.y))),
        fma_rn(mat.m33, value.z, fma_rn(mat.m23, value.y, fma_rn(mat.m13, value.x, acc.z)))
    );
}

__host__ __device__ Vec3 sym_mul(ReconSymMat mat, Vec3 value) {
    return sym_madd(mat, value, make_vec3(0.0f, 0.0f, 0.0f));
}

__host__ __device__ ReconSymMat sym_outer(Vec3 normal, float scale) {
    return make_sym_mat(
        normal.x * normal.x * scale,
        normal.x * normal.y * scale,
        normal.x * normal.z * scale,
        normal.y * normal.y * scale,
        normal.y * normal.z * scale,
        normal.z * normal.z * scale
    );
}

__device__ void atomic_add_sym(ReconSymMat* dst, ReconSymMat value) {
    atomicAdd(&dst->m11, value.m11);
    atomicAdd(&dst->m12, value.m12);
    atomicAdd(&dst->m13, value.m13);
    atomicAdd(&dst->m22, value.m22);
    atomicAdd(&dst->m23, value.m23);
    atomicAdd(&dst->m33, value.m33);
}

__device__ ReconSymMat sym_inverse_or_diag(ReconSymMat mat, unsigned long long* guard_count) {
    const float a = mat.m11;
    const float b = mat.m12;
    const float c = mat.m13;
    const float d = mat.m22;
    const float e = mat.m23;
    const float f = mat.m33;
    const float df_ee = fma_rn(d, f, -e * e);
    const float bf_ce = fma_rn(b, f, -c * e);
    const float be_cd = fma_rn(b, e, -c * d);
    const float det = fma_rn(c, be_cd, fma_rn(-b, bf_ce, a * df_ee));
    if (isfinite(det) && fabsf(det) > 1.0e-12f) {
        const float inv_det = 1.0f / det;
        return make_sym_mat(
            df_ee * inv_det,
            fma_rn(c, e, -b * f) * inv_det,
            be_cd * inv_det,
            fma_rn(a, f, -c * c) * inv_det,
            fma_rn(b, c, -a * e) * inv_det,
            fma_rn(a, d, -b * b) * inv_det
        );
    }
    if (guard_count) {
        atomicAdd(guard_count, 1ull);
    }
    return make_sym_mat(
        1.0f / fmaxf(fabsf(a), 1.0e-4f),
        0.0f,
        0.0f,
        1.0f / fmaxf(fabsf(d), 1.0e-4f),
        0.0f,
        1.0f / fmaxf(fabsf(f), 1.0e-4f)
    );
}

__device__ void write_sym_texels(float4* texels, int entry, ReconSymMat value) {
    if (!texels || entry < 0) {
        return;
    }
    texels[entry * 2 + 0] = make_float4(value.m11, value.m12, value.m13, value.m22);
    texels[entry * 2 + 1] = make_float4(value.m23, value.m33, 0.0f, 0.0f);
}

__device__ Vec3 array_vec3(const float values[3]) {
    return make_vec3(values[0], values[1], values[2]);
}

__device__ Vec3 normalize_or(Vec3 value, Vec3 fallback) {
    float len = length(value);
    if (!isfinite(len) || len <= kEps) {
        return fallback;
    }
    return mul(value, 1.0f / len);
}

__device__ int surface_normal_at_vertex(Abi41Solver solver, int vertex, Vec3* out_normal) {
    if (!out_normal
        || vertex < 0
        || vertex >= solver.cfg.vertex_count
        || !solver.surface_vertex_offsets
        || !solver.surface_vertex_triangles
        || !solver.triangles
        || !solver.pos) {
        return 0;
    }
    int start = solver.surface_vertex_offsets[vertex];
    int end = solver.surface_vertex_offsets[vertex + 1];
    if (end <= start) {
        return 0;
    }

    Vec3 accumulated = make_vec3(0.0f, 0.0f, 0.0f);
    for (int cursor = start; cursor < end; ++cursor) {
        int t = solver.surface_vertex_triangles[cursor];
        if (t < 0 || t >= solver.cfg.triangle_count) {
            continue;
        }
        ReconTriangle tri = solver.triangles[t];
        int i0 = static_cast<int>(tri.v0);
        int i1 = static_cast<int>(tri.v1);
        int i2 = static_cast<int>(tri.v2);
        if (i0 < 0 || i0 >= solver.cfg.vertex_count
            || i1 < 0 || i1 >= solver.cfg.vertex_count
            || i2 < 0 || i2 >= solver.cfg.vertex_count) {
            continue;
        }
        Vec3 p0 = solver.pos[i0];
        Vec3 p1 = solver.pos[i1];
        Vec3 p2 = solver.pos[i2];
        Vec3 normal = cross(sub(p1, p0), sub(p2, p0));
        float len_sq = dot(normal, normal);
        if (!isfinite(len_sq) || len_sq <= 1.0e-10f) {
            continue;
        }
        accumulated = add(accumulated, mul(normal, rsqrtf(len_sq)));
    }

    float accum_len_sq = dot(accumulated, accumulated);
    if (!isfinite(accum_len_sq) || accum_len_sq <= 1.0e-10f) {
        return 0;
    }
    *out_normal = mul(accumulated, rsqrtf(accum_len_sq));
    return 1;
}

__device__ float fract01(float value) {
    return value - floorf(value);
}

__device__ float force_field_noise(Vec3 p, int seed, float salt) {
    float value = sinf(
        p.x * 12.9898f
        + p.y * 78.233f
        + p.z * 37.719f
        + (static_cast<float>(seed) + salt) * 19.191f
    ) * 43758.5453f;
    return fract01(value) * 2.0f - 1.0f;
}

__device__ float force_field_falloff(const SsblXpbdForceField& field, float distance, float radial_distance) {
    if (!isfinite(distance)) {
        return 0.0f;
    }
    float min_distance = fmaxf(field.distance_min, 0.0f);
    float max_distance = fmaxf(field.distance_max, 0.0f);
    float radial_min = fmaxf(field.radial_min, 0.0f);
    float radial_max = fmaxf(field.radial_max, 0.0f);
    if (field.use_max_distance && max_distance > 0.0f && distance >= max_distance) {
        return 0.0f;
    }
    if (field.use_radial_max && radial_max > 0.0f && radial_distance >= radial_max) {
        return 0.0f;
    }

    float attenuation = 1.0f;
    if (field.use_max_distance && max_distance > min_distance && distance > min_distance) {
        attenuation *= clamp01((max_distance - distance) / fmaxf(max_distance - min_distance, kEps));
    }
    if (field.use_min_distance && field.falloff_power > 0.0f && distance > min_distance) {
        float reference = fmaxf(min_distance, 1.0e-4f);
        attenuation *= powf(reference / fmaxf(distance, reference), field.falloff_power);
    }
    if (field.use_radial_max && radial_max > radial_min && radial_distance > radial_min) {
        attenuation *= clamp01((radial_max - radial_distance) / fmaxf(radial_max - radial_min, kEps));
    }
    if (field.use_radial_min && field.radial_falloff > 0.0f && radial_distance > radial_min) {
        float reference = fmaxf(radial_min, 1.0e-4f);
        attenuation *= powf(reference / fmaxf(radial_distance, reference), field.radial_falloff);
    }
    return isfinite(attenuation) ? attenuation : 0.0f;
}

__device__ Vec3 limit_force_field_acceleration(Vec3 value) {
    float len = length(value);
    if (!isfinite(len) || len <= kAbi41MaxForceFieldAcceleration) {
        return isfinite(len) ? value : make_vec3(0.0f, 0.0f, 0.0f);
    }
    return mul(value, kAbi41MaxForceFieldAcceleration / fmaxf(len, kEps));
}

__device__ Vec3 evaluate_force_field(
    const SsblXpbdForceField& field,
    Vec3 p,
    Vec3 velocity,
    Vec3 surface_normal,
    int has_surface_normal
) {
    if (!isfinite(field.strength) || field.strength == 0.0f) {
        return make_vec3(0.0f, 0.0f, 0.0f);
    }
    Vec3 origin = array_vec3(field.origin);
    Vec3 delta = sub(p, origin);
    Vec3 axis = normalize_or(array_vec3(field.axis), make_vec3(0.0f, 0.0f, 1.0f));
    Vec3 radial_delta = field.use_2d_force ? sub(delta, mul(axis, dot(delta, axis))) : delta;
    float distance = length(delta);
    float radial_distance = length(radial_delta);
    float strength = field.strength * force_field_falloff(field, distance, radial_distance);
    if (strength == 0.0f || !isfinite(strength)) {
        return make_vec3(0.0f, 0.0f, 0.0f);
    }

    if (field.type == kAbi41ForceFieldWind) {
        if (!has_surface_normal) {
            return make_vec3(0.0f, 0.0f, 0.0f);
        }
        Vec3 wind_dir = normalize_or(array_vec3(field.direction), make_vec3(0.0f, 0.0f, 0.0f));
        float projection = dot(wind_dir, surface_normal);
        if (!isfinite(projection) || projection <= 0.0f) {
            return make_vec3(0.0f, 0.0f, 0.0f);
        }
        float magnitude = strength * projection;
        if (field.noise > 0.0f) {
            magnitude *= fmaxf(0.0f, 1.0f + field.noise * force_field_noise(p, field.seed, 0.0f));
        }
        return limit_force_field_acceleration(mul(wind_dir, magnitude));
    }
    if (field.type == kAbi41ForceFieldForce || field.type == kAbi41ForceFieldCharge) {
        Vec3 source_delta = field.use_2d_force ? radial_delta : delta;
        float source_distance = field.use_2d_force ? radial_distance : distance;
        if (source_distance <= 1.0e-6f) {
            return make_vec3(0.0f, 0.0f, 0.0f);
        }
        float scale = strength / fmaxf(source_distance, kEps);
        if (field.type == kAbi41ForceFieldCharge) {
            scale /= fmaxf(source_distance * source_distance, 0.01f);
        }
        return limit_force_field_acceleration(mul(source_delta, scale));
    }
    if (field.type == kAbi41ForceFieldVortex) {
        Vec3 radial = sub(delta, mul(axis, dot(delta, axis)));
        if (dot(radial, radial) <= 1.0e-10f) {
            return make_vec3(0.0f, 0.0f, 0.0f);
        }
        Vec3 tangent = normalize_or(cross(axis, radial), make_vec3(0.0f, 0.0f, 0.0f));
        return mul(tangent, strength);
    }
    if (field.type == kAbi41ForceFieldHarmonic) {
        float source_distance = field.use_2d_force ? radial_distance : distance;
        Vec3 source_delta = field.use_2d_force ? radial_delta : delta;
        if (source_distance <= 1.0e-6f) {
            return make_vec3(0.0f, 0.0f, 0.0f);
        }
        Vec3 direction = mul(source_delta, 1.0f / fmaxf(source_distance, kEps));
        float rest_length = fmaxf(field.rest_length, 0.0f);
        float spring = -strength * (source_distance - rest_length);
        float damping = -dot(velocity, direction) * fmaxf(field.harmonic_damping, 0.0f) * fabsf(strength);
        return limit_force_field_acceleration(mul(direction, spring + damping));
    }
    if (field.type == kAbi41ForceFieldLennardJ) {
        if (distance <= 1.0e-6f) {
            return make_vec3(0.0f, 0.0f, 0.0f);
        }
        Vec3 direction = mul(delta, 1.0f / fmaxf(distance, kEps));
        float radius = fmaxf(field.rest_length, fmaxf(field.size, 0.1f));
        float ratio = fminf(radius / fmaxf(distance, 1.0e-3f), 6.0f);
        float ratio2 = ratio * ratio;
        float ratio6 = ratio2 * ratio2 * ratio2;
        float magnitude = strength * (ratio6 * ratio6 - ratio6);
        return limit_force_field_acceleration(mul(direction, magnitude));
    }
    if (field.type == kAbi41ForceFieldMagnet) {
        Vec3 magnetic_axis = normalize_or(array_vec3(field.direction), make_vec3(0.0f, 0.0f, 0.0f));
        return limit_force_field_acceleration(mul(cross(velocity, magnetic_axis), strength));
    }
    if (field.type == kAbi41ForceFieldDrag) {
        float speed = length(velocity);
        if (speed <= 1.0e-6f) {
            return make_vec3(0.0f, 0.0f, 0.0f);
        }
        float linear = fmaxf(field.linear_drag, 0.0f);
        float quadratic = fmaxf(field.quadratic_drag, 0.0f);
        if (linear <= 0.0f && quadratic <= 0.0f) {
            linear = fabsf(strength);
        }
        return limit_force_field_acceleration(mul(velocity, -(linear + quadratic * speed)));
    }
    if (field.type == kAbi41ForceFieldTurbulence || field.type == kAbi41ForceFieldTexture) {
        float frequency = fmaxf(fmaxf(field.noise, field.texture_nabla), 0.25f);
        if (field.size > 1.0e-6f) {
            frequency = fmaxf(frequency, 1.0f / field.size);
        }
        Vec3 q = mul(delta, frequency);
        Vec3 noise_vec = make_vec3(
            force_field_noise(q, field.seed, 0.0f),
            force_field_noise(q, field.seed, 7.0f),
            force_field_noise(q, field.seed, 13.0f)
        );
        return limit_force_field_acceleration(mul(noise_vec, strength));
    }
    return make_vec3(0.0f, 0.0f, 0.0f);
}

__device__ Vec3 force_field_acceleration(
    Abi41Solver solver,
    Vec3 p,
    Vec3 velocity,
    Vec3 surface_normal,
    int has_surface_normal
) {
    Vec3 acceleration = make_vec3(0.0f, 0.0f, 0.0f);
    if (!solver.force_fields || solver.force_field_count <= 0) {
        return acceleration;
    }
    int count = solver.force_field_count;
    if (count > kAbi41MaxForceFields) {
        count = kAbi41MaxForceFields;
    }
    for (int index = 0; index < count; ++index) {
        acceleration = add(acceleration, evaluate_force_field(
            solver.force_fields[index],
            p,
            velocity,
            surface_normal,
            has_surface_normal
        ));
    }
    return limit_force_field_acceleration(acceleration);
}

__device__ void atomic_max_float(float* dst, float value) {
    if (!dst || value <= 0.0f) {
        return;
    }
    int* address_as_int = reinterpret_cast<int*>(dst);
    int old = *address_as_int;
    while (value > __int_as_float(old)) {
        const int assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
        if (old == assumed) {
            break;
        }
    }
}

__device__ void abi41_accumulate_self_delta(Abi41Solver solver, int vertex, Vec3 delta, float weight = 1.0f) {
    if (vertex < 0
        || vertex >= solver.cfg.vertex_count
        || !solver.self_accumulated_delta
        || !solver.self_accumulated_weight
        || solver.inv_mass[vertex] <= 0.0f
        || (solver.state_flags[vertex] & ssbl_abi41::kPinnedOrKinematicFlag) != 0u
        || weight <= 0.0f) {
        return;
    }
    atomic_add(&solver.self_accumulated_delta[vertex], delta);
    atomicAdd(&solver.self_accumulated_weight[vertex], weight);
}

__device__ Vec3 closest_point_on_triangle(Vec3 p, Vec3 a, Vec3 b, Vec3 c) {
    Vec3 ab = sub(b, a);
    Vec3 ac = sub(c, a);
    Vec3 ap = sub(p, a);
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        return a;
    }

    Vec3 bp = sub(p, b);
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        return b;
    }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / fmaxf(d1 - d3, kEps);
        return add(a, mul(ab, v));
    }

    Vec3 cp = sub(p, c);
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        return c;
    }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / fmaxf(d2 - d6, kEps);
        return add(a, mul(ac, w));
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / fmaxf((d4 - d3) + (d5 - d6), kEps);
        return add(b, mul(sub(c, b), w));
    }

    float denom = 1.0f / fmaxf(va + vb + vc, kEps);
    float v = vb * denom;
    float w = vc * denom;
    return add(a, add(mul(ab, v), mul(ac, w)));
}

__device__ int static_sdf_index(Abi41Solver solver, int x, int y, int z) {
    return (z * solver.static_sdf_dim_y + y) * solver.static_sdf_dim_x + x;
}

__device__ float lerp_float(float a, float b, float t) {
    return a + (b - a) * t;
}

__device__ bool static_sdf_sample(Abi41Solver solver, Vec3 p, float* out_distance) {
    if (!out_distance
        || !solver.static_sdf
        || solver.static_sdf_ready == 0
        || solver.static_sdf_dim_x < 2
        || solver.static_sdf_dim_y < 2
        || solver.static_sdf_dim_z < 2
        || solver.static_sdf_voxel_size <= 0.0f) {
        return false;
    }
    const float inv_voxel = 1.0f / solver.static_sdf_voxel_size;
    const float gx = (p.x - solver.static_sdf_origin.x) * inv_voxel;
    const float gy = (p.y - solver.static_sdf_origin.y) * inv_voxel;
    const float gz = (p.z - solver.static_sdf_origin.z) * inv_voxel;
    if (gx < 0.0f || gy < 0.0f || gz < 0.0f
        || gx > static_cast<float>(solver.static_sdf_dim_x - 1)
        || gy > static_cast<float>(solver.static_sdf_dim_y - 1)
        || gz > static_cast<float>(solver.static_sdf_dim_z - 1)) {
        return false;
    }
    const int x0 = clamp_int_value(static_cast<int>(floorf(gx)), 0, solver.static_sdf_dim_x - 1);
    const int y0 = clamp_int_value(static_cast<int>(floorf(gy)), 0, solver.static_sdf_dim_y - 1);
    const int z0 = clamp_int_value(static_cast<int>(floorf(gz)), 0, solver.static_sdf_dim_z - 1);
    const int x1 = clamp_int_value(x0 + 1, 0, solver.static_sdf_dim_x - 1);
    const int y1 = clamp_int_value(y0 + 1, 0, solver.static_sdf_dim_y - 1);
    const int z1 = clamp_int_value(z0 + 1, 0, solver.static_sdf_dim_z - 1);
    const float tx = gx - static_cast<float>(x0);
    const float ty = gy - static_cast<float>(y0);
    const float tz = gz - static_cast<float>(z0);

    const float c000 = solver.static_sdf[static_sdf_index(solver, x0, y0, z0)];
    const float c100 = solver.static_sdf[static_sdf_index(solver, x1, y0, z0)];
    const float c010 = solver.static_sdf[static_sdf_index(solver, x0, y1, z0)];
    const float c110 = solver.static_sdf[static_sdf_index(solver, x1, y1, z0)];
    const float c001 = solver.static_sdf[static_sdf_index(solver, x0, y0, z1)];
    const float c101 = solver.static_sdf[static_sdf_index(solver, x1, y0, z1)];
    const float c011 = solver.static_sdf[static_sdf_index(solver, x0, y1, z1)];
    const float c111 = solver.static_sdf[static_sdf_index(solver, x1, y1, z1)];
    const float c00 = lerp_float(c000, c100, tx);
    const float c10 = lerp_float(c010, c110, tx);
    const float c01 = lerp_float(c001, c101, tx);
    const float c11 = lerp_float(c011, c111, tx);
    const float c0 = lerp_float(c00, c10, ty);
    const float c1 = lerp_float(c01, c11, ty);
    *out_distance = lerp_float(c0, c1, tz);
    return isfinite(*out_distance);
}

__device__ float static_sdf_sample_clamped(Abi41Solver solver, Vec3 p) {
    const float inv_voxel = 1.0f / fmaxf(solver.static_sdf_voxel_size, kEps);
    const float max_x = static_cast<float>(solver.static_sdf_dim_x - 1);
    const float max_y = static_cast<float>(solver.static_sdf_dim_y - 1);
    const float max_z = static_cast<float>(solver.static_sdf_dim_z - 1);
    float gx = fminf(fmaxf((p.x - solver.static_sdf_origin.x) * inv_voxel, 0.0f), max_x);
    float gy = fminf(fmaxf((p.y - solver.static_sdf_origin.y) * inv_voxel, 0.0f), max_y);
    float gz = fminf(fmaxf((p.z - solver.static_sdf_origin.z) * inv_voxel, 0.0f), max_z);
    const int x0 = clamp_int_value(static_cast<int>(floorf(gx)), 0, solver.static_sdf_dim_x - 1);
    const int y0 = clamp_int_value(static_cast<int>(floorf(gy)), 0, solver.static_sdf_dim_y - 1);
    const int z0 = clamp_int_value(static_cast<int>(floorf(gz)), 0, solver.static_sdf_dim_z - 1);
    const int x1 = clamp_int_value(x0 + 1, 0, solver.static_sdf_dim_x - 1);
    const int y1 = clamp_int_value(y0 + 1, 0, solver.static_sdf_dim_y - 1);
    const int z1 = clamp_int_value(z0 + 1, 0, solver.static_sdf_dim_z - 1);
    const float tx = gx - static_cast<float>(x0);
    const float ty = gy - static_cast<float>(y0);
    const float tz = gz - static_cast<float>(z0);
    const float c000 = solver.static_sdf[static_sdf_index(solver, x0, y0, z0)];
    const float c100 = solver.static_sdf[static_sdf_index(solver, x1, y0, z0)];
    const float c010 = solver.static_sdf[static_sdf_index(solver, x0, y1, z0)];
    const float c110 = solver.static_sdf[static_sdf_index(solver, x1, y1, z0)];
    const float c001 = solver.static_sdf[static_sdf_index(solver, x0, y0, z1)];
    const float c101 = solver.static_sdf[static_sdf_index(solver, x1, y0, z1)];
    const float c011 = solver.static_sdf[static_sdf_index(solver, x0, y1, z1)];
    const float c111 = solver.static_sdf[static_sdf_index(solver, x1, y1, z1)];
    return lerp_float(
        lerp_float(lerp_float(c000, c100, tx), lerp_float(c010, c110, tx), ty),
        lerp_float(lerp_float(c001, c101, tx), lerp_float(c011, c111, tx), ty),
        tz
    );
}

__device__ bool ray_pos_x_intersects_triangle(Vec3 p, Vec3 a, Vec3 b, Vec3 c) {
    const Vec3 dir = make_vec3(1.0f, 0.0f, 0.0f);
    const Vec3 edge1 = sub(b, a);
    const Vec3 edge2 = sub(c, a);
    const Vec3 h = cross(dir, edge2);
    const float det = dot(edge1, h);
    if (fabsf(det) <= 1.0e-8f) {
        return false;
    }
    const float inv_det = 1.0f / det;
    const Vec3 s = sub(p, a);
    const float u = inv_det * dot(s, h);
    if (u < -1.0e-5f || u > 1.0f + 1.0e-5f) {
        return false;
    }
    const Vec3 q = cross(s, edge1);
    const float v = inv_det * dot(dir, q);
    if (v < -1.0e-5f || u + v > 1.0f + 1.0e-5f) {
        return false;
    }
    const float t = inv_det * dot(edge2, q);
    return t > 1.0e-6f;
}

__device__ Vec3 nearest_static_triangle_normal(Abi41Solver solver, Vec3 p) {
    Vec3 best_normal = make_vec3(0.0f, 0.0f, 1.0f);
    float best_sq = 3.402823466e+38f;
    for (int t = 0; t < solver.static_triangle_count; ++t) {
        const Vec3 a = solver.static_triangles[t * 3 + 0];
        const Vec3 b = solver.static_triangles[t * 3 + 1];
        const Vec3 c = solver.static_triangles[t * 3 + 2];
        const Vec3 q = closest_point_on_triangle(p, a, b, c);
        const Vec3 d = sub(p, q);
        const float dist_sq = dot(d, d);
        if (dist_sq < best_sq) {
            best_sq = dist_sq;
            if (dist_sq > kEps * kEps) {
                best_normal = normalize_or(d, best_normal);
            } else {
                best_normal = normalize_or(cross(sub(b, a), sub(c, a)), best_normal);
            }
        }
    }
    return normalize_or(best_normal, make_vec3(0.0f, 0.0f, 1.0f));
}

__device__ Vec3 static_sdf_normal(Abi41Solver solver, Vec3 p) {
    const float h = fmaxf(solver.static_sdf_voxel_size, 1.0e-5f);
    const float dx = static_sdf_sample_clamped(solver, add(p, make_vec3(h, 0.0f, 0.0f)))
        - static_sdf_sample_clamped(solver, add(p, make_vec3(-h, 0.0f, 0.0f)));
    const float dy = static_sdf_sample_clamped(solver, add(p, make_vec3(0.0f, h, 0.0f)))
        - static_sdf_sample_clamped(solver, add(p, make_vec3(0.0f, -h, 0.0f)));
    const float dz = static_sdf_sample_clamped(solver, add(p, make_vec3(0.0f, 0.0f, h)))
        - static_sdf_sample_clamped(solver, add(p, make_vec3(0.0f, 0.0f, -h)));
    const Vec3 n = make_vec3(dx, dy, dz);
    const float len_sq = dot(n, n);
    if (!isfinite(len_sq) || len_sq <= 1.0e-12f) {
        return nearest_static_triangle_normal(solver, p);
    }
    return mul(n, rsqrtf(len_sq));
}

__global__ void abi41_build_static_sdf_kernel(Abi41Solver solver) {
    const int voxel = blockIdx.x * blockDim.x + threadIdx.x;
    const int voxel_count = solver.static_sdf_dim_x * solver.static_sdf_dim_y * solver.static_sdf_dim_z;
    if (voxel >= voxel_count
        || !solver.static_sdf
        || !solver.static_triangles
        || solver.static_triangle_count <= 0) {
        return;
    }
    const int x = voxel % solver.static_sdf_dim_x;
    const int y = (voxel / solver.static_sdf_dim_x) % solver.static_sdf_dim_y;
    const int z = voxel / (solver.static_sdf_dim_x * solver.static_sdf_dim_y);
    Vec3 p = add(
        solver.static_sdf_origin,
        make_vec3(
            static_cast<float>(x) * solver.static_sdf_voxel_size,
            static_cast<float>(y) * solver.static_sdf_voxel_size,
            static_cast<float>(z) * solver.static_sdf_voxel_size
        )
    );
    Vec3 ray_point = add(
        p,
        make_vec3(0.0f, solver.static_sdf_voxel_size * 1.37e-4f, solver.static_sdf_voxel_size * 2.11e-4f)
    );
    float best_sq = 3.402823466e+38f;
    int parity = 0;
    for (int t = 0; t < solver.static_triangle_count; ++t) {
        const Vec3 a = solver.static_triangles[t * 3 + 0];
        const Vec3 b = solver.static_triangles[t * 3 + 1];
        const Vec3 c = solver.static_triangles[t * 3 + 2];
        const Vec3 q = closest_point_on_triangle(p, a, b, c);
        const Vec3 d = sub(p, q);
        best_sq = fminf(best_sq, dot(d, d));
        if (solver.static_sdf_signed != 0 && ray_pos_x_intersects_triangle(ray_point, a, b, c)) {
            parity ^= 1;
        }
    }
    float distance = sqrtf(fmaxf(best_sq, 0.0f));
    if (solver.static_sdf_signed != 0 && parity != 0) {
        distance = -distance;
    }
    solver.static_sdf[voxel] = distance;
}

__global__ void abi41_static_sdf_collision_kernel(Abi41Solver solver, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count
        || solver.inv_mass[i] <= 0.0f
        || !solver.static_sdf
        || solver.static_sdf_ready == 0
        || (solver.state_flags[i] & ssbl_abi41::kPinnedOrKinematicFlag) != 0u) {
        return;
    }
    const float margin = fmaxf(solver.cfg.collision_margin, 0.0f);
    if (margin <= 0.0f) {
        return;
    }
    Vec3 p = solver.pos[i];
    float distance = 0.0f;
    if (!static_sdf_sample(solver, p, &distance) || distance >= margin) {
        return;
    }
    Vec3 normal = static_sdf_normal(solver, p);
    if (!finite_vec(normal)) {
        return;
    }
    const float penetration = margin - distance;
    if (!isfinite(penetration) || penetration <= 0.0f) {
        return;
    }
    Vec3 corrected = add(p, mul(normal, penetration));
    Vec3 vel = solver.vel[i];
    const float vn = dot(vel, normal);
    const float inward_speed = fmaxf(-vn, 0.0f);
    if (vn < 0.0f) {
        vel = sub(vel, mul(normal, vn));
    }
    Vec3 tangent = sub(vel, mul(normal, dot(vel, normal)));
    const float damping = clamp01(solver.cfg.contact_tangent_damping);
    if (damping > 0.0f) {
        vel = sub(vel, mul(tangent, damping));
    }
    tangent = sub(vel, mul(normal, dot(vel, normal)));
    const float tangent_len = length(tangent);
    const float friction = fmaxf(solver.cfg.contact_friction, 0.0f);
    if (friction > 0.0f && tangent_len > 1.0e-6f) {
        const float normal_proxy = inward_speed + penetration / fmaxf(dt, kEps);
        const float drop = fminf(tangent_len, friction * normal_proxy);
        vel = sub(vel, mul(tangent, drop / fmaxf(tangent_len, kEps)));
    }
    solver.pos[i] = corrected;
    solver.vel[i] = vel;
    solver.prev[i] = sub(corrected, mul(vel, fmaxf(dt, kEps)));
    atomicAdd(&solver.abi41_counts[kAbi41CountStaticSdfContacts], 1ull);
}

__global__ void abi41_integrate_kernel(Abi41Solver solver, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count) {
        return;
    }
    solver.prev[i] = solver.pos[i];
    if (solver.inv_mass[i] <= 0.0f || (solver.state_flags[i] & ssbl_abi41::kPinnedOrKinematicFlag) != 0u) {
        solver.vel[i] = make_vec3(0.0f, 0.0f, 0.0f);
        solver.pos[i] = solver.rest[i];
        return;
    }
    Vec3 surface_normal = make_vec3(0.0f, 0.0f, 0.0f);
    int has_surface_normal = surface_normal_at_vertex(solver, i, &surface_normal);
    Vec3 acceleration = make_vec3(solver.cfg.gravity[0], solver.cfg.gravity[1], solver.cfg.gravity[2]);
    if (solver.cfg.use_volume_pressure && solver.cfg.pressure_strength > 0.0f && has_surface_normal) {
        float pressure_force = fmaxf(solver.cfg.pressure_strength, 0.0f);
        acceleration = add(acceleration, mul(surface_normal, pressure_force * solver.inv_mass[i]));
    }
    acceleration = add(acceleration, force_field_acceleration(
        solver,
        solver.pos[i],
        solver.vel[i],
        surface_normal,
        has_surface_normal
    ));
    Vec3 v = add(solver.vel[i], mul(acceleration, dt));
    v = mul(v, solver.cfg.damping);
    solver.vel[i] = v;
    solver.pos[i] = add(solver.pos[i], mul(v, dt));
}

__global__ void abi41_spring_project_kernel(Abi41Solver solver, float dt) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= solver.cfg.edge_count) {
        return;
    }
    ReconSpring spring = solver.springs[s];
    int i = static_cast<int>(spring.id0);
    int j = static_cast<int>(spring.id1);
    if (i < 0 || j < 0 || i >= solver.cfg.vertex_count || j >= solver.cfg.vertex_count) {
        return;
    }
    float wi = solver.inv_mass[i];
    float wj = solver.inv_mass[j];
    float weight = wi + wj;
    if (weight <= 0.0f) {
        return;
    }
    Vec3 delta = sub(solver.pos[j], solver.pos[i]);
    float len_sq = fmaxf(dot(delta, delta), kEps);
    float len = sqrtf(len_sq);
    float c = len - spring.rest_length;
    float alpha = fmaxf(solver.cfg.stretch_compliance, 0.0f) / fmaxf(dt * dt, kEps);
    float dlambda = -c / (weight + alpha);
    Vec3 corr = mul(delta, kAbi41SpringRelaxation * dlambda / fmaxf(len, kEps));
    float corr_len = sqrtf(fmaxf(dot(corr, corr), 0.0f));
    float max_corr = fmaxf(0.0025f, fminf(fmaxf(spring.rest_length, 0.0f) * 0.10f, 0.025f));
    if (corr_len > max_corr) {
        corr = mul(corr, max_corr / fmaxf(corr_len, kEps));
        atomicAdd(&solver.abi41_counts[kAbi41CountHardFallbacks], 1ull);
    }
    if (wi > 0.0f) {
        atomic_add(&solver.pos[i], mul(corr, -wi));
    }
    if (wj > 0.0f) {
        atomic_add(&solver.pos[j], mul(corr, wj));
    }
}

__global__ void abi41_pcg_reset_vertex_kernel(Abi41Solver solver) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0 && solver.pcg_reductions) {
        solver.pcg_reductions[kAbi41PcgReductionDAD] = 0.0f;
        solver.pcg_reductions[kAbi41PcgReductionRZ] = 0.0f;
        solver.pcg_reductions[kAbi41PcgReductionRZNext] = 0.0f;
    }
    if (i >= solver.cfg.vertex_count) {
        return;
    }
    if (solver.pcg_diag_values) {
        solver.pcg_diag_values[i] = identity_sym_mat(1.0f);
    }
    if (solver.pcg_preconditioner_inv) {
        solver.pcg_preconditioner_inv[i] = identity_sym_mat(1.0f);
    }
    if (solver.pcg_rhs) {
        solver.pcg_rhs[i] = make_vec3(0.0f, 0.0f, 0.0f);
    }
    if (solver.pcg_solution) {
        solver.pcg_solution[i] = make_vec3(0.0f, 0.0f, 0.0f);
    }
    if (solver.pcg_residual) {
        solver.pcg_residual[i] = make_vec3(0.0f, 0.0f, 0.0f);
    }
    if (solver.pcg_z) {
        solver.pcg_z[i] = make_vec3(0.0f, 0.0f, 0.0f);
    }
    if (solver.pcg_search_dir) {
        solver.pcg_search_dir[i] = make_vec3(0.0f, 0.0f, 0.0f);
    }
    if (solver.pcg_adir) {
        solver.pcg_adir[i] = make_vec3(0.0f, 0.0f, 0.0f);
    }
}

__global__ void abi41_pcg_build_stretch_system_kernel(Abi41Solver solver) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (!solver.cfg.stretch_optimization_enabled
        || s >= solver.cfg.edge_count
        || !solver.springs
        || !solver.pcg_rhs
        || !solver.pcg_diag_values
        || !solver.pcg_offdiag_texels
        || !solver.pcg_edge_entry_ij
        || !solver.pcg_edge_entry_ji) {
        return;
    }

    const ReconSpring spring = solver.springs[s];
    const int i = static_cast<int>(spring.id0);
    const int j = static_cast<int>(spring.id1);
    if (i < 0 || j < 0 || i >= solver.cfg.vertex_count || j >= solver.cfg.vertex_count || i == j) {
        return;
    }
    const int entry_ij = solver.pcg_edge_entry_ij[s];
    const int entry_ji = solver.pcg_edge_entry_ji[s];
    if (entry_ij < 0 || entry_ji < 0) {
        return;
    }
    const ReconSymMat zero_block = zero_sym_mat();
    write_sym_texels(solver.pcg_offdiag_texels, entry_ij, zero_block);
    write_sym_texels(solver.pcg_offdiag_texels, entry_ji, zero_block);

    float wi = solver.inv_mass[i];
    float wj = solver.inv_mass[j];
    if (solver.state_flags) {
        if ((solver.state_flags[i] & ssbl_abi41::kPinnedOrKinematicFlag) != 0u) {
            wi = 0.0f;
        }
        if ((solver.state_flags[j] & ssbl_abi41::kPinnedOrKinematicFlag) != 0u) {
            wj = 0.0f;
        }
    }
    const bool active_i = wi > 0.0f;
    const bool active_j = wj > 0.0f;
    const float weight = wi + wj;
    if (!isfinite(weight) || weight <= 0.0f) {
        return;
    }

    const float rest = spring.rest_length;
    if (!isfinite(rest) || rest <= kEps) {
        return;
    }
    const Vec3 pi = solver.pos[i];
    const Vec3 pj = solver.pos[j];
    if (!finite_vec(pi) || !finite_vec(pj)) {
        return;
    }
    const Vec3 delta = sub(pj, pi);
    const float len_sq = dot(delta, delta);
    if (!isfinite(len_sq) || len_sq <= kEps) {
        return;
    }
    const float len = sqrtf(len_sq);
    const float over = len - rest;
    if (!isfinite(over) || over <= 0.0f) {
        return;
    }

    const float strength = clamp01(solver.cfg.stretch_optimization_strength);
    if (strength <= 0.0f) {
        return;
    }
    const float max_edge_delta = fmaxf(
        1.0e-5f,
        fminf(fmaxf(rest, solver.cfg.cloth_thickness) * 0.5f, 0.25f)
    );
    const float projected = fminf(over * strength, max_edge_delta);
    if (!isfinite(projected) || projected <= 0.0f) {
        return;
    }

    const Vec3 normal = mul(delta, 1.0f / len);
    const Vec3 desired = mul(normal, projected / weight);
    if (active_i) {
        atomic_add(&solver.pcg_rhs[i], mul(desired, wi));
    }
    if (active_j) {
        atomic_add(&solver.pcg_rhs[j], mul(desired, -wj));
    }

    const float stiffness = fmaxf(strength, 1.0e-3f);
    const ReconSymMat block = sym_outer(normal, stiffness);
    if (active_i) {
        atomic_add_sym(&solver.pcg_diag_values[i], block);
    }
    if (active_j) {
        atomic_add_sym(&solver.pcg_diag_values[j], block);
    }
    if (active_i && active_j) {
        const ReconSymMat offdiag = make_sym_mat(
            -block.m11,
            -block.m12,
            -block.m13,
            -block.m22,
            -block.m23,
            -block.m33
        );
        write_sym_texels(solver.pcg_offdiag_texels, entry_ij, offdiag);
        write_sym_texels(solver.pcg_offdiag_texels, entry_ji, offdiag);
    }
}

__global__ void abi41_pcg_finalize_preconditioner_kernel(Abi41Solver solver) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0.0f;
    if (i < solver.cfg.vertex_count
        && solver.pcg_diag_values
        && solver.pcg_preconditioner_inv
        && solver.pcg_rhs
        && solver.pcg_residual
        && solver.pcg_z
        && solver.pcg_search_dir
        && solver.pcg_solution) {
        const bool pinned = solver.state_flags
            && ((solver.state_flags[i] & ssbl_abi41::kPinnedOrKinematicFlag) != 0u);
        if (solver.inv_mass[i] <= 0.0f || pinned) {
            solver.pcg_rhs[i] = make_vec3(0.0f, 0.0f, 0.0f);
            solver.pcg_residual[i] = make_vec3(0.0f, 0.0f, 0.0f);
            solver.pcg_z[i] = make_vec3(0.0f, 0.0f, 0.0f);
            solver.pcg_search_dir[i] = make_vec3(0.0f, 0.0f, 0.0f);
            solver.pcg_solution[i] = make_vec3(0.0f, 0.0f, 0.0f);
            solver.pcg_preconditioner_inv[i] = identity_sym_mat(1.0f);
        } else {
            const ReconSymMat inv = sym_inverse_or_diag(solver.pcg_diag_values[i], solver.pcg_guard_count);
            const Vec3 r = solver.pcg_rhs[i];
            const Vec3 z = sym_mul(inv, r);
            solver.pcg_preconditioner_inv[i] = inv;
            solver.pcg_residual[i] = r;
            solver.pcg_z[i] = z;
            solver.pcg_search_dir[i] = z;
            solver.pcg_solution[i] = make_vec3(0.0f, 0.0f, 0.0f);
            local_sum = dot(r, z);
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if ((threadIdx.x & 31) == 0 && solver.pcg_reductions) {
        atomicAdd(&solver.pcg_reductions[kAbi41PcgReductionRZ], local_sum);
    }
}

__global__ void abi41_pcg_compute_ad_kernel(Abi41Solver solver) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0.0f;
    if (row < solver.cfg.vertex_count
        && solver.pcg_adir
        && solver.pcg_search_dir
        && solver.pcg_row_offsets
        && solver.pcg_col_indices
        && solver.pcg_diag_values
        && solver.pcg_texture_ready != 0) {
        const bool pinned = solver.state_flags
            && ((solver.state_flags[row] & ssbl_abi41::kPinnedOrKinematicFlag) != 0u);
        if (solver.inv_mass[row] <= 0.0f || pinned) {
            solver.pcg_adir[row] = make_vec3(0.0f, 0.0f, 0.0f);
        } else {
            Vec3 q = sym_mul(solver.pcg_diag_values[row], solver.pcg_search_dir[row]);
            ReconCSRTextureObject texture;
            texture.tex = solver.pcg_offdiag_texture;
            const unsigned int start = solver.pcg_row_offsets[row];
            const unsigned int end = solver.pcg_row_offsets[row + 1];
            for (unsigned int cursor = start; cursor < end; ++cursor) {
                const unsigned int col = solver.pcg_col_indices[cursor];
                if (col < static_cast<unsigned int>(solver.cfg.vertex_count)) {
                    q = sym_madd(texture.getMatrixBlock(cursor), solver.pcg_search_dir[col], q);
                }
            }
            solver.pcg_adir[row] = q;
            local_sum = dot(solver.pcg_search_dir[row], q);
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if ((threadIdx.x & 31) == 0 && solver.pcg_reductions) {
        atomicAdd(&solver.pcg_reductions[kAbi41PcgReductionDAD], local_sum);
    }
}

__global__ void abi41_pcg_update_solution_residual_z_kernel(Abi41Solver solver, float alpha, int apply_solution_now) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0.0f;
    if (i < solver.cfg.vertex_count
        && solver.pcg_solution
        && solver.pcg_search_dir
        && solver.pcg_residual
        && solver.pcg_adir
        && solver.pcg_z
        && solver.pcg_preconditioner_inv) {
        const bool pinned = solver.state_flags
            && ((solver.state_flags[i] & ssbl_abi41::kPinnedOrKinematicFlag) != 0u);
        if (solver.inv_mass[i] <= 0.0f || pinned) {
            solver.pcg_solution[i] = make_vec3(0.0f, 0.0f, 0.0f);
            solver.pcg_residual[i] = make_vec3(0.0f, 0.0f, 0.0f);
            solver.pcg_z[i] = make_vec3(0.0f, 0.0f, 0.0f);
        } else {
            const Vec3 solution = fma_vec(solver.pcg_search_dir[i], alpha, solver.pcg_solution[i]);
            const Vec3 residual = fma_vec(solver.pcg_adir[i], -alpha, solver.pcg_residual[i]);
            const Vec3 safe_solution = finite_vec(solution) ? solution : make_vec3(0.0f, 0.0f, 0.0f);
            const Vec3 safe_residual = finite_vec(residual) ? residual : make_vec3(0.0f, 0.0f, 0.0f);
            const Vec3 z = sym_mul(solver.pcg_preconditioner_inv[i], safe_residual);
            solver.pcg_solution[i] = safe_solution;
            solver.pcg_residual[i] = safe_residual;
            solver.pcg_z[i] = z;
            local_sum = dot(safe_residual, z);

            if (apply_solution_now != 0) {
                Vec3 delta = safe_solution;
                const float len = length(delta);
                const float max_delta = 0.25f;
                if (len > max_delta) {
                    delta = mul(delta, max_delta / fmaxf(len, kEps));
                }
                solver.pos[i] = add(solver.pos[i], delta);
                if (solver.pcg_max_delta_device) {
                    atomic_max_float(solver.pcg_max_delta_device, length(delta));
                }
            }
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if ((threadIdx.x & 31) == 0 && solver.pcg_reductions) {
        atomicAdd(&solver.pcg_reductions[kAbi41PcgReductionRZNext], local_sum);
    }
}

__global__ void abi41_pcg_update_solution_residual_z_device_alpha_kernel(Abi41Solver solver, int apply_solution_now) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!solver.pcg_reductions) {
        return;
    }

    const float rz_old = solver.pcg_reductions[kAbi41PcgReductionRZ];
    const float dad = solver.pcg_reductions[kAbi41PcgReductionDAD];
    float status = kAbi41PcgStatusOk;
    float alpha = 0.0f;
    if (!isfinite(rz_old) || rz_old < 0.0f) {
        status = kAbi41PcgStatusBadResidual;
    } else if (rz_old <= 1.0e-14f) {
        status = kAbi41PcgStatusZeroResidual;
    } else if (!isfinite(dad) || dad <= 1.0e-14f) {
        status = kAbi41PcgStatusBadDAD;
    } else {
        alpha = rz_old / dad;
        if (!isfinite(alpha)) {
            status = kAbi41PcgStatusBadAlpha;
        }
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        solver.pcg_reductions[kAbi41PcgReductionStatus] = status;
    }
    if (status != kAbi41PcgStatusOk) {
        return;
    }

    float local_sum = 0.0f;
    if (i < solver.cfg.vertex_count
        && solver.pcg_solution
        && solver.pcg_search_dir
        && solver.pcg_residual
        && solver.pcg_adir
        && solver.pcg_z
        && solver.pcg_preconditioner_inv) {
        const bool pinned = solver.state_flags
            && ((solver.state_flags[i] & ssbl_abi41::kPinnedOrKinematicFlag) != 0u);
        if (solver.inv_mass[i] <= 0.0f || pinned) {
            solver.pcg_solution[i] = make_vec3(0.0f, 0.0f, 0.0f);
            solver.pcg_residual[i] = make_vec3(0.0f, 0.0f, 0.0f);
            solver.pcg_z[i] = make_vec3(0.0f, 0.0f, 0.0f);
        } else {
            const Vec3 solution = fma_vec(solver.pcg_search_dir[i], alpha, solver.pcg_solution[i]);
            const Vec3 residual = fma_vec(solver.pcg_adir[i], -alpha, solver.pcg_residual[i]);
            const Vec3 safe_solution = finite_vec(solution) ? solution : make_vec3(0.0f, 0.0f, 0.0f);
            const Vec3 safe_residual = finite_vec(residual) ? residual : make_vec3(0.0f, 0.0f, 0.0f);
            const Vec3 z = sym_mul(solver.pcg_preconditioner_inv[i], safe_residual);
            solver.pcg_solution[i] = safe_solution;
            solver.pcg_residual[i] = safe_residual;
            solver.pcg_z[i] = z;
            local_sum = dot(safe_residual, z);

            if (apply_solution_now != 0) {
                Vec3 delta = safe_solution;
                const float len = length(delta);
                const float max_delta = 0.25f;
                if (len > max_delta) {
                    delta = mul(delta, max_delta / fmaxf(len, kEps));
                }
                solver.pos[i] = add(solver.pos[i], delta);
                if (solver.pcg_max_delta_device) {
                    atomic_max_float(solver.pcg_max_delta_device, length(delta));
                }
            }
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&solver.pcg_reductions[kAbi41PcgReductionRZNext], local_sum);
    }
}

__global__ void abi41_pin_project_kernel(Abi41Solver solver) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= solver.pin_count) {
        return;
    }
    int i = solver.pin_indices[p];
    if (i < 0 || i >= solver.cfg.vertex_count) {
        return;
    }
    solver.pos[i] = solver.pin_targets[p];
    solver.prev[i] = solver.pin_targets[p];
    solver.vel[i] = make_vec3(0.0f, 0.0f, 0.0f);
}

__global__ void abi41_analytic_collision_kernel(Abi41Solver solver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count || solver.inv_mass[i] <= 0.0f) {
        return;
    }
    Vec3 p = solver.pos[i];
    float margin = fmaxf(solver.cfg.collision_margin, 0.0f);
    if (solver.runtime_colliders.use_ground && p.z < solver.runtime_colliders.ground_height + margin) {
        p.z = solver.runtime_colliders.ground_height + margin;
        if (solver.vel[i].z < 0.0f) {
            solver.vel[i].z = 0.0f;
        }
    }
    if (solver.runtime_colliders.use_sphere && solver.runtime_colliders.sphere_radius > 0.0f) {
        Vec3 center = make_vec3(
            solver.runtime_colliders.sphere_center[0],
            solver.runtime_colliders.sphere_center[1],
            solver.runtime_colliders.sphere_center[2]
        );
        Vec3 d = sub(p, center);
        float dist_sq = fmaxf(dot(d, d), kEps);
        float dist = sqrtf(dist_sq);
        float target = solver.runtime_colliders.sphere_radius + margin;
        if (dist < target) {
            Vec3 n = mul(d, 1.0f / fmaxf(dist, kEps));
            p = add(center, mul(n, target));
            float vn = dot(solver.vel[i], n);
            if (vn < 0.0f) {
                solver.vel[i] = sub(solver.vel[i], mul(n, vn));
            }
        }
    }
    if (solver.runtime_colliders.use_wall) {
        Vec3 origin = make_vec3(
            solver.runtime_colliders.wall_origin[0],
            solver.runtime_colliders.wall_origin[1],
            solver.runtime_colliders.wall_origin[2]
        );
        Vec3 normal = make_vec3(
            solver.runtime_colliders.wall_normal[0],
            solver.runtime_colliders.wall_normal[1],
            solver.runtime_colliders.wall_normal[2]
        );
        float n_len = sqrtf(fmaxf(dot(normal, normal), kEps));
        normal = mul(normal, 1.0f / n_len);
        float gap = dot(sub(p, origin), normal);
        if (gap < margin) {
            p = add(p, mul(normal, margin - gap));
            float vn = dot(solver.vel[i], normal);
            if (vn < 0.0f) {
                solver.vel[i] = sub(solver.vel[i], mul(normal, vn));
            }
        }
    }
    solver.pos[i] = p;
}

__global__ void abi41_build_dynamic_triangle_hash_kernel(Abi41Solver solver) {
    const int triangle = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangle >= solver.dynamic_triangle_count
        || !solver.dynamic_triangles
        || !solver.dynamic_triangle_bucket_counts
        || !solver.dynamic_triangle_bucket_indices
        || !solver.dynamic_triangle_cell_coords
        || solver.dynamic_triangle_hash_bucket_count <= 0) {
        return;
    }
    Vec3 a = solver.dynamic_triangles[triangle * 3 + 0];
    Vec3 b = solver.dynamic_triangles[triangle * 3 + 1];
    Vec3 c = solver.dynamic_triangles[triangle * 3 + 2];
    Vec3 center = mul(add(add(a, b), c), 1.0f / 3.0f);
    const int cx = cell_coord(center.x, solver.dynamic_triangle_hash_cell_size);
    const int cy = cell_coord(center.y, solver.dynamic_triangle_hash_cell_size);
    const int cz = cell_coord(center.z, solver.dynamic_triangle_hash_cell_size);
    solver.dynamic_triangle_cell_coords[triangle * 3 + 0] = cx;
    solver.dynamic_triangle_cell_coords[triangle * 3 + 1] = cy;
    solver.dynamic_triangle_cell_coords[triangle * 3 + 2] = cz;

    const float margin = fmaxf(solver.cfg.collision_margin, 0.0f);
    const float min_x = fminf(a.x, fminf(b.x, c.x)) - margin;
    const float min_y = fminf(a.y, fminf(b.y, c.y)) - margin;
    const float min_z = fminf(a.z, fminf(b.z, c.z)) - margin;
    const float max_x = fmaxf(a.x, fmaxf(b.x, c.x)) + margin;
    const float max_y = fmaxf(a.y, fmaxf(b.y, c.y)) + margin;
    const float max_z = fmaxf(a.z, fmaxf(b.z, c.z)) + margin;
    const int min_cx = cell_coord(min_x, solver.dynamic_triangle_hash_cell_size);
    const int min_cy = cell_coord(min_y, solver.dynamic_triangle_hash_cell_size);
    const int min_cz = cell_coord(min_z, solver.dynamic_triangle_hash_cell_size);
    const int max_cx = cell_coord(max_x, solver.dynamic_triangle_hash_cell_size);
    const int max_cy = cell_coord(max_y, solver.dynamic_triangle_hash_cell_size);
    const int max_cz = cell_coord(max_z, solver.dynamic_triangle_hash_cell_size);
    for (int z = min_cz; z <= max_cz; ++z) {
        for (int y = min_cy; y <= max_cy; ++y) {
            for (int x = min_cx; x <= max_cx; ++x) {
                const unsigned int bucket = hash_cell(x, y, z, solver.dynamic_triangle_hash_bucket_count);
                const int slot = atomicAdd(&solver.dynamic_triangle_bucket_counts[bucket], 1);
                if (slot < kAbi41TriangleHashBucketSlots) {
                    solver.dynamic_triangle_bucket_indices[
                        static_cast<int>(bucket) * kAbi41TriangleHashBucketSlots + slot
                    ] = triangle;
                }
            }
        }
    }
}

__device__ void consider_dynamic_triangle_candidate(
    Abi41Solver solver,
    int triangle,
    Vec3 p,
    float margin,
    Vec3* best_delta,
    float* best_penetration,
    int* best_triangle
) {
    if (triangle < 0 || triangle >= solver.dynamic_triangle_count) {
        return;
    }
    Vec3 a = solver.dynamic_triangles[triangle * 3 + 0];
    Vec3 b = solver.dynamic_triangles[triangle * 3 + 1];
    Vec3 c = solver.dynamic_triangles[triangle * 3 + 2];
    Vec3 q = closest_point_on_triangle(p, a, b, c);
    Vec3 d = sub(p, q);
    float dist_sq = dot(d, d);
    float dist = sqrtf(fmaxf(dist_sq, kEps));
    if (dist >= margin) {
        return;
    }
    Vec3 normal = mul(d, 1.0f / fmaxf(dist, kEps));
    if (dist_sq <= kEps * 4.0f) {
        Vec3 tri_n = cross(sub(b, a), sub(c, a));
        float tri_len = sqrtf(fmaxf(dot(tri_n, tri_n), kEps));
        normal = mul(tri_n, 1.0f / tri_len);
    }
    float penetration = margin - dist;
    if (penetration > *best_penetration) {
        *best_penetration = penetration;
        *best_delta = mul(normal, penetration);
        *best_triangle = triangle;
    }
}

__global__ void abi41_build_dynamic_triangle_pairs_kernel(Abi41Solver solver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count
        || solver.inv_mass[i] <= 0.0f
        || !solver.dynamic_triangles
        || !solver.triangle_pairs
        || !solver.triangle_pair_count
        || solver.triangle_pair_capacity <= 0) {
        return;
    }
    float margin = fmaxf(solver.cfg.collision_margin, 0.0f);
    if (margin <= 0.0f) {
        return;
    }
    Vec3 p = solver.pos[i];
    Vec3 best_delta = make_vec3(0.0f, 0.0f, 0.0f);
    float best_penetration = 0.0f;
    int best_triangle = -1;
    if (solver.dynamic_triangle_hash_ready != 0
        && solver.dynamic_triangle_bucket_counts
        && solver.dynamic_triangle_bucket_indices
        && solver.dynamic_triangle_cell_coords
        && solver.dynamic_triangle_hash_bucket_count > 0) {
        const int cx = cell_coord(p.x, solver.dynamic_triangle_hash_cell_size);
        const int cy = cell_coord(p.y, solver.dynamic_triangle_hash_cell_size);
        const int cz = cell_coord(p.z, solver.dynamic_triangle_hash_cell_size);
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int qx = cx + dx;
                    const int qy = cy + dy;
                    const int qz = cz + dz;
                    const unsigned int bucket = hash_cell(qx, qy, qz, solver.dynamic_triangle_hash_bucket_count);
                    const int stored = solver.dynamic_triangle_bucket_counts[bucket];
                    const int limit = stored < kAbi41TriangleHashBucketSlots ? stored : kAbi41TriangleHashBucketSlots;
                    for (int slot = 0; slot < limit; ++slot) {
                        const int triangle = solver.dynamic_triangle_bucket_indices[
                            static_cast<int>(bucket) * kAbi41TriangleHashBucketSlots + slot
                        ];
                        consider_dynamic_triangle_candidate(
                            solver,
                            triangle,
                            p,
                            margin,
                            &best_delta,
                            &best_penetration,
                            &best_triangle
                        );
                    }
                }
            }
        }
    } else {
        for (int t = 0; t < solver.dynamic_triangle_count; ++t) {
            consider_dynamic_triangle_candidate(
                solver,
                t,
                p,
                margin,
                &best_delta,
                &best_penetration,
                &best_triangle
            );
        }
    }
    if (best_penetration > 0.0f) {
        const int pair_index = atomicAdd(solver.triangle_pair_count, 1);
        atomicAdd(&solver.abi41_counts[kAbi41CountTrianglePairs], 1ull);
        if (pair_index < solver.triangle_pair_capacity) {
            TriangleProximityPair pair{};
            pair.vertex = i;
            pair.triangle = best_triangle;
            pair.source = 1;
            pair.delta = best_delta;
            solver.triangle_pairs[pair_index] = pair;
        } else {
            atomicAdd(&solver.abi41_counts[kAbi41CountTrianglePairOverflow], 1ull);
        }
    }
}

__global__ void abi41_resolve_triangle_pairs_kernel(Abi41Solver solver) {
    int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (!solver.triangle_pairs || !solver.triangle_pair_count) {
        return;
    }
    int count = *solver.triangle_pair_count;
    if (pair_index >= count || pair_index >= solver.triangle_pair_capacity) {
        return;
    }
    TriangleProximityPair pair = solver.triangle_pairs[pair_index];
    int i = pair.vertex;
    if (i < 0 || i >= solver.cfg.vertex_count || solver.inv_mass[i] <= 0.0f) {
        return;
    }
    solver.pos[i] = add(solver.pos[i], pair.delta);
    if (pair.source == 1) {
        atomicAdd(&solver.abi41_counts[kAbi41CountExactImpulseContacts], 1ull);
    } else {
        float vn = dot(solver.vel[i], pair.delta);
        if (vn < 0.0f) {
            float len_sq = fmaxf(dot(pair.delta, pair.delta), kEps);
            solver.vel[i] = sub(solver.vel[i], mul(pair.delta, vn / len_sq));
        }
        atomicAdd(&solver.abi41_counts[kAbi41CountHardFallbacks], 1ull);
    }
}

__global__ void abi41_build_dynamic_particle_hash_kernel(Abi41Solver solver) {
    const int particle = blockIdx.x * blockDim.x + threadIdx.x;
    if (particle >= solver.dynamic_particle_count
        || !solver.dynamic_particle_positions
        || !solver.dynamic_particle_bucket_counts
        || !solver.dynamic_particle_bucket_indices
        || !solver.dynamic_particle_cell_coords
        || solver.dynamic_particle_hash_bucket_count <= 0) {
        return;
    }
    const Vec3 p = solver.dynamic_particle_positions[particle];
    const int cx = cell_coord(p.x, solver.dynamic_particle_hash_cell_size);
    const int cy = cell_coord(p.y, solver.dynamic_particle_hash_cell_size);
    const int cz = cell_coord(p.z, solver.dynamic_particle_hash_cell_size);
    solver.dynamic_particle_cell_coords[particle * 3 + 0] = cx;
    solver.dynamic_particle_cell_coords[particle * 3 + 1] = cy;
    solver.dynamic_particle_cell_coords[particle * 3 + 2] = cz;

    const unsigned int bucket = hash_cell(cx, cy, cz, solver.dynamic_particle_hash_bucket_count);
    const int slot = atomicAdd(&solver.dynamic_particle_bucket_counts[bucket], 1);
    if (slot < kAbi41ParticleHashBucketSlots) {
        solver.dynamic_particle_bucket_indices[
            static_cast<int>(bucket) * kAbi41ParticleHashBucketSlots + slot
        ] = particle;
    } else {
        atomicAdd(&solver.abi41_counts[kAbi41CountDynamicParticleOverflow], 1ull);
    }
}

__device__ void consider_dynamic_particle_candidate(
    Abi41Solver solver,
    int particle,
    Vec3 p,
    float margin,
    Vec3* best_delta,
    float* best_penetration
) {
    if (particle < 0 || particle >= solver.dynamic_particle_count) {
        return;
    }
    Vec3 center = solver.dynamic_particle_positions[particle];
    float radius = fmaxf(solver.dynamic_particle_radii[particle], 0.0f) + margin;
    if (radius <= 0.0f) {
        return;
    }
    Vec3 d = sub(p, center);
    float dist_sq = dot(d, d);
    float dist = sqrtf(fmaxf(dist_sq, kEps));
    if (dist >= radius) {
        return;
    }
    atomicAdd(&solver.abi41_counts[kAbi41CountDynamicParticleCandidates], 1ull);
    Vec3 normal = dist > kEps ? mul(d, 1.0f / dist) : make_vec3(0.0f, 0.0f, 1.0f);
    float penetration = radius - dist;
    if (penetration > *best_penetration) {
        *best_penetration = penetration;
        *best_delta = mul(normal, penetration);
    }
}

__global__ void abi41_dynamic_particle_collision_kernel(Abi41Solver solver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count
        || solver.inv_mass[i] <= 0.0f
        || !solver.dynamic_particle_positions
        || !solver.dynamic_particle_radii) {
        return;
    }
    Vec3 p = solver.pos[i];
    float margin = fmaxf(solver.cfg.collision_margin, 0.0f);
    Vec3 best_delta = make_vec3(0.0f, 0.0f, 0.0f);
    float best_penetration = 0.0f;
    if (solver.dynamic_particle_hash_ready != 0
        && solver.dynamic_particle_bucket_counts
        && solver.dynamic_particle_bucket_indices
        && solver.dynamic_particle_cell_coords
        && solver.dynamic_particle_hash_bucket_count > 0) {
        const int cx = cell_coord(p.x, solver.dynamic_particle_hash_cell_size);
        const int cy = cell_coord(p.y, solver.dynamic_particle_hash_cell_size);
        const int cz = cell_coord(p.z, solver.dynamic_particle_hash_cell_size);
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int qx = cx + dx;
                    const int qy = cy + dy;
                    const int qz = cz + dz;
                    const unsigned int bucket = hash_cell(qx, qy, qz, solver.dynamic_particle_hash_bucket_count);
                    const int stored = solver.dynamic_particle_bucket_counts[bucket];
                    const int limit = stored < kAbi41ParticleHashBucketSlots ? stored : kAbi41ParticleHashBucketSlots;
                    for (int slot = 0; slot < limit; ++slot) {
                        const int particle = solver.dynamic_particle_bucket_indices[
                            static_cast<int>(bucket) * kAbi41ParticleHashBucketSlots + slot
                        ];
                        if (solver.dynamic_particle_cell_coords[particle * 3 + 0] != qx
                            || solver.dynamic_particle_cell_coords[particle * 3 + 1] != qy
                            || solver.dynamic_particle_cell_coords[particle * 3 + 2] != qz) {
                            continue;
                        }
                        consider_dynamic_particle_candidate(
                            solver,
                            particle,
                            p,
                            margin,
                            &best_delta,
                            &best_penetration
                        );
                    }
                }
            }
        }
    } else {
        for (int particle = 0; particle < solver.dynamic_particle_count; ++particle) {
            consider_dynamic_particle_candidate(
                solver,
                particle,
                p,
                margin,
                &best_delta,
                &best_penetration
            );
        }
    }
    if (best_penetration > 0.0f) {
        solver.pos[i] = add(p, best_delta);
        atomicAdd(&solver.abi41_counts[kAbi41CountExactImpulseContacts], 1ull);
        atomicAdd(&solver.abi41_counts[kAbi41CountDynamicParticleContacts], 1ull);
    }
}

__host__ __device__ float abi41_self_contact_radius(SsblXpbdConfig cfg) {
    return fmaxf(fmaxf(cfg.cloth_thickness, cfg.collision_margin), 0.0f);
}

__global__ void abi41_build_self_hash_kernel(Abi41Solver solver) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count
        || !solver.pos
        || !solver.self_bucket_counts
        || !solver.self_bucket_indices
        || !solver.self_cell_coords
        || solver.self_hash_bucket_count <= 0) {
        return;
    }
    const Vec3 p = solver.pos[i];
    const int cx = cell_coord(p.x, solver.self_hash_cell_size);
    const int cy = cell_coord(p.y, solver.self_hash_cell_size);
    const int cz = cell_coord(p.z, solver.self_hash_cell_size);
    solver.self_cell_coords[i * 3 + 0] = cx;
    solver.self_cell_coords[i * 3 + 1] = cy;
    solver.self_cell_coords[i * 3 + 2] = cz;

    const unsigned int bucket = hash_cell(cx, cy, cz, solver.self_hash_bucket_count);
    const int slot = atomicAdd(&solver.self_bucket_counts[bucket], 1);
    if (slot < kAbi41SelfHashBucketSlots) {
        solver.self_bucket_indices[static_cast<int>(bucket) * kAbi41SelfHashBucketSlots + slot] = i;
    } else {
        atomicAdd(&solver.abi41_counts[kAbi41CountSelfOverflow], 1ull);
    }
}

__device__ bool abi41_triangle_indices_valid(Abi41Solver solver, ReconTriangle tri) {
    const int a = static_cast<int>(tri.v0);
    const int b = static_cast<int>(tri.v1);
    const int c = static_cast<int>(tri.v2);
    return a >= 0 && b >= 0 && c >= 0
        && a < solver.cfg.vertex_count
        && b < solver.cfg.vertex_count
        && c < solver.cfg.vertex_count
        && a != b && b != c && a != c;
}

__device__ void triangle_barycentric(Vec3 p, Vec3 a, Vec3 b, Vec3 c, float* wa, float* wb, float* wc) {
    Vec3 v0 = sub(b, a);
    Vec3 v1 = sub(c, a);
    Vec3 v2 = sub(p, a);
    const float d00 = dot(v0, v0);
    const float d01 = dot(v0, v1);
    const float d11 = dot(v1, v1);
    const float d20 = dot(v2, v0);
    const float d21 = dot(v2, v1);
    const float denom = d00 * d11 - d01 * d01;
    if (fabsf(denom) <= kEps) {
        *wa = 1.0f / 3.0f;
        *wb = 1.0f / 3.0f;
        *wc = 1.0f / 3.0f;
        return;
    }
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    w = fminf(fmaxf(w, 0.0f), 1.0f - v);
    const float u = fmaxf(1.0f - v - w, 0.0f);
    const float sum = fmaxf(u + v + w, kEps);
    *wa = u / sum;
    *wb = v / sum;
    *wc = w / sum;
}

__device__ bool abi41_rest_vertex_triangle_neighbor(Abi41Solver solver, int vertex, ReconTriangle tri, float target, float onset) {
    const int a = static_cast<int>(tri.v0);
    const int b = static_cast<int>(tri.v1);
    const int c = static_cast<int>(tri.v2);
    if (vertex == a || vertex == b || vertex == c) {
        return true;
    }
    const Vec3 rp = solver.rest[vertex];
    const Vec3 ra = solver.rest[a];
    const Vec3 rb = solver.rest[b];
    const Vec3 rc = solver.rest[c];
    const Vec3 rq = closest_point_on_triangle(rp, ra, rb, rc);
    const Vec3 rd = sub(rp, rq);
    const float rest_skip = fmaxf(onset * 1.25f, target * 2.5f);
    return dot(rd, rd) < rest_skip * rest_skip;
}

__global__ void abi41_build_self_triangle_hash_kernel(Abi41Solver solver) {
    const int triangle_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangle_index >= solver.cfg.triangle_count
        || !solver.triangles
        || !solver.self_triangle_bucket_counts
        || !solver.self_triangle_bucket_indices
        || !solver.self_triangle_cell_coords
        || solver.self_triangle_hash_bucket_count <= 0) {
        return;
    }
    const ReconTriangle tri = solver.triangles[triangle_index];
    if (!abi41_triangle_indices_valid(solver, tri)) {
        return;
    }
    const Vec3 a = solver.pos[static_cast<int>(tri.v0)];
    const Vec3 b = solver.pos[static_cast<int>(tri.v1)];
    const Vec3 c = solver.pos[static_cast<int>(tri.v2)];
    const float margin = abi41_self_contact_radius(solver.cfg) * 1.8f;
    const float min_x = fminf(a.x, fminf(b.x, c.x)) - margin;
    const float min_y = fminf(a.y, fminf(b.y, c.y)) - margin;
    const float min_z = fminf(a.z, fminf(b.z, c.z)) - margin;
    const float max_x = fmaxf(a.x, fmaxf(b.x, c.x)) + margin;
    const float max_y = fmaxf(a.y, fmaxf(b.y, c.y)) + margin;
    const float max_z = fmaxf(a.z, fmaxf(b.z, c.z)) + margin;
    const int min_cx = cell_coord(min_x, solver.self_triangle_hash_cell_size);
    const int min_cy = cell_coord(min_y, solver.self_triangle_hash_cell_size);
    const int min_cz = cell_coord(min_z, solver.self_triangle_hash_cell_size);
    const int max_cx = cell_coord(max_x, solver.self_triangle_hash_cell_size);
    const int max_cy = cell_coord(max_y, solver.self_triangle_hash_cell_size);
    const int max_cz = cell_coord(max_z, solver.self_triangle_hash_cell_size);
    solver.self_triangle_cell_coords[triangle_index * 3 + 0] = min_cx;
    solver.self_triangle_cell_coords[triangle_index * 3 + 1] = min_cy;
    solver.self_triangle_cell_coords[triangle_index * 3 + 2] = min_cz;

    for (int z = min_cz; z <= max_cz; ++z) {
        for (int y = min_cy; y <= max_cy; ++y) {
            for (int x = min_cx; x <= max_cx; ++x) {
                const unsigned int bucket = hash_cell(x, y, z, solver.self_triangle_hash_bucket_count);
                const int slot = atomicAdd(&solver.self_triangle_bucket_counts[bucket], 1);
                if (slot < kAbi41SelfTriangleHashBucketSlots) {
                    solver.self_triangle_bucket_indices[
                        static_cast<int>(bucket) * kAbi41SelfTriangleHashBucketSlots + slot
                    ] = triangle_index;
                } else {
                    atomicAdd(&solver.abi41_counts[kAbi41CountSelfOverflow], 1ull);
                }
            }
        }
    }
}

__device__ bool abi41_self_neighbor_valid(Abi41Solver solver, int source, int other, float target, float onset) {
    if (source < 0
        || other < 0
        || source >= solver.cfg.vertex_count
        || other >= solver.cfg.vertex_count
        || source == other
        || target <= 0.0f
        || onset <= target
        || solver.inv_mass[source] <= 0.0f
        || (solver.state_flags[source] & ssbl_abi41::kPinnedOrKinematicFlag) != 0u) {
        return false;
    }
    const Vec3 rest_delta = sub(solver.rest[source], solver.rest[other]);
    const float rest_skip = fmaxf(onset * 1.25f, target * 2.5f);
    if (dot(rest_delta, rest_delta) < rest_skip * rest_skip) {
        return false;
    }
    const Vec3 current_delta = sub(solver.pos[source], solver.pos[other]);
    const Vec3 previous_delta = sub(solver.prev[source], solver.prev[other]);
    return dot(current_delta, current_delta) < onset * onset
        || dot(previous_delta, previous_delta) < onset * onset;
}

__global__ void abi41_reset_self_accumulation_kernel(Abi41Solver solver) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count) {
        return;
    }
    if (solver.self_collision_counts) {
        solver.self_collision_counts[i] = 0u;
    }
    if (solver.self_collision_radii) {
        solver.self_collision_radii[i] = abi41_self_contact_radius(solver.cfg) * 0.5f;
    }
    if (solver.self_accumulated_delta) {
        solver.self_accumulated_delta[i] = make_vec3(0.0f, 0.0f, 0.0f);
    }
    if (solver.self_accumulated_weight) {
        solver.self_accumulated_weight[i] = 0.0f;
    }
    if (solver.self_averaged_delta) {
        solver.self_averaged_delta[i] = make_vec3(0.0f, 0.0f, 0.0f);
    }
}

__global__ void abi41_build_self_neighbor_table_hash_kernel(Abi41Solver solver) {
    const int source = blockIdx.x * blockDim.x + threadIdx.x;
    if (source >= solver.cfg.vertex_count
        || !solver.self_collision_counts
        || !solver.self_collision_indices
        || !solver.self_bucket_counts
        || !solver.self_bucket_indices
        || !solver.self_cell_coords
        || solver.self_hash_bucket_count <= 0) {
        return;
    }
    const float target = abi41_self_contact_radius(solver.cfg);
    if (target <= 0.0f) {
        solver.self_collision_counts[source] = 0u;
        return;
    }
    const float onset = fmaxf(target * 1.8f, target + kEps);
    const Vec3 p = solver.pos[source];
    const int cx = cell_coord(p.x, solver.self_hash_cell_size);
    const int cy = cell_coord(p.y, solver.self_hash_cell_size);
    const int cz = cell_coord(p.z, solver.self_hash_cell_size);
    unsigned int accepted = 0u;
    for (int dz = -1; dz <= 1 && accepted < kAbi41SelfCollisionNeighborSlots; ++dz) {
        for (int dy = -1; dy <= 1 && accepted < kAbi41SelfCollisionNeighborSlots; ++dy) {
            for (int dx = -1; dx <= 1 && accepted < kAbi41SelfCollisionNeighborSlots; ++dx) {
                const int qx = cx + dx;
                const int qy = cy + dy;
                const int qz = cz + dz;
                const unsigned int bucket = hash_cell(qx, qy, qz, solver.self_hash_bucket_count);
                const int stored = solver.self_bucket_counts[bucket];
                const int limit = stored < kAbi41SelfHashBucketSlots ? stored : kAbi41SelfHashBucketSlots;
                for (int slot = 0; slot < limit && accepted < kAbi41SelfCollisionNeighborSlots; ++slot) {
                    const int other = solver.self_bucket_indices[static_cast<int>(bucket) * kAbi41SelfHashBucketSlots + slot];
                    if (solver.self_cell_coords[other * 3 + 0] != qx
                        || solver.self_cell_coords[other * 3 + 1] != qy
                        || solver.self_cell_coords[other * 3 + 2] != qz
                        || !abi41_self_neighbor_valid(solver, source, other, target, onset)) {
                        continue;
                    }
                    solver.self_collision_indices[source * kAbi41SelfCollisionNeighborSlots + static_cast<int>(accepted)] =
                        static_cast<unsigned int>(other);
                    ++accepted;
                }
            }
        }
    }
    if (accepted >= kAbi41SelfCollisionNeighborSlots) {
        atomicAdd(&solver.abi41_counts[kAbi41CountSelfOverflow], 1ull);
    }
    solver.self_collision_counts[source] = accepted;
}

__global__ void abi41_build_self_neighbor_table_kernel(Abi41Solver solver) {
    const int source = blockIdx.x * blockDim.x + threadIdx.x;
    if (source >= solver.cfg.vertex_count || !solver.self_collision_counts || !solver.self_collision_indices) {
        return;
    }
    const float target = abi41_self_contact_radius(solver.cfg);
    if (target <= 0.0f) {
        solver.self_collision_counts[source] = 0u;
        return;
    }
    const float onset = fmaxf(target * 1.8f, target + kEps);
    unsigned int accepted = 0u;
    for (int other = 0; other < solver.cfg.vertex_count && accepted < kAbi41SelfCollisionNeighborSlots; ++other) {
        if (!abi41_self_neighbor_valid(solver, source, other, target, onset)) {
            continue;
        }
        solver.self_collision_indices[source * kAbi41SelfCollisionNeighborSlots + static_cast<int>(accepted)] =
            static_cast<unsigned int>(other);
        ++accepted;
    }
    if (accepted >= kAbi41SelfCollisionNeighborSlots) {
        atomicAdd(&solver.abi41_counts[kAbi41CountSelfOverflow], 1ull);
    }
    solver.self_collision_counts[source] = accepted;
}

__global__ void abi41_set_self_collision_repulsion_kernel(Abi41Solver solver) {
    const int source = blockIdx.x * blockDim.x + threadIdx.x;
    if (source >= solver.cfg.vertex_count
        || !solver.self_collision_counts
        || !solver.self_collision_indices
        || !solver.self_collision_radii) {
        return;
    }
    if (solver.inv_mass[source] <= 0.0f
        || (solver.state_flags[source] & ssbl_abi41::kPinnedOrKinematicFlag) != 0u) {
        return;
    }
    const unsigned int contact_count = solver.self_collision_counts[source];
    if (contact_count == 0u) {
        return;
    }

    const Vec3 p_curr = solver.pos[source];
    const Vec3 p_prev = solver.prev[source];
    const float r_self = solver.self_collision_radii[source];
    const unsigned int limit = contact_count < kAbi41SelfCollisionNeighborSlots
        ? contact_count
        : static_cast<unsigned int>(kAbi41SelfCollisionNeighborSlots);
    for (unsigned int n = 0; n < limit; ++n) {
        const int other = static_cast<int>(solver.self_collision_indices[source * kAbi41SelfCollisionNeighborSlots + static_cast<int>(n)]);
        if (other < 0 || other >= solver.cfg.vertex_count || other == source) {
            continue;
        }
        atomicAdd(&solver.abi41_counts[kAbi41CountSelfCandidates], 1ull);
        const Vec3 q_curr = solver.pos[other];
        const Vec3 q_prev = solver.prev[other];
        const float thickness = r_self + solver.self_collision_radii[other];
        if (thickness <= 0.0f) {
            continue;
        }

        Vec3 response = make_vec3(0.0f, 0.0f, 0.0f);
        bool hit = false;
        const Vec3 d_ref = sub(p_prev, q_prev);
        const float c_coeff = dot(d_ref, d_ref) - thickness * thickness;
        const Vec3 disp_self = sub(p_curr, p_prev);
        const Vec3 disp_other = sub(q_curr, q_prev);
        const Vec3 rel_disp = sub(disp_self, disp_other);
        const float a_coeff = dot(rel_disp, rel_disp);
        const float b_coeff = dot(d_ref, rel_disp);

        if (c_coeff >= 0.0f && b_coeff < 0.0f && a_coeff > 1.0e-5f) {
            const float discriminant = b_coeff * b_coeff - a_coeff * c_coeff;
            if (discriminant >= 0.0f) {
                const float t = (-b_coeff - sqrtf(discriminant)) / a_coeff;
                if (t > 0.0f && t <= 1.0f) {
                    const float s = 1.0f - t;
                    const Vec3 p_hit = add(mul(p_prev, s), mul(p_curr, t));
                    const Vec3 q_hit = add(mul(q_prev, s), mul(q_curr, t));
                    Vec3 normal = sub(p_hit, q_hit);
                    const float normal_len_sq = dot(normal, normal);
                    if (normal_len_sq > kEps) {
                        normal = mul(normal, rsqrtf(normal_len_sq));
                        const Vec3 vel_p = sub(p_curr, p_hit);
                        const Vec3 vel_q = sub(q_curr, q_hit);
                        float v_normal = dot(sub(vel_p, vel_q), normal);
                        if (v_normal > 0.0f) {
                            const bool other_pinned = solver.inv_mass[other] <= 0.0f
                                || (solver.state_flags[other] & ssbl_abi41::kPinnedOrKinematicFlag) != 0u;
                            const float scale = other_pinned ? 1.0f : 0.5f;
                            response = mul(normal, -v_normal * scale);
                            hit = true;
                        }
                    }
                }
            }
        }

        if (!hit) {
            Vec3 delta = sub(p_curr, q_curr);
            const float dist_sq = dot(delta, delta);
            if (dist_sq >= thickness * thickness) {
                continue;
            }
            Vec3 normal = make_vec3(0.0f, 0.0f, 1.0f);
            float dist = sqrtf(fmaxf(dist_sq, kEps));
            if (dist_sq > kEps) {
                normal = mul(delta, 1.0f / dist);
            } else {
                const Vec3 rest_delta = sub(solver.rest[source], solver.rest[other]);
                const float rest_len_sq = dot(rest_delta, rest_delta);
                if (rest_len_sq > kEps) {
                    normal = mul(rest_delta, rsqrtf(rest_len_sq));
                }
                dist = 0.0f;
            }
            const bool other_pinned = solver.inv_mass[other] <= 0.0f
                || (solver.state_flags[other] & ssbl_abi41::kPinnedOrKinematicFlag) != 0u;
            const float scale = other_pinned ? 1.0f : 0.5f;
            response = mul(normal, fminf((thickness - dist) * scale, thickness * kAbi41SelfAveragingClampScale));
            hit = true;
        }

        if (hit && dot(response, response) > kEps) {
            const float max_len = fmaxf(thickness * kAbi41SelfAveragingClampScale, 1.0e-4f);
            const float response_len = length(response);
            if (response_len > max_len) {
                response = mul(response, max_len / response_len);
            }
            abi41_accumulate_self_delta(solver, source, response, 1.0f);
            atomicAdd(&solver.abi41_counts[kAbi41CountExactImpulseContacts], 1ull);
        }
    }
}

__global__ void abi41_averaging_position_kernel(Abi41Solver solver) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count || !solver.self_accumulated_delta || !solver.self_accumulated_weight || !solver.self_averaged_delta) {
        return;
    }
    solver.self_averaged_delta[i] = make_vec3(0.0f, 0.0f, 0.0f);
    if (solver.inv_mass[i] <= 0.0f || (solver.state_flags[i] & ssbl_abi41::kPinnedOrKinematicFlag) != 0u) {
        return;
    }
    const float weight = solver.self_accumulated_weight[i];
    if (weight <= 0.0f) {
        return;
    }
    Vec3 averaged = mul(solver.self_accumulated_delta[i], 1.0f / weight);
    const float target = abi41_self_contact_radius(solver.cfg);
    const float max_delta = fmaxf(target * kAbi41SelfAveragingClampScale, 1.0e-4f);
    const float delta_len = length(averaged);
    if (delta_len > max_delta) {
        averaged = mul(averaged, max_delta / delta_len);
    }
    solver.self_averaged_delta[i] = averaged;
    atomic_max_float(solver.self_max_smoothed_delta_device, length(averaged));
}

__global__ void abi41_apply_self_averaged_delta_kernel(Abi41Solver solver) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count || !solver.self_averaged_delta) {
        return;
    }
    if (solver.inv_mass[i] <= 0.0f || (solver.state_flags[i] & ssbl_abi41::kPinnedOrKinematicFlag) != 0u) {
        return;
    }
    solver.pos[i] = add(solver.pos[i], solver.self_averaged_delta[i]);
}

__device__ bool abi41_apply_soft_vertex_triangle_pair(
    Abi41Solver solver,
    int vertex,
    int triangle_index,
    float target,
    float onset
) {
    if (triangle_index < 0 || triangle_index >= solver.cfg.triangle_count || target <= 0.0f || onset <= target) {
        return false;
    }
    const ReconTriangle tri = solver.triangles[triangle_index];
    if (!abi41_triangle_indices_valid(solver, tri)
        || abi41_rest_vertex_triangle_neighbor(solver, vertex, tri, target, onset)) {
        return false;
    }

    const int ia = static_cast<int>(tri.v0);
    const int ib = static_cast<int>(tri.v1);
    const int ic = static_cast<int>(tri.v2);
    const Vec3 p = solver.pos[vertex];
    const Vec3 a = solver.pos[ia];
    const Vec3 b = solver.pos[ib];
    const Vec3 c = solver.pos[ic];
    const Vec3 q = closest_point_on_triangle(p, a, b, c);
    Vec3 d = sub(p, q);
    const float dist_sq = dot(d, d);
    float dist = sqrtf(fmaxf(dist_sq, kEps));
    if (dist >= onset) {
        return false;
    }
    atomicAdd(&solver.abi41_counts[kAbi41CountSelfCandidates], 1ull);

    Vec3 normal = make_vec3(0.0f, 0.0f, 1.0f);
    if (dist_sq > kEps) {
        normal = mul(d, 1.0f / dist);
    } else {
        normal = cross(sub(b, a), sub(c, a));
        const float n_len_sq = dot(normal, normal);
        if (n_len_sq > kEps) {
            normal = mul(normal, rsqrtf(n_len_sq));
        } else {
            const Vec3 rest_delta = sub(solver.rest[vertex], closest_point_on_triangle(
                solver.rest[vertex],
                solver.rest[ia],
                solver.rest[ib],
                solver.rest[ic]
            ));
            const float rest_len_sq = dot(rest_delta, rest_delta);
            if (rest_len_sq > kEps) {
                normal = mul(rest_delta, rsqrtf(rest_len_sq));
            }
        }
        d = mul(normal, dist);
        dist = 0.0f;
    }

    float wa = 1.0f / 3.0f;
    float wb = 1.0f / 3.0f;
    float wc = 1.0f / 3.0f;
    triangle_barycentric(q, a, b, c, &wa, &wb, &wc);

    const float penetration = target - dist;
    float push = 0.0f;
    if (penetration > 0.0f) {
        push = fminf(penetration * 0.35f + target * 0.01f, target * 0.25f);
    } else {
        const float x = 1.0f - fminf(dist / onset, 1.0f);
        push = fminf(x * x * target * 0.06f, target * 0.04f);
    }

    const float wv = solver.inv_mass[vertex];
    const float wta = solver.inv_mass[ia] * wa;
    const float wtb = solver.inv_mass[ib] * wb;
    const float wtc = solver.inv_mass[ic] * wc;
    const float total = wv + wta + wtb + wtc;
    if (push <= 0.0f || total <= 0.0f) {
        return true;
    }

    const Vec3 delta = mul(normal, push / total);
    if (wv > 0.0f) {
        abi41_accumulate_self_delta(solver, vertex, mul(delta, wv), 1.0f);
    }
    if (wta > 0.0f) {
        abi41_accumulate_self_delta(solver, ia, mul(delta, -wta), 1.0f);
    }
    if (wtb > 0.0f) {
        abi41_accumulate_self_delta(solver, ib, mul(delta, -wtb), 1.0f);
    }
    if (wtc > 0.0f) {
        abi41_accumulate_self_delta(solver, ic, mul(delta, -wtc), 1.0f);
    }
    atomicAdd(&solver.abi41_counts[kAbi41CountSoftContacts], 1ull);
    return true;
}

__global__ void abi41_soft_vertex_triangle_repulsion_hash_kernel(Abi41Solver solver) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= solver.cfg.vertex_count
        || !solver.triangles
        || !solver.self_triangle_bucket_counts
        || !solver.self_triangle_bucket_indices
        || !solver.self_triangle_cell_coords
        || solver.self_triangle_hash_bucket_count <= 0) {
        return;
    }
    const float target = abi41_self_contact_radius(solver.cfg);
    if (target <= 0.0f) {
        return;
    }
    const float onset = fmaxf(target * 1.8f, target + kEps);
    int max_neighbors = solver.cfg.max_self_collision_neighbors;
    if (max_neighbors < 1) {
        max_neighbors = 1;
    }
    if (max_neighbors > kAbi41SelfTriangleHashBucketSlots) {
        max_neighbors = kAbi41SelfTriangleHashBucketSlots;
    }

    const Vec3 p = solver.pos[vertex];
    const int cx = cell_coord(p.x, solver.self_triangle_hash_cell_size);
    const int cy = cell_coord(p.y, solver.self_triangle_hash_cell_size);
    const int cz = cell_coord(p.z, solver.self_triangle_hash_cell_size);
    int accepted = 0;
    const unsigned int bucket = hash_cell(cx, cy, cz, solver.self_triangle_hash_bucket_count);
    const int stored = solver.self_triangle_bucket_counts[bucket];
    const int limit = stored < kAbi41SelfTriangleHashBucketSlots ? stored : kAbi41SelfTriangleHashBucketSlots;
    for (int slot = 0; slot < limit && accepted < max_neighbors; ++slot) {
        const int triangle_index = solver.self_triangle_bucket_indices[
            static_cast<int>(bucket) * kAbi41SelfTriangleHashBucketSlots + slot
        ];
        if (abi41_apply_soft_vertex_triangle_pair(solver, vertex, triangle_index, target, onset)) {
            ++accepted;
        }
    }
}

__global__ void abi41_soft_vertex_triangle_repulsion_kernel(Abi41Solver solver) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= solver.cfg.vertex_count || !solver.triangles) {
        return;
    }
    const float target = abi41_self_contact_radius(solver.cfg);
    if (target <= 0.0f) {
        return;
    }
    const float onset = fmaxf(target * 1.8f, target + kEps);
    int max_neighbors = solver.cfg.max_self_collision_neighbors;
    if (max_neighbors < 1) {
        max_neighbors = 1;
    }
    if (max_neighbors > kAbi41SelfTriangleHashBucketSlots) {
        max_neighbors = kAbi41SelfTriangleHashBucketSlots;
    }
    int accepted = 0;
    for (int triangle_index = 0; triangle_index < solver.cfg.triangle_count && accepted < max_neighbors; ++triangle_index) {
        if (abi41_apply_soft_vertex_triangle_pair(solver, vertex, triangle_index, target, onset)) {
            ++accepted;
        }
    }
}

__host__ __device__ float clamp01(float value) {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

__device__ bool abi41_edge_indices_valid(Abi41Solver solver, ReconSpring edge) {
    const int a = static_cast<int>(edge.id0);
    const int b = static_cast<int>(edge.id1);
    return a >= 0 && b >= 0
        && a < solver.cfg.vertex_count
        && b < solver.cfg.vertex_count
        && a != b;
}

__device__ bool abi41_edges_share_vertex(ReconSpring a, ReconSpring b) {
    return a.id0 == b.id0 || a.id0 == b.id1 || a.id1 == b.id0 || a.id1 == b.id1;
}

__device__ void closest_segment_parameters(
    Vec3 p1,
    Vec3 q1,
    Vec3 p2,
    Vec3 q2,
    float* s,
    float* t
) {
    const Vec3 d1 = sub(q1, p1);
    const Vec3 d2 = sub(q2, p2);
    const Vec3 r = sub(p1, p2);
    const float a = dot(d1, d1);
    const float e = dot(d2, d2);
    const float f = dot(d2, r);
    if (a <= kEps && e <= kEps) {
        *s = 0.0f;
        *t = 0.0f;
        return;
    }
    if (a <= kEps) {
        *s = 0.0f;
        *t = clamp01(f / fmaxf(e, kEps));
        return;
    }
    const float c = dot(d1, r);
    if (e <= kEps) {
        *t = 0.0f;
        *s = clamp01(-c / fmaxf(a, kEps));
        return;
    }
    const float b = dot(d1, d2);
    const float denom = a * e - b * b;
    if (fabsf(denom) > kEps) {
        *s = clamp01((b * f - c * e) / denom);
    } else {
        *s = 0.0f;
    }
    float tnom = b * (*s) + f;
    if (tnom < 0.0f) {
        *t = 0.0f;
        *s = clamp01(-c / fmaxf(a, kEps));
    } else if (tnom > e) {
        *t = 1.0f;
        *s = clamp01((b - c) / fmaxf(a, kEps));
    } else {
        *t = tnom / e;
    }
}

__device__ bool abi41_rest_edges_neighbor(Abi41Solver solver, ReconSpring edge_a, ReconSpring edge_b, float target, float onset) {
    if (abi41_edges_share_vertex(edge_a, edge_b)) {
        return true;
    }
    const int a0 = static_cast<int>(edge_a.id0);
    const int a1 = static_cast<int>(edge_a.id1);
    const int b0 = static_cast<int>(edge_b.id0);
    const int b1 = static_cast<int>(edge_b.id1);
    float s = 0.0f;
    float t = 0.0f;
    closest_segment_parameters(
        solver.rest[a0],
        solver.rest[a1],
        solver.rest[b0],
        solver.rest[b1],
        &s,
        &t
    );
    const Vec3 pa = add(solver.rest[a0], mul(sub(solver.rest[a1], solver.rest[a0]), s));
    const Vec3 pb = add(solver.rest[b0], mul(sub(solver.rest[b1], solver.rest[b0]), t));
    const Vec3 d = sub(pa, pb);
    const float rest_skip = fmaxf(onset * 1.25f, target * 2.5f);
    return dot(d, d) < rest_skip * rest_skip;
}

__global__ void abi41_build_self_edge_hash_kernel(Abi41Solver solver) {
    const int edge_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_index >= solver.cfg.edge_count
        || !solver.springs
        || !solver.self_edge_bucket_counts
        || !solver.self_edge_bucket_indices
        || !solver.self_edge_cell_coords
        || solver.self_edge_hash_bucket_count <= 0) {
        return;
    }
    const ReconSpring edge = solver.springs[edge_index];
    if (!abi41_edge_indices_valid(solver, edge)) {
        return;
    }
    const Vec3 a = solver.pos[static_cast<int>(edge.id0)];
    const Vec3 b = solver.pos[static_cast<int>(edge.id1)];
    const float margin = abi41_self_contact_radius(solver.cfg) * 1.8f;
    const float min_x = fminf(a.x, b.x) - margin;
    const float min_y = fminf(a.y, b.y) - margin;
    const float min_z = fminf(a.z, b.z) - margin;
    const float max_x = fmaxf(a.x, b.x) + margin;
    const float max_y = fmaxf(a.y, b.y) + margin;
    const float max_z = fmaxf(a.z, b.z) + margin;
    const int min_cx = cell_coord(min_x, solver.self_edge_hash_cell_size);
    const int min_cy = cell_coord(min_y, solver.self_edge_hash_cell_size);
    const int min_cz = cell_coord(min_z, solver.self_edge_hash_cell_size);
    const int max_cx = cell_coord(max_x, solver.self_edge_hash_cell_size);
    const int max_cy = cell_coord(max_y, solver.self_edge_hash_cell_size);
    const int max_cz = cell_coord(max_z, solver.self_edge_hash_cell_size);
    solver.self_edge_cell_coords[edge_index * 3 + 0] = min_cx;
    solver.self_edge_cell_coords[edge_index * 3 + 1] = min_cy;
    solver.self_edge_cell_coords[edge_index * 3 + 2] = min_cz;

    for (int z = min_cz; z <= max_cz; ++z) {
        for (int y = min_cy; y <= max_cy; ++y) {
            for (int x = min_cx; x <= max_cx; ++x) {
                const unsigned int bucket = hash_cell(x, y, z, solver.self_edge_hash_bucket_count);
                const int slot = atomicAdd(&solver.self_edge_bucket_counts[bucket], 1);
                if (slot < kAbi41SelfEdgeHashBucketSlots) {
                    solver.self_edge_bucket_indices[
                        static_cast<int>(bucket) * kAbi41SelfEdgeHashBucketSlots + slot
                    ] = edge_index;
                } else {
                    atomicAdd(&solver.abi41_counts[kAbi41CountSelfOverflow], 1ull);
                }
            }
        }
    }
}

__device__ bool abi41_apply_soft_edge_edge_pair(Abi41Solver solver, int edge_a_index, int edge_b_index, float target, float onset) {
    if (edge_b_index <= edge_a_index
        || edge_a_index < 0
        || edge_b_index < 0
        || edge_a_index >= solver.cfg.edge_count
        || edge_b_index >= solver.cfg.edge_count
        || target <= 0.0f
        || onset <= target) {
        return false;
    }
    const ReconSpring edge_a = solver.springs[edge_a_index];
    const ReconSpring edge_b = solver.springs[edge_b_index];
    if (!abi41_edge_indices_valid(solver, edge_a)
        || !abi41_edge_indices_valid(solver, edge_b)
        || abi41_rest_edges_neighbor(solver, edge_a, edge_b, target, onset)) {
        return false;
    }

    const int a0 = static_cast<int>(edge_a.id0);
    const int a1 = static_cast<int>(edge_a.id1);
    const int b0 = static_cast<int>(edge_b.id0);
    const int b1 = static_cast<int>(edge_b.id1);
    float s = 0.0f;
    float t = 0.0f;
    closest_segment_parameters(solver.pos[a0], solver.pos[a1], solver.pos[b0], solver.pos[b1], &s, &t);
    const Vec3 pa = add(solver.pos[a0], mul(sub(solver.pos[a1], solver.pos[a0]), s));
    const Vec3 pb = add(solver.pos[b0], mul(sub(solver.pos[b1], solver.pos[b0]), t));
    Vec3 d = sub(pa, pb);
    const float dist_sq = dot(d, d);
    float dist = sqrtf(fmaxf(dist_sq, kEps));
    if (dist >= onset) {
        return false;
    }
    atomicAdd(&solver.abi41_counts[kAbi41CountSelfCandidates], 1ull);

    Vec3 normal = make_vec3(0.0f, 0.0f, 1.0f);
    if (dist_sq > kEps) {
        normal = mul(d, 1.0f / dist);
    } else {
        normal = cross(sub(solver.pos[a1], solver.pos[a0]), sub(solver.pos[b1], solver.pos[b0]));
        const float normal_len_sq = dot(normal, normal);
        if (normal_len_sq > kEps) {
            normal = mul(normal, rsqrtf(normal_len_sq));
        } else {
            float rs = 0.0f;
            float rt = 0.0f;
            closest_segment_parameters(solver.rest[a0], solver.rest[a1], solver.rest[b0], solver.rest[b1], &rs, &rt);
            const Vec3 rpa = add(solver.rest[a0], mul(sub(solver.rest[a1], solver.rest[a0]), rs));
            const Vec3 rpb = add(solver.rest[b0], mul(sub(solver.rest[b1], solver.rest[b0]), rt));
            const Vec3 rd = sub(rpa, rpb);
            const float rd_len_sq = dot(rd, rd);
            if (rd_len_sq > kEps) {
                normal = mul(rd, rsqrtf(rd_len_sq));
            }
        }
        dist = 0.0f;
    }

    const float penetration = target - dist;
    float push = 0.0f;
    if (penetration > 0.0f) {
        push = fminf(penetration * 0.35f + target * 0.01f, target * 0.25f);
    } else {
        const float x = 1.0f - fminf(dist / onset, 1.0f);
        push = fminf(x * x * target * 0.06f, target * 0.04f);
    }

    const float a0_weight = 1.0f - s;
    const float a1_weight = s;
    const float b0_weight = 1.0f - t;
    const float b1_weight = t;
    const float wa0 = solver.inv_mass[a0] * a0_weight;
    const float wa1 = solver.inv_mass[a1] * a1_weight;
    const float wb0 = solver.inv_mass[b0] * b0_weight;
    const float wb1 = solver.inv_mass[b1] * b1_weight;
    const float total = solver.inv_mass[a0] * a0_weight * a0_weight
        + solver.inv_mass[a1] * a1_weight * a1_weight
        + solver.inv_mass[b0] * b0_weight * b0_weight
        + solver.inv_mass[b1] * b1_weight * b1_weight;
    if (push <= 0.0f || total <= 0.0f) {
        return true;
    }

    const Vec3 delta = mul(normal, push / total);
    if (wa0 > 0.0f) {
        abi41_accumulate_self_delta(solver, a0, mul(delta, wa0), 1.0f);
    }
    if (wa1 > 0.0f) {
        abi41_accumulate_self_delta(solver, a1, mul(delta, wa1), 1.0f);
    }
    if (wb0 > 0.0f) {
        abi41_accumulate_self_delta(solver, b0, mul(delta, -wb0), 1.0f);
    }
    if (wb1 > 0.0f) {
        abi41_accumulate_self_delta(solver, b1, mul(delta, -wb1), 1.0f);
    }
    atomicAdd(&solver.abi41_counts[kAbi41CountEdgeEdgeContacts], 1ull);
    return true;
}

__global__ void abi41_soft_edge_edge_repulsion_hash_kernel(Abi41Solver solver) {
    const int edge_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_index >= solver.cfg.edge_count
        || !solver.springs
        || !solver.self_edge_bucket_counts
        || !solver.self_edge_bucket_indices
        || !solver.self_edge_cell_coords
        || solver.self_edge_hash_bucket_count <= 0) {
        return;
    }
    const ReconSpring edge = solver.springs[edge_index];
    if (!abi41_edge_indices_valid(solver, edge)) {
        return;
    }
    const float target = abi41_self_contact_radius(solver.cfg);
    if (target <= 0.0f) {
        return;
    }
    const float onset = fmaxf(target * 1.8f, target + kEps);
    int max_neighbors = solver.cfg.max_self_collision_neighbors;
    if (max_neighbors < 1) {
        max_neighbors = 1;
    }
    if (max_neighbors > kAbi41SelfEdgeHashBucketSlots) {
        max_neighbors = kAbi41SelfEdgeHashBucketSlots;
    }
    const Vec3 a = solver.pos[static_cast<int>(edge.id0)];
    const Vec3 b = solver.pos[static_cast<int>(edge.id1)];
    const Vec3 mid = mul(add(a, b), 0.5f);
    const int cx = cell_coord(mid.x, solver.self_edge_hash_cell_size);
    const int cy = cell_coord(mid.y, solver.self_edge_hash_cell_size);
    const int cz = cell_coord(mid.z, solver.self_edge_hash_cell_size);
    const unsigned int bucket = hash_cell(cx, cy, cz, solver.self_edge_hash_bucket_count);
    const int stored = solver.self_edge_bucket_counts[bucket];
    const int limit = stored < kAbi41SelfEdgeHashBucketSlots ? stored : kAbi41SelfEdgeHashBucketSlots;
    int accepted = 0;
    for (int slot = 0; slot < limit && accepted < max_neighbors; ++slot) {
        const int other_edge = solver.self_edge_bucket_indices[
            static_cast<int>(bucket) * kAbi41SelfEdgeHashBucketSlots + slot
        ];
        if (abi41_apply_soft_edge_edge_pair(solver, edge_index, other_edge, target, onset)) {
            ++accepted;
        }
    }
}

__global__ void abi41_soft_edge_edge_repulsion_kernel(Abi41Solver solver) {
    const int edge_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_index >= solver.cfg.edge_count || !solver.springs) {
        return;
    }
    const float target = abi41_self_contact_radius(solver.cfg);
    if (target <= 0.0f) {
        return;
    }
    const float onset = fmaxf(target * 1.8f, target + kEps);
    int max_neighbors = solver.cfg.max_self_collision_neighbors;
    if (max_neighbors < 1) {
        max_neighbors = 1;
    }
    if (max_neighbors > kAbi41SelfEdgeHashBucketSlots) {
        max_neighbors = kAbi41SelfEdgeHashBucketSlots;
    }
    int accepted = 0;
    for (int other_edge = edge_index + 1; other_edge < solver.cfg.edge_count && accepted < max_neighbors; ++other_edge) {
        if (abi41_apply_soft_edge_edge_pair(solver, edge_index, other_edge, target, onset)) {
            ++accepted;
        }
    }
}

__global__ void abi41_update_velocity_kernel(Abi41Solver solver, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count) {
        return;
    }
    if (solver.inv_mass[i] <= 0.0f || (solver.state_flags[i] & ssbl_abi41::kPinnedOrKinematicFlag) != 0u) {
        solver.vel[i] = make_vec3(0.0f, 0.0f, 0.0f);
        return;
    }
    solver.vel[i] = mul(sub(solver.pos[i], solver.prev[i]), 1.0f / fmaxf(dt, kEps));
}

template <typename T>
bool alloc_and_copy(T** dst, const T* src, int count, const char* label) {
    *dst = nullptr;
    if (count <= 0) {
        return true;
    }
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(dst), sizeof(T) * count);
    if (!set_cuda_error(err, label)) {
        return false;
    }
    if (src) {
        err = cudaMemcpy(*dst, src, sizeof(T) * count, cudaMemcpyHostToDevice);
        if (!set_cuda_error(err, label)) {
            cudaFree(*dst);
            *dst = nullptr;
            return false;
        }
    }
    return true;
}

bool upload_force_fields(
    Abi41Solver* solver,
    const SsblXpbdForceField* force_fields,
    int force_field_count,
    int unsupported_force_field_count
) {
    if (!solver) {
        return set_error("invalid force field update");
    }
    if (force_field_count < 0 || force_field_count > kAbi41MaxForceFields) {
        return set_error("force field count exceeds SSBL maximum");
    }
    solver->unsupported_force_field_count = std::max(unsupported_force_field_count, 0);
    if (force_field_count <= 0) {
        solver->force_field_count = 0;
        return true;
    }
    if (!force_fields) {
        return set_error("missing force field data");
    }
    if (force_field_count > solver->force_field_capacity) {
        cudaFree(solver->force_fields);
        solver->force_fields = nullptr;
        solver->force_field_capacity = 0;
        cudaError_t err = cudaMalloc(
            reinterpret_cast<void**>(&solver->force_fields),
            sizeof(SsblXpbdForceField) * force_field_count
        );
        if (!set_cuda_error(err, "force field allocation")) {
            return false;
        }
        solver->force_field_capacity = force_field_count;
    }
    if (!set_cuda_error(
            cudaMemcpy(
                solver->force_fields,
                force_fields,
                sizeof(SsblXpbdForceField) * force_field_count,
                cudaMemcpyHostToDevice),
            "upload force fields")) {
        return false;
    }
    solver->force_field_count = force_field_count;
    return true;
}

int next_power_of_two(int value) {
    int result = 1;
    value = std::max(value, 1);
    while (result < value && result < (1 << 30)) {
        result <<= 1;
    }
    return result;
}

struct QuantizedVecKey {
    long long x;
    long long y;
    long long z;

    bool operator==(const QuantizedVecKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct QuantizedVecKeyHash {
    size_t operator()(const QuantizedVecKey& key) const {
        std::uint64_t h = static_cast<std::uint64_t>(key.x) * 73856093ull
            ^ static_cast<std::uint64_t>(key.y) * 19349663ull
            ^ static_cast<std::uint64_t>(key.z) * 83492791ull;
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdull;
        h ^= h >> 33;
        return static_cast<size_t>(h);
    }
};

QuantizedVecKey quantized_vec_key(Vec3 value) {
    constexpr double kScale = 1000000.0;
    return QuantizedVecKey{
        static_cast<long long>(std::llround(static_cast<double>(value.x) * kScale)),
        static_cast<long long>(std::llround(static_cast<double>(value.y) * kScale)),
        static_cast<long long>(std::llround(static_cast<double>(value.z) * kScale))
    };
}

int static_vertex_id(
    std::unordered_map<QuantizedVecKey, int, QuantizedVecKeyHash>* ids,
    Vec3 value
) {
    QuantizedVecKey key = quantized_vec_key(value);
    auto found = ids->find(key);
    if (found != ids->end()) {
        return found->second;
    }
    const int id = static_cast<int>(ids->size());
    ids->emplace(key, id);
    return id;
}

std::uint64_t static_edge_key(int a, int b) {
    const unsigned int lo = static_cast<unsigned int>(std::min(a, b));
    const unsigned int hi = static_cast<unsigned int>(std::max(a, b));
    return (static_cast<std::uint64_t>(lo) << 32) | static_cast<std::uint64_t>(hi);
}

bool static_mesh_is_closed(const float* triangles, int triangle_count) {
    if (!triangles || triangle_count <= 0) {
        return false;
    }
    const Vec3* tri_vecs = reinterpret_cast<const Vec3*>(triangles);
    std::unordered_map<QuantizedVecKey, int, QuantizedVecKeyHash> vertex_ids;
    std::unordered_map<std::uint64_t, int> edge_counts;
    vertex_ids.reserve(static_cast<size_t>(triangle_count) * 2);
    edge_counts.reserve(static_cast<size_t>(triangle_count) * 3);
    for (int t = 0; t < triangle_count; ++t) {
        const Vec3 a = tri_vecs[t * 3 + 0];
        const Vec3 b = tri_vecs[t * 3 + 1];
        const Vec3 c = tri_vecs[t * 3 + 2];
        if (!std::isfinite(a.x) || !std::isfinite(a.y) || !std::isfinite(a.z)
            || !std::isfinite(b.x) || !std::isfinite(b.y) || !std::isfinite(b.z)
            || !std::isfinite(c.x) || !std::isfinite(c.y) || !std::isfinite(c.z)) {
            return false;
        }
        const int ia = static_vertex_id(&vertex_ids, a);
        const int ib = static_vertex_id(&vertex_ids, b);
        const int ic = static_vertex_id(&vertex_ids, c);
        if (ia == ib || ib == ic || ic == ia) {
            continue;
        }
        edge_counts[static_edge_key(ia, ib)] += 1;
        edge_counts[static_edge_key(ib, ic)] += 1;
        edge_counts[static_edge_key(ic, ia)] += 1;
    }
    if (edge_counts.empty()) {
        return false;
    }
    for (const auto& item : edge_counts) {
        if (item.second != 2) {
            return false;
        }
    }
    return true;
}

void reset_static_sdf(Abi41Solver* solver, bool release_buffer) {
    if (!solver) {
        return;
    }
    solver->static_sdf_ready = 0;
    solver->static_sdf_dim_x = 0;
    solver->static_sdf_dim_y = 0;
    solver->static_sdf_dim_z = 0;
    solver->static_sdf_origin = make_vec3(0.0f, 0.0f, 0.0f);
    solver->static_sdf_voxel_size = 0.0f;
    solver->static_sdf_signed = 0;
    solver->static_sdf_build_ms = 0.0f;
    if (release_buffer) {
        cudaFree(solver->static_sdf);
        solver->static_sdf = nullptr;
        solver->static_sdf_capacity = 0;
    }
}

float static_sdf_default_voxel_size(const Abi41Solver* solver) {
    const float requested = solver ? solver->cfg.static_sdf_voxel_size : 0.0f;
    if (std::isfinite(requested) && requested > 0.0f) {
        return requested;
    }
    const float margin = solver ? std::max(solver->cfg.collision_margin, 0.0f) : 0.0f;
    return std::max(margin * 0.5f, 0.002f);
}

int static_sdf_band_voxels(const Abi41Solver* solver) {
    const int requested = solver ? solver->cfg.static_sdf_band_voxels : 0;
    return std::max(requested > 0 ? requested : 4, 1);
}

int static_sdf_max_resolution(const Abi41Solver* solver) {
    const int requested = solver ? solver->cfg.static_sdf_max_resolution : 0;
    return std::max(requested > 0 ? requested : 160, 16);
}

void populate_static_sdf_diagnostics(Abi41Solver* solver) {
    if (!solver) {
        return;
    }
    const long long voxel_count = (
        solver->static_sdf_ready != 0
        ? static_cast<long long>(solver->static_sdf_dim_x)
            * static_cast<long long>(solver->static_sdf_dim_y)
            * static_cast<long long>(solver->static_sdf_dim_z)
        : 0ll
    );
    solver->diag.static_sdf_rebuild_count = solver->static_sdf_rebuild_count;
    solver->diag.static_sdf_voxel_count = voxel_count;
    solver->diag.static_sdf_grid_x = solver->static_sdf_dim_x;
    solver->diag.static_sdf_grid_y = solver->static_sdf_dim_y;
    solver->diag.static_sdf_grid_z = solver->static_sdf_dim_z;
    solver->diag.static_sdf_build_ms = solver->static_sdf_build_ms;
    solver->diag.static_sdf_unsigned_fallback_count = solver->static_sdf_unsigned_fallback_count;
}

bool build_static_sdf(Abi41Solver* solver, const float* triangles, int triangle_count) {
    if (!solver) {
        return set_error("invalid static SDF build");
    }
    reset_static_sdf(solver, false);
    if (triangle_count <= 0) {
        return true;
    }
    if (!triangles) {
        return set_error("static triangle data is required for SDF build");
    }
    const auto started = std::chrono::high_resolution_clock::now();
    const Vec3* tri_vecs = reinterpret_cast<const Vec3*>(triangles);
    Vec3 min_v = make_vec3(
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()
    );
    Vec3 max_v = make_vec3(
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()
    );
    for (int i = 0; i < triangle_count * 3; ++i) {
        const Vec3 v = tri_vecs[i];
        if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z)) {
            return set_error("static triangle data contains non-finite coordinates");
        }
        min_v.x = std::min(min_v.x, v.x);
        min_v.y = std::min(min_v.y, v.y);
        min_v.z = std::min(min_v.z, v.z);
        max_v.x = std::max(max_v.x, v.x);
        max_v.y = std::max(max_v.y, v.y);
        max_v.z = std::max(max_v.z, v.z);
    }

    float voxel_size = static_sdf_default_voxel_size(solver);
    const int band = static_sdf_band_voxels(solver);
    const int max_res = static_sdf_max_resolution(solver);
    const float margin = std::max(solver->cfg.collision_margin, 0.0f);
    const float expand = margin + static_cast<float>(band) * voxel_size + voxel_size * 2.0f;
    min_v = sub(min_v, make_vec3(expand, expand, expand));
    max_v = add(max_v, make_vec3(expand, expand, expand));
    Vec3 extent = sub(max_v, min_v);
    extent.x = std::max(extent.x, voxel_size);
    extent.y = std::max(extent.y, voxel_size);
    extent.z = std::max(extent.z, voxel_size);
    const float largest_extent = std::max(extent.x, std::max(extent.y, extent.z));
    if (largest_extent / voxel_size + 1.0f > static_cast<float>(max_res)) {
        voxel_size = largest_extent / static_cast<float>(std::max(max_res - 1, 1));
    }
    voxel_size = std::max(voxel_size, 1.0e-5f);
    int dim_x = std::max(static_cast<int>(std::ceil(extent.x / voxel_size)) + 1, 2);
    int dim_y = std::max(static_cast<int>(std::ceil(extent.y / voxel_size)) + 1, 2);
    int dim_z = std::max(static_cast<int>(std::ceil(extent.z / voxel_size)) + 1, 2);
    dim_x = std::min(dim_x, max_res);
    dim_y = std::min(dim_y, max_res);
    dim_z = std::min(dim_z, max_res);
    const long long voxel_count_ll = static_cast<long long>(dim_x) * static_cast<long long>(dim_y) * static_cast<long long>(dim_z);
    if (voxel_count_ll <= 0 || voxel_count_ll > static_cast<long long>(std::numeric_limits<int>::max())) {
        return set_error("static SDF grid is too large");
    }
    const int voxel_count = static_cast<int>(voxel_count_ll);
    if (voxel_count > solver->static_sdf_capacity) {
        cudaFree(solver->static_sdf);
        solver->static_sdf = nullptr;
        solver->static_sdf_capacity = 0;
        if (!alloc_and_copy(&solver->static_sdf, static_cast<const float*>(nullptr), voxel_count, "static SDF allocation")) {
            return false;
        }
        solver->static_sdf_capacity = voxel_count;
    }
    solver->static_sdf_dim_x = dim_x;
    solver->static_sdf_dim_y = dim_y;
    solver->static_sdf_dim_z = dim_z;
    solver->static_sdf_origin = min_v;
    solver->static_sdf_voxel_size = voxel_size;
    solver->static_sdf_signed = static_mesh_is_closed(triangles, triangle_count) ? 1 : 0;
    if (solver->static_sdf_signed == 0) {
        solver->static_sdf_unsigned_fallback_count += 1;
    }
    abi41_build_static_sdf_kernel<<<block_count(voxel_count), kThreads>>>(*solver);
    if (!set_cuda_error(cudaGetLastError(), "build static SDF")) {
        reset_static_sdf(solver, false);
        return false;
    }
    if (!set_cuda_error(cudaDeviceSynchronize(), "sync static SDF build")) {
        reset_static_sdf(solver, false);
        return false;
    }
    solver->static_sdf_ready = 1;
    solver->static_sdf_rebuild_count += 1;
    solver->static_sdf_build_ms = elapsed_ms_since(started);
    populate_static_sdf_diagnostics(solver);
    return true;
}

bool prepare_dynamic_triangle_hash_buffers(Abi41Solver* solver, int triangle_count) {
    solver->dynamic_triangle_hash_ready = 0;
    if (triangle_count < kAbi41TriangleHashMinCount) {
        return true;
    }
    if (triangle_count > solver->dynamic_triangle_cell_capacity) {
        cudaFree(solver->dynamic_triangle_cell_coords);
        solver->dynamic_triangle_cell_coords = nullptr;
        solver->dynamic_triangle_cell_capacity = 0;
        if (!alloc_and_copy(&solver->dynamic_triangle_cell_coords, static_cast<const int*>(nullptr), triangle_count * 3, "dynamic triangle cell allocation")) {
            return false;
        }
        solver->dynamic_triangle_cell_capacity = triangle_count;
    } else if (!solver->dynamic_triangle_cell_coords) {
        if (!alloc_and_copy(&solver->dynamic_triangle_cell_coords, static_cast<const int*>(nullptr), triangle_count * 3, "dynamic triangle cell allocation")) {
            return false;
        }
        solver->dynamic_triangle_cell_capacity = triangle_count;
    }

    const int bucket_count = next_power_of_two(std::max(kAbi41MinTriangleHashBuckets, triangle_count * 4));
    if (bucket_count != solver->dynamic_triangle_hash_bucket_count) {
        cudaFree(solver->dynamic_triangle_bucket_counts);
        cudaFree(solver->dynamic_triangle_bucket_indices);
        solver->dynamic_triangle_bucket_counts = nullptr;
        solver->dynamic_triangle_bucket_indices = nullptr;
        solver->dynamic_triangle_hash_bucket_count = 0;
        if (!alloc_and_copy(&solver->dynamic_triangle_bucket_counts, static_cast<const int*>(nullptr), bucket_count, "dynamic triangle hash count allocation")
            || !alloc_and_copy(&solver->dynamic_triangle_bucket_indices, static_cast<const int*>(nullptr), bucket_count * kAbi41TriangleHashBucketSlots, "dynamic triangle hash index allocation")) {
            cudaFree(solver->dynamic_triangle_bucket_counts);
            cudaFree(solver->dynamic_triangle_bucket_indices);
            solver->dynamic_triangle_bucket_counts = nullptr;
            solver->dynamic_triangle_bucket_indices = nullptr;
            return false;
        }
        solver->dynamic_triangle_hash_bucket_count = bucket_count;
    }
    solver->dynamic_triangle_hash_ready = 1;
    return true;
}

bool build_dynamic_triangle_hash(Abi41Solver* solver) {
    if (!solver || solver->dynamic_triangle_hash_ready == 0 || solver->dynamic_triangle_count <= 0) {
        return true;
    }
    if (!set_cuda_error(
        cudaMemset(solver->dynamic_triangle_bucket_counts, 0, sizeof(int) * solver->dynamic_triangle_hash_bucket_count),
        "reset dynamic triangle hash"
    )) {
        return false;
    }
    abi41_build_dynamic_triangle_hash_kernel<<<block_count(solver->dynamic_triangle_count), kThreads>>>(*solver);
    return set_cuda_error(cudaGetLastError(), "build dynamic triangle hash");
}

bool prepare_dynamic_particle_hash_buffers(Abi41Solver* solver, int particle_count) {
    solver->dynamic_particle_hash_ready = 0;
    if (particle_count < kAbi41ParticleHashMinCount) {
        return true;
    }
    if (particle_count > solver->dynamic_particle_cell_capacity) {
        cudaFree(solver->dynamic_particle_cell_coords);
        solver->dynamic_particle_cell_coords = nullptr;
        solver->dynamic_particle_cell_capacity = 0;
        if (!alloc_and_copy(&solver->dynamic_particle_cell_coords, static_cast<const int*>(nullptr), particle_count * 3, "dynamic particle cell allocation")) {
            return false;
        }
        solver->dynamic_particle_cell_capacity = particle_count;
    } else if (!solver->dynamic_particle_cell_coords) {
        if (!alloc_and_copy(&solver->dynamic_particle_cell_coords, static_cast<const int*>(nullptr), particle_count * 3, "dynamic particle cell allocation")) {
            return false;
        }
        solver->dynamic_particle_cell_capacity = particle_count;
    }

    const int bucket_count = next_power_of_two(std::max(kAbi41MinParticleHashBuckets, particle_count * 4));
    if (bucket_count != solver->dynamic_particle_hash_bucket_count) {
        cudaFree(solver->dynamic_particle_bucket_counts);
        cudaFree(solver->dynamic_particle_bucket_indices);
        solver->dynamic_particle_bucket_counts = nullptr;
        solver->dynamic_particle_bucket_indices = nullptr;
        solver->dynamic_particle_hash_bucket_count = 0;
        if (!alloc_and_copy(&solver->dynamic_particle_bucket_counts, static_cast<const int*>(nullptr), bucket_count, "dynamic particle hash count allocation")
            || !alloc_and_copy(&solver->dynamic_particle_bucket_indices, static_cast<const int*>(nullptr), bucket_count * kAbi41ParticleHashBucketSlots, "dynamic particle hash index allocation")) {
            return false;
        }
        solver->dynamic_particle_hash_bucket_count = bucket_count;
    }
    solver->dynamic_particle_hash_ready = 1;
    return true;
}

bool build_dynamic_particle_hash(Abi41Solver* solver) {
    if (!solver || solver->dynamic_particle_hash_ready == 0 || solver->dynamic_particle_count <= 0) {
        return true;
    }
    if (!set_cuda_error(
        cudaMemset(solver->dynamic_particle_bucket_counts, 0, sizeof(int) * solver->dynamic_particle_hash_bucket_count),
        "reset dynamic particle hash"
    )) {
        return false;
    }
    abi41_build_dynamic_particle_hash_kernel<<<block_count(solver->dynamic_particle_count), kThreads>>>(*solver);
    return set_cuda_error(cudaGetLastError(), "build dynamic particle hash");
}

bool prepare_self_collision_hash_buffers(Abi41Solver* solver) {
    if (!solver) {
        return true;
    }
    solver->self_hash_ready = 0;
    if (!solver->cfg.self_collision || solver->cfg.vertex_count < kAbi41SelfHashMinCount) {
        return true;
    }
    const float target = std::max(std::max(solver->cfg.cloth_thickness, solver->cfg.collision_margin), 0.0f);
    if (target <= 0.0f) {
        return true;
    }
    solver->self_hash_cell_size = std::max(target * 1.8f, 1.0e-3f);
    const int vertex_count = solver->cfg.vertex_count;
    if (vertex_count > solver->self_cell_capacity) {
        cudaFree(solver->self_cell_coords);
        solver->self_cell_coords = nullptr;
        solver->self_cell_capacity = 0;
        if (!alloc_and_copy(&solver->self_cell_coords, static_cast<const int*>(nullptr), vertex_count * 3, "self collision cell allocation")) {
            return false;
        }
        solver->self_cell_capacity = vertex_count;
    } else if (!solver->self_cell_coords) {
        if (!alloc_and_copy(&solver->self_cell_coords, static_cast<const int*>(nullptr), vertex_count * 3, "self collision cell allocation")) {
            return false;
        }
        solver->self_cell_capacity = vertex_count;
    }

    const int bucket_count = next_power_of_two(std::max(kAbi41MinSelfHashBuckets, vertex_count * 4));
    if (bucket_count != solver->self_hash_bucket_count) {
        cudaFree(solver->self_bucket_counts);
        cudaFree(solver->self_bucket_indices);
        solver->self_bucket_counts = nullptr;
        solver->self_bucket_indices = nullptr;
        solver->self_hash_bucket_count = 0;
        if (!alloc_and_copy(&solver->self_bucket_counts, static_cast<const int*>(nullptr), bucket_count, "self collision hash count allocation")
            || !alloc_and_copy(&solver->self_bucket_indices, static_cast<const int*>(nullptr), bucket_count * kAbi41SelfHashBucketSlots, "self collision hash index allocation")) {
            cudaFree(solver->self_bucket_counts);
            cudaFree(solver->self_bucket_indices);
            solver->self_bucket_counts = nullptr;
            solver->self_bucket_indices = nullptr;
            return false;
        }
        solver->self_hash_bucket_count = bucket_count;
    }
    solver->self_hash_ready = 1;
    return true;
}

bool build_self_collision_hash(Abi41Solver* solver) {
    if (!solver || solver->self_hash_ready == 0 || solver->cfg.vertex_count <= 0) {
        return true;
    }
    if (!set_cuda_error(
        cudaMemset(solver->self_bucket_counts, 0, sizeof(int) * solver->self_hash_bucket_count),
        "reset self collision hash"
    )) {
        return false;
    }
    abi41_build_self_hash_kernel<<<block_count(solver->cfg.vertex_count), kThreads>>>(*solver);
    return set_cuda_error(cudaGetLastError(), "build self collision hash");
}

bool reset_self_accumulation(Abi41Solver* solver) {
    if (!solver || solver->cfg.vertex_count <= 0 || !solver->self_accumulated_delta || !solver->self_accumulated_weight) {
        return true;
    }
    abi41_reset_self_accumulation_kernel<<<block_count(solver->cfg.vertex_count), kThreads>>>(*solver);
    return set_cuda_error(cudaGetLastError(), "reset self accumulation");
}

bool build_self_neighbor_table(Abi41Solver* solver) {
    if (!solver || solver->cfg.vertex_count <= 0 || !solver->self_collision_counts || !solver->self_collision_indices) {
        return true;
    }
    if (solver->self_hash_ready != 0) {
        abi41_build_self_neighbor_table_hash_kernel<<<block_count(solver->cfg.vertex_count), kThreads>>>(*solver);
    } else {
        abi41_build_self_neighbor_table_kernel<<<block_count(solver->cfg.vertex_count), kThreads>>>(*solver);
    }
    return set_cuda_error(cudaGetLastError(), "build self neighbor table");
}

bool prepare_self_triangle_hash_buffers(Abi41Solver* solver) {
    if (!solver) {
        return true;
    }
    solver->self_triangle_hash_ready = 0;
    if (!solver->cfg.self_collision || solver->cfg.triangle_count < kAbi41SelfTriangleHashMinCount) {
        return true;
    }
    const float target = std::max(std::max(solver->cfg.cloth_thickness, solver->cfg.collision_margin), 0.0f);
    if (target <= 0.0f) {
        return true;
    }
    solver->self_triangle_hash_cell_size = std::max(target * 1.8f, 1.0e-3f);
    const int triangle_count = solver->cfg.triangle_count;
    if (triangle_count > solver->self_triangle_cell_capacity) {
        cudaFree(solver->self_triangle_cell_coords);
        solver->self_triangle_cell_coords = nullptr;
        solver->self_triangle_cell_capacity = 0;
        if (!alloc_and_copy(&solver->self_triangle_cell_coords, static_cast<const int*>(nullptr), triangle_count * 3, "self triangle cell allocation")) {
            return false;
        }
        solver->self_triangle_cell_capacity = triangle_count;
    } else if (!solver->self_triangle_cell_coords) {
        if (!alloc_and_copy(&solver->self_triangle_cell_coords, static_cast<const int*>(nullptr), triangle_count * 3, "self triangle cell allocation")) {
            return false;
        }
        solver->self_triangle_cell_capacity = triangle_count;
    }

    const int bucket_count = next_power_of_two(std::max(kAbi41MinSelfTriangleHashBuckets, triangle_count * 4));
    if (bucket_count != solver->self_triangle_hash_bucket_count) {
        cudaFree(solver->self_triangle_bucket_counts);
        cudaFree(solver->self_triangle_bucket_indices);
        solver->self_triangle_bucket_counts = nullptr;
        solver->self_triangle_bucket_indices = nullptr;
        solver->self_triangle_hash_bucket_count = 0;
        if (!alloc_and_copy(&solver->self_triangle_bucket_counts, static_cast<const int*>(nullptr), bucket_count, "self triangle hash count allocation")
            || !alloc_and_copy(&solver->self_triangle_bucket_indices, static_cast<const int*>(nullptr), bucket_count * kAbi41SelfTriangleHashBucketSlots, "self triangle hash index allocation")) {
            cudaFree(solver->self_triangle_bucket_counts);
            cudaFree(solver->self_triangle_bucket_indices);
            solver->self_triangle_bucket_counts = nullptr;
            solver->self_triangle_bucket_indices = nullptr;
            return false;
        }
        solver->self_triangle_hash_bucket_count = bucket_count;
    }
    solver->self_triangle_hash_ready = 1;
    return true;
}

bool build_self_triangle_hash(Abi41Solver* solver) {
    if (!solver || solver->self_triangle_hash_ready == 0 || solver->cfg.triangle_count <= 0) {
        return true;
    }
    if (!set_cuda_error(
        cudaMemset(solver->self_triangle_bucket_counts, 0, sizeof(int) * solver->self_triangle_hash_bucket_count),
        "reset self triangle hash"
    )) {
        return false;
    }
    abi41_build_self_triangle_hash_kernel<<<block_count(solver->cfg.triangle_count), kThreads>>>(*solver);
    return set_cuda_error(cudaGetLastError(), "build self triangle hash");
}

bool prepare_self_edge_hash_buffers(Abi41Solver* solver) {
    if (!solver) {
        return true;
    }
    solver->self_edge_hash_ready = 0;
    if (!solver->cfg.self_collision || solver->cfg.edge_count < kAbi41SelfEdgeHashMinCount) {
        return true;
    }
    const float target = std::max(std::max(solver->cfg.cloth_thickness, solver->cfg.collision_margin), 0.0f);
    if (target <= 0.0f) {
        return true;
    }
    solver->self_edge_hash_cell_size = std::max(target * 1.8f, 1.0e-3f);
    const int edge_count = solver->cfg.edge_count;
    if (edge_count > solver->self_edge_cell_capacity) {
        cudaFree(solver->self_edge_cell_coords);
        solver->self_edge_cell_coords = nullptr;
        solver->self_edge_cell_capacity = 0;
        if (!alloc_and_copy(&solver->self_edge_cell_coords, static_cast<const int*>(nullptr), edge_count * 3, "self edge cell allocation")) {
            return false;
        }
        solver->self_edge_cell_capacity = edge_count;
    } else if (!solver->self_edge_cell_coords) {
        if (!alloc_and_copy(&solver->self_edge_cell_coords, static_cast<const int*>(nullptr), edge_count * 3, "self edge cell allocation")) {
            return false;
        }
        solver->self_edge_cell_capacity = edge_count;
    }

    const int bucket_count = next_power_of_two(std::max(kAbi41MinSelfEdgeHashBuckets, edge_count * 4));
    if (bucket_count != solver->self_edge_hash_bucket_count) {
        cudaFree(solver->self_edge_bucket_counts);
        cudaFree(solver->self_edge_bucket_indices);
        solver->self_edge_bucket_counts = nullptr;
        solver->self_edge_bucket_indices = nullptr;
        solver->self_edge_hash_bucket_count = 0;
        if (!alloc_and_copy(&solver->self_edge_bucket_counts, static_cast<const int*>(nullptr), bucket_count, "self edge hash count allocation")
            || !alloc_and_copy(&solver->self_edge_bucket_indices, static_cast<const int*>(nullptr), bucket_count * kAbi41SelfEdgeHashBucketSlots, "self edge hash index allocation")) {
            cudaFree(solver->self_edge_bucket_counts);
            cudaFree(solver->self_edge_bucket_indices);
            solver->self_edge_bucket_counts = nullptr;
            solver->self_edge_bucket_indices = nullptr;
            return false;
        }
        solver->self_edge_hash_bucket_count = bucket_count;
    }
    solver->self_edge_hash_ready = 1;
    return true;
}

bool build_self_edge_hash(Abi41Solver* solver) {
    if (!solver || solver->self_edge_hash_ready == 0 || solver->cfg.edge_count <= 0) {
        return true;
    }
    if (!set_cuda_error(
        cudaMemset(solver->self_edge_bucket_counts, 0, sizeof(int) * solver->self_edge_hash_bucket_count),
        "reset self edge hash"
    )) {
        return false;
    }
    abi41_build_self_edge_hash_kernel<<<block_count(solver->cfg.edge_count), kThreads>>>(*solver);
    return set_cuda_error(cudaGetLastError(), "build self edge hash");
}

bool upload_pins(Abi41Solver* solver, const int* indices, const float* positions, int count) {
    if (!solver) {
        return set_error("invalid solver handle");
    }
    if (count < 0) {
        return set_error("invalid pin count");
    }
    if (count == 0) {
        solver->pin_count = 0;
        return true;
    }
    if (!indices || !positions) {
        return set_error("pin arrays are required when pin count is nonzero");
    }
    if (count > solver->pin_capacity) {
        cudaFree(solver->pin_indices);
        cudaFree(solver->pin_targets);
        solver->pin_indices = nullptr;
        solver->pin_targets = nullptr;
        solver->pin_capacity = count;
        if (!alloc_and_copy(&solver->pin_indices, indices, count, "pin index allocation")) {
            solver->pin_capacity = 0;
            return false;
        }
        if (!alloc_and_copy(&solver->pin_targets, reinterpret_cast<const Vec3*>(positions), count, "pin target allocation")) {
            solver->pin_capacity = 0;
            return false;
        }
    } else {
        if (!set_cuda_error(cudaMemcpy(solver->pin_indices, indices, sizeof(int) * count, cudaMemcpyHostToDevice), "pin index upload")) {
            return false;
        }
        if (!set_cuda_error(cudaMemcpy(solver->pin_targets, positions, sizeof(Vec3) * count, cudaMemcpyHostToDevice), "pin target upload")) {
            return false;
        }
    }
    solver->pin_count = count;
    return true;
}

bool build_surface_vertex_triangles(
    Abi41Solver* solver,
    int vertex_count,
    const std::vector<ReconTriangle>& triangles
) {
    if (!solver || vertex_count <= 0 || triangles.empty()) {
        return true;
    }
    std::vector<int> counts(vertex_count, 0);
    for (const ReconTriangle& tri : triangles) {
        const int i0 = static_cast<int>(tri.v0);
        const int i1 = static_cast<int>(tri.v1);
        const int i2 = static_cast<int>(tri.v2);
        if (i0 >= 0 && i0 < vertex_count) {
            ++counts[i0];
        }
        if (i1 >= 0 && i1 < vertex_count) {
            ++counts[i1];
        }
        if (i2 >= 0 && i2 < vertex_count) {
            ++counts[i2];
        }
    }

    std::vector<int> offsets(vertex_count + 1, 0);
    for (int i = 0; i < vertex_count; ++i) {
        offsets[i + 1] = offsets[i] + counts[i];
    }
    std::vector<int> cursor(offsets.begin(), offsets.end());
    std::vector<int> incident(offsets.back(), -1);
    for (int t = 0; t < static_cast<int>(triangles.size()); ++t) {
        const ReconTriangle tri = triangles[t];
        const int i0 = static_cast<int>(tri.v0);
        const int i1 = static_cast<int>(tri.v1);
        const int i2 = static_cast<int>(tri.v2);
        if (i0 >= 0 && i0 < vertex_count) {
            incident[cursor[i0]++] = t;
        }
        if (i1 >= 0 && i1 < vertex_count) {
            incident[cursor[i1]++] = t;
        }
        if (i2 >= 0 && i2 < vertex_count) {
            incident[cursor[i2]++] = t;
        }
    }

    solver->surface_vertex_triangle_count = static_cast<int>(incident.size());
    return alloc_and_copy(
        &solver->surface_vertex_offsets,
        offsets.data(),
        vertex_count + 1,
        "surface vertex offset allocation"
    ) && alloc_and_copy(
        &solver->surface_vertex_triangles,
        incident.data(),
        solver->surface_vertex_triangle_count,
        "surface incident triangle allocation"
    );
}

void destroy_pcg_texture(Abi41Solver* solver) {
    if (solver && solver->pcg_offdiag_texture != 0) {
        cudaDestroyTextureObject(solver->pcg_offdiag_texture);
        solver->pcg_offdiag_texture = 0;
        solver->pcg_texture_ready = 0;
    }
}

bool prepare_pcg_stretch_buffers(
    Abi41Solver* solver,
    const std::vector<ReconSpring>& springs
) {
    if (!solver || !solver->cfg.stretch_optimization_enabled) {
        return true;
    }
    const int vertex_count = solver->cfg.vertex_count;
    const int edge_count = solver->cfg.edge_count;
    if (vertex_count <= 0 || edge_count <= 0) {
        return set_error("PCG stretch optimization requires vertices and edges");
    }

    std::vector<unsigned int> degrees(static_cast<size_t>(vertex_count), 0u);
    std::vector<int> edge_entry_ij(static_cast<size_t>(edge_count), -1);
    std::vector<int> edge_entry_ji(static_cast<size_t>(edge_count), -1);
    for (int e = 0; e < edge_count; ++e) {
        const int i = static_cast<int>(springs[e].id0);
        const int j = static_cast<int>(springs[e].id1);
        if (i >= 0 && i < vertex_count && j >= 0 && j < vertex_count && i != j) {
            ++degrees[static_cast<size_t>(i)];
            ++degrees[static_cast<size_t>(j)];
        }
    }

    std::vector<unsigned int> row_offsets(static_cast<size_t>(vertex_count) + 1u, 0u);
    for (int i = 0; i < vertex_count; ++i) {
        row_offsets[static_cast<size_t>(i) + 1u] = row_offsets[static_cast<size_t>(i)] + degrees[static_cast<size_t>(i)];
    }
    const unsigned int nnz = row_offsets[static_cast<size_t>(vertex_count)];
    if (nnz == 0u) {
        return set_error("PCG stretch optimization found no valid CSR edges");
    }

    std::vector<unsigned int> cursor = row_offsets;
    std::vector<unsigned int> col_indices(static_cast<size_t>(nnz), 0u);
    for (int e = 0; e < edge_count; ++e) {
        const int i = static_cast<int>(springs[e].id0);
        const int j = static_cast<int>(springs[e].id1);
        if (i < 0 || i >= vertex_count || j < 0 || j >= vertex_count || i == j) {
            continue;
        }
        const unsigned int ij = cursor[static_cast<size_t>(i)]++;
        const unsigned int ji = cursor[static_cast<size_t>(j)]++;
        col_indices[static_cast<size_t>(ij)] = static_cast<unsigned int>(j);
        col_indices[static_cast<size_t>(ji)] = static_cast<unsigned int>(i);
        edge_entry_ij[static_cast<size_t>(e)] = static_cast<int>(ij);
        edge_entry_ji[static_cast<size_t>(e)] = static_cast<int>(ji);
    }

    destroy_pcg_texture(solver);
    solver->pcg_csr_nnz = static_cast<int>(nnz);
    bool ok = alloc_and_copy(&solver->pcg_row_offsets, row_offsets.data(), vertex_count + 1, "PCG CSR row offset allocation")
        && alloc_and_copy(&solver->pcg_col_indices, col_indices.data(), static_cast<int>(nnz), "PCG CSR column allocation")
        && alloc_and_copy(&solver->pcg_edge_entry_ij, edge_entry_ij.data(), edge_count, "PCG edge ij map allocation")
        && alloc_and_copy(&solver->pcg_edge_entry_ji, edge_entry_ji.data(), edge_count, "PCG edge ji map allocation")
        && alloc_and_copy(&solver->pcg_diag_values, static_cast<const ReconSymMat*>(nullptr), vertex_count, "PCG diagonal block allocation")
        && alloc_and_copy(&solver->pcg_preconditioner_inv, static_cast<const ReconSymMat*>(nullptr), vertex_count, "PCG preconditioner allocation")
        && alloc_and_copy(&solver->pcg_offdiag_texels, static_cast<const float4*>(nullptr), static_cast<int>(nnz) * 2, "PCG offdiag texture allocation")
        && alloc_and_copy(&solver->pcg_rhs, static_cast<const Vec3*>(nullptr), vertex_count, "PCG rhs allocation")
        && alloc_and_copy(&solver->pcg_solution, static_cast<const Vec3*>(nullptr), vertex_count, "PCG solution allocation")
        && alloc_and_copy(&solver->pcg_residual, static_cast<const Vec3*>(nullptr), vertex_count, "PCG residual allocation")
        && alloc_and_copy(&solver->pcg_z, static_cast<const Vec3*>(nullptr), vertex_count, "PCG preconditioned residual allocation")
        && alloc_and_copy(&solver->pcg_search_dir, static_cast<const Vec3*>(nullptr), vertex_count, "PCG search direction allocation")
        && alloc_and_copy(&solver->pcg_adir, static_cast<const Vec3*>(nullptr), vertex_count, "PCG A*d allocation")
        && alloc_and_copy(&solver->pcg_reductions, static_cast<const float*>(nullptr), kAbi41PcgReductionSlots, "PCG reduction allocation")
        && alloc_and_copy(&solver->pcg_max_delta_device, static_cast<const float*>(nullptr), 1, "PCG max delta allocation")
        && alloc_and_copy(&solver->pcg_guard_count, static_cast<const unsigned long long*>(nullptr), 1, "PCG guard counter allocation");
    if (!ok) {
        return false;
    }

    cudaResourceDesc resource_desc{};
    resource_desc.resType = cudaResourceTypeLinear;
    resource_desc.res.linear.devPtr = solver->pcg_offdiag_texels;
    resource_desc.res.linear.desc = cudaCreateChannelDesc<float4>();
    resource_desc.res.linear.sizeInBytes = sizeof(float4) * static_cast<size_t>(nnz) * 2u;
    cudaTextureDesc texture_desc{};
    texture_desc.readMode = cudaReadModeElementType;
    cudaTextureObject_t texture = 0;
    if (!set_cuda_error(
            cudaCreateTextureObject(&texture, &resource_desc, &texture_desc, nullptr),
            "create PCG CSR texture")) {
        return false;
    }
    solver->pcg_offdiag_texture = texture;
    solver->pcg_texture_ready = 1;
    solver->diag.abi41_pcg_csr_nnz = solver->pcg_csr_nnz;
    solver->diag.abi41_pcg_texture_ready = 1;
    return true;
}

bool fetch_pcg_reduction(Abi41Solver* solver, int slot, float* out_value, const char* label) {
    if (!solver || !solver->pcg_reductions || !out_value || slot < 0 || slot >= kAbi41PcgReductionSlots) {
        return set_error("invalid PCG reduction fetch");
    }
    return set_cuda_error(cudaMemcpy(out_value, solver->pcg_reductions + slot, sizeof(float), cudaMemcpyDeviceToHost), label);
}

bool fetch_pcg_reductions(Abi41Solver* solver, float* out_values, const char* label) {
    if (!solver || !solver->pcg_reductions || !out_values) {
        return set_error("invalid PCG reductions fetch");
    }
    return set_cuda_error(
        cudaMemcpy(out_values, solver->pcg_reductions, sizeof(float) * kAbi41PcgReductionSlots, cudaMemcpyDeviceToHost),
        label
    );
}

bool run_abi41_hard_stretch_pcg(Abi41Solver* solver, int v_blocks, int e_blocks) {
    if (!solver || !solver->cfg.stretch_optimization_enabled || solver->cfg.stretch_optimization_strength <= 0.0f) {
        return true;
    }
    if (solver->pcg_csr_nnz <= 0 || solver->pcg_texture_ready == 0 || solver->pcg_offdiag_texture == 0) {
        return set_error("PCG stretch optimization is enabled without a ready CSR texture");
    }

    abi41_pcg_reset_vertex_kernel<<<v_blocks, kThreads>>>(*solver);
    abi41_pcg_build_stretch_system_kernel<<<e_blocks, kThreads>>>(*solver);
    abi41_pcg_finalize_preconditioner_kernel<<<v_blocks, kThreads>>>(*solver);
    if (!set_cuda_error(cudaGetLastError(), "launch PCG stretch system build")) {
        return false;
    }

    if (abi41_pcg_device_scalar_enabled()) {
        abi41_pcg_compute_ad_kernel<<<v_blocks, kThreads>>>(*solver);
        abi41_pcg_update_solution_residual_z_device_alpha_kernel<<<v_blocks, kThreads>>>(*solver, 1);
        if (!set_cuda_error(cudaGetLastError(), "launch PCG device-scalar solve")) {
            return false;
        }

        float reductions[kAbi41PcgReductionSlots]{};
        if (!fetch_pcg_reductions(solver, reductions, "fetch PCG device-scalar reductions")) {
            return false;
        }

        const float rz_old = reductions[kAbi41PcgReductionRZ];
        const float rz_new = reductions[kAbi41PcgReductionRZNext];
        const float status = reductions[kAbi41PcgReductionStatus];
        if (!std::isfinite(rz_old) || rz_old < 0.0f || status == kAbi41PcgStatusBadResidual) {
            solver->diag.abi41_pcg_guarded += 1;
            return true;
        }

        const float initial_residual = sqrtf(fmaxf(rz_old, 0.0f));
        solver->diag.abi41_pcg_initial_residual = fmaxf(
            solver->diag.abi41_pcg_initial_residual,
            initial_residual
        );
        solver->diag.abi41_pcg_final_residual = initial_residual;
        if (status == kAbi41PcgStatusZeroResidual || rz_old <= 1.0e-14f) {
            return true;
        }
        if (status != kAbi41PcgStatusOk) {
            solver->diag.abi41_pcg_guarded += 1;
            return true;
        }
        if (!std::isfinite(rz_new) || rz_new < 0.0f) {
            solver->diag.abi41_pcg_guarded += 1;
            return true;
        }

        solver->diag.abi41_pcg_iterations += kAbi41PcgMaxIterations;
        solver->diag.abi41_pcg_final_residual = sqrtf(fmaxf(rz_new, 0.0f));
        return true;
    }

    float rz_old = 0.0f;
    if (!fetch_pcg_reduction(solver, kAbi41PcgReductionRZ, &rz_old, "fetch PCG initial residual")) {
        return false;
    }
    if (!std::isfinite(rz_old) || rz_old < 0.0f) {
        solver->diag.abi41_pcg_guarded += 1;
        return true;
    }

    const float initial_residual = sqrtf(fmaxf(rz_old, 0.0f));
    solver->diag.abi41_pcg_initial_residual = fmaxf(
        solver->diag.abi41_pcg_initial_residual,
        initial_residual
    );
    solver->diag.abi41_pcg_final_residual = initial_residual;
    if (rz_old <= 1.0e-14f) {
        return true;
    }

    abi41_pcg_compute_ad_kernel<<<v_blocks, kThreads>>>(*solver);
    if (!set_cuda_error(cudaGetLastError(), "launch PCG dAd reduction")) {
        return false;
    }
    float dad = 0.0f;
    if (!fetch_pcg_reduction(solver, kAbi41PcgReductionDAD, &dad, "fetch PCG dAd")) {
        return false;
    }
    if (!std::isfinite(dad) || dad <= 1.0e-14f) {
        solver->diag.abi41_pcg_guarded += 1;
        return true;
    }

    const float alpha = rz_old / dad;
    if (!std::isfinite(alpha)) {
        solver->diag.abi41_pcg_guarded += 1;
        return true;
    }
    abi41_pcg_update_solution_residual_z_kernel<<<v_blocks, kThreads>>>(*solver, alpha, 1);
    if (!set_cuda_error(cudaGetLastError(), "launch PCG residual update")) {
        return false;
    }
    float rz_new = 0.0f;
    if (!fetch_pcg_reduction(solver, kAbi41PcgReductionRZNext, &rz_new, "fetch PCG updated residual")) {
        return false;
    }
    if (!std::isfinite(rz_new) || rz_new < 0.0f) {
        solver->diag.abi41_pcg_guarded += 1;
        return true;
    }

    solver->diag.abi41_pcg_iterations += kAbi41PcgMaxIterations;
    solver->diag.abi41_pcg_final_residual = sqrtf(fmaxf(rz_new, 0.0f));
    return true;
}

bool reset_abi41_counts(Abi41Solver* solver) {
    if (!solver->abi41_counts) {
        return true;
    }
    if (!set_cuda_error(cudaMemset(solver->abi41_counts, 0, sizeof(unsigned long long) * kAbi41CountSlots), "reset recon diagnostics")) {
        return false;
    }
    if (solver->self_max_smoothed_delta_device
        && !set_cuda_error(cudaMemset(solver->self_max_smoothed_delta_device, 0, sizeof(float)), "reset self max smoothed delta")) {
        return false;
    }
    if (solver->pcg_reductions
        && !set_cuda_error(cudaMemset(solver->pcg_reductions, 0, sizeof(float) * kAbi41PcgReductionSlots), "reset PCG reductions")) {
        return false;
    }
    if (solver->pcg_max_delta_device
        && !set_cuda_error(cudaMemset(solver->pcg_max_delta_device, 0, sizeof(float)), "reset PCG max delta")) {
        return false;
    }
    if (solver->pcg_guard_count
        && !set_cuda_error(cudaMemset(solver->pcg_guard_count, 0, sizeof(unsigned long long)), "reset PCG guard count")) {
        return false;
    }
    return true;
}

bool fetch_abi41_counts(Abi41Solver* solver) {
    unsigned long long counts[kAbi41CountSlots]{};
    if (solver->abi41_counts
        && !set_cuda_error(cudaMemcpy(counts, solver->abi41_counts, sizeof(counts), cudaMemcpyDeviceToHost), "fetch recon diagnostics")) {
        return false;
    }
    solver->diag.abi41_soft_contact_count = static_cast<long long>(counts[kAbi41CountSoftContacts]);
    solver->diag.abi41_exact_impulse_contact_count = static_cast<long long>(counts[kAbi41CountExactImpulseContacts]);
    solver->diag.abi41_edge_edge_contact_count = static_cast<long long>(counts[kAbi41CountEdgeEdgeContacts]);
    solver->diag.abi41_hard_projection_fallbacks = static_cast<long long>(counts[kAbi41CountHardFallbacks]);
    solver->diag.abi41_max_smoothed_delta = 0.0f;
    if (solver->self_max_smoothed_delta_device
        && !set_cuda_error(
            cudaMemcpy(&solver->diag.abi41_max_smoothed_delta, solver->self_max_smoothed_delta_device, sizeof(float), cudaMemcpyDeviceToHost),
            "fetch self max smoothed delta")) {
        return false;
    }
    solver->diag.fast_soft_repulsion_candidates = static_cast<long long>(counts[kAbi41CountSelfCandidates]);
    solver->diag.fast_soft_repulsion_applied = static_cast<long long>(counts[kAbi41CountSoftContacts]);
    solver->diag.dynamic_particle_candidate_count = static_cast<long long>(counts[kAbi41CountDynamicParticleCandidates]);
    solver->diag.dynamic_particle_contacts = static_cast<long long>(counts[kAbi41CountDynamicParticleContacts]);
    solver->diag.dynamic_particle_overflow = 0;
    solver->diag.static_sdf_contact_count = static_cast<long long>(counts[kAbi41CountStaticSdfContacts]);
    solver->diag.candidate_count = static_cast<long long>(
        counts[kAbi41CountDynamicParticleCandidates] + counts[kAbi41CountTrianglePairs]
        + counts[kAbi41CountSelfCandidates] + counts[kAbi41CountStaticSdfContacts]
    );
    solver->diag.abi41_pcg_csr_nnz = solver->pcg_csr_nnz;
    solver->diag.abi41_pcg_texture_ready = solver->pcg_texture_ready;
    if (solver->pcg_guard_count) {
        unsigned long long guarded = 0;
        if (!set_cuda_error(
                cudaMemcpy(&guarded, solver->pcg_guard_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost),
                "fetch PCG guard count")) {
            return false;
        }
        solver->diag.abi41_pcg_guarded += static_cast<long long>(guarded);
    }
    if (solver->pcg_max_delta_device
        && !set_cuda_error(
            cudaMemcpy(&solver->diag.abi41_pcg_max_delta, solver->pcg_max_delta_device, sizeof(float), cudaMemcpyDeviceToHost),
            "fetch PCG max delta")) {
        return false;
    }
    populate_static_sdf_diagnostics(solver);
    return true;
}

void free_solver(Abi41Solver* solver) {
    if (!solver) {
        return;
    }
    destroy_pcg_texture(solver);
    cudaFree(solver->pos);
    cudaFree(solver->prev);
    cudaFree(solver->vel);
    cudaFree(solver->rest);
    cudaFree(solver->inv_mass);
    cudaFree(solver->state_flags);
    cudaFree(solver->springs);
    cudaFree(solver->triangles);
    cudaFree(solver->surface_vertex_offsets);
    cudaFree(solver->surface_vertex_triangles);
    cudaFree(solver->self_bucket_counts);
    cudaFree(solver->self_bucket_indices);
    cudaFree(solver->self_cell_coords);
    cudaFree(solver->self_triangle_bucket_counts);
    cudaFree(solver->self_triangle_bucket_indices);
    cudaFree(solver->self_triangle_cell_coords);
    cudaFree(solver->self_edge_bucket_counts);
    cudaFree(solver->self_edge_bucket_indices);
    cudaFree(solver->self_edge_cell_coords);
    cudaFree(solver->self_collision_counts);
    cudaFree(solver->self_collision_indices);
    cudaFree(solver->self_collision_radii);
    cudaFree(solver->self_accumulated_delta);
    cudaFree(solver->self_accumulated_weight);
    cudaFree(solver->self_averaged_delta);
    cudaFree(solver->self_max_smoothed_delta_device);
    cudaFree(solver->static_triangles);
    cudaFree(solver->static_sdf);
    cudaFree(solver->dynamic_triangles);
    cudaFree(solver->dynamic_triangle_bucket_counts);
    cudaFree(solver->dynamic_triangle_bucket_indices);
    cudaFree(solver->dynamic_triangle_cell_coords);
    cudaFree(solver->dynamic_particle_positions);
    cudaFree(solver->dynamic_particle_radii);
    cudaFree(solver->dynamic_particle_bucket_counts);
    cudaFree(solver->dynamic_particle_bucket_indices);
    cudaFree(solver->dynamic_particle_cell_coords);
    cudaFree(solver->triangle_pairs);
    cudaFree(solver->triangle_pair_count);
    cudaFree(solver->pin_indices);
    cudaFree(solver->pin_targets);
    cudaFree(solver->pcg_row_offsets);
    cudaFree(solver->pcg_col_indices);
    cudaFree(solver->pcg_edge_entry_ij);
    cudaFree(solver->pcg_edge_entry_ji);
    cudaFree(solver->pcg_diag_values);
    cudaFree(solver->pcg_preconditioner_inv);
    cudaFree(solver->pcg_offdiag_texels);
    cudaFree(solver->pcg_rhs);
    cudaFree(solver->pcg_solution);
    cudaFree(solver->pcg_residual);
    cudaFree(solver->pcg_z);
    cudaFree(solver->pcg_search_dir);
    cudaFree(solver->pcg_adir);
    cudaFree(solver->pcg_reductions);
    cudaFree(solver->pcg_max_delta_device);
    cudaFree(solver->pcg_guard_count);
    cudaFree(solver->force_fields);
    cudaFree(solver->abi41_counts);
    delete solver;
}

bool finite_config(const SsblXpbdConfig* config) {
    return config
        && config->vertex_count >= 0
        && config->edge_count >= 0
        && config->triangle_count >= 0
        && std::isfinite(config->dt)
        && config->dt > 0.0f;
}

bool upload_static_triangles(Abi41Solver* solver, const float* triangles, int triangle_count) {
    if (!solver) {
        return set_error("invalid static triangle update");
    }
    triangle_count = std::max(triangle_count, 0);
    if (triangle_count == 0) {
        solver->static_triangle_count = 0;
        solver->diag.static_triangle_count = 0;
        solver->cfg.static_triangle_count = 0;
        reset_static_sdf(solver, false);
        populate_static_sdf_diagnostics(solver);
        return true;
    }
    if (!triangles) {
        return set_error("static triangle data is required when triangle count is nonzero");
    }
    int vec_count = triangle_count * 3;
    if (triangle_count > solver->static_triangle_capacity) {
        cudaFree(solver->static_triangles);
        solver->static_triangles = nullptr;
        solver->static_triangle_capacity = 0;
        if (!alloc_and_copy(&solver->static_triangles, reinterpret_cast<const Vec3*>(triangles), vec_count, "static triangle upload")) {
            return false;
        }
        solver->static_triangle_capacity = triangle_count;
    } else if (!set_cuda_error(cudaMemcpy(solver->static_triangles, triangles, sizeof(Vec3) * vec_count, cudaMemcpyHostToDevice), "static triangle upload")) {
        return false;
    }
    solver->static_triangle_count = triangle_count;
    solver->diag.static_triangle_count = triangle_count;
    solver->cfg.static_triangle_count = triangle_count;
    if (!build_static_sdf(solver, triangles, triangle_count)) {
        return false;
    }
    return true;
}

bool upload_dynamic_triangles(Abi41Solver* solver, const float* triangles, int triangle_count) {
    if (!solver) {
        return set_error("invalid dynamic triangle update");
    }
    triangle_count = std::max(triangle_count, 0);
    if (triangle_count == 0) {
        solver->dynamic_triangle_count = 0;
        solver->diag.dynamic_triangle_count = 0;
        solver->dynamic_triangle_hash_ready = 0;
        return true;
    }
    if (!triangles) {
        return set_error("dynamic triangle data is required when triangle count is nonzero");
    }
    float max_extent = 0.0f;
    const Vec3* triangle_vecs = reinterpret_cast<const Vec3*>(triangles);
    for (int t = 0; t < triangle_count; ++t) {
        const Vec3 a = triangle_vecs[t * 3 + 0];
        const Vec3 b = triangle_vecs[t * 3 + 1];
        const Vec3 c = triangle_vecs[t * 3 + 2];
        const float min_x = std::min(a.x, std::min(b.x, c.x));
        const float min_y = std::min(a.y, std::min(b.y, c.y));
        const float min_z = std::min(a.z, std::min(b.z, c.z));
        const float max_x = std::max(a.x, std::max(b.x, c.x));
        const float max_y = std::max(a.y, std::max(b.y, c.y));
        const float max_z = std::max(a.z, std::max(b.z, c.z));
        max_extent = std::max(max_extent, std::max(max_x - min_x, std::max(max_y - min_y, max_z - min_z)));
    }
    const float margin = std::max(solver->cfg.collision_margin, 0.0f);
    const float thickness = std::max(solver->cfg.cloth_thickness, 0.0f);
    solver->dynamic_triangle_hash_cell_size = std::max(std::max(max_extent + margin * 2.0f, thickness), 1.0e-3f);
    int vec_count = triangle_count * 3;
    if (triangle_count > solver->dynamic_triangle_capacity) {
        cudaFree(solver->dynamic_triangles);
        solver->dynamic_triangles = nullptr;
        solver->dynamic_triangle_capacity = 0;
        if (!alloc_and_copy(&solver->dynamic_triangles, reinterpret_cast<const Vec3*>(triangles), vec_count, "dynamic triangle upload")) {
            return false;
        }
        solver->dynamic_triangle_capacity = triangle_count;
    } else if (!set_cuda_error(cudaMemcpy(solver->dynamic_triangles, triangles, sizeof(Vec3) * vec_count, cudaMemcpyHostToDevice), "dynamic triangle upload")) {
        return false;
    }
    solver->dynamic_triangle_count = triangle_count;
    solver->diag.dynamic_triangle_count = triangle_count;
    if (!prepare_dynamic_triangle_hash_buffers(solver, triangle_count)) {
        return false;
    }
    return true;
}

bool upload_dynamic_particles(
    Abi41Solver* solver,
    const float* positions,
    const float* radii,
    int particle_count
) {
    if (!solver) {
        return set_error("invalid dynamic particle update");
    }
    particle_count = std::max(particle_count, 0);
    if (particle_count == 0) {
        solver->dynamic_particle_count = 0;
        solver->diag.dynamic_particle_count = 0;
        solver->dynamic_particle_hash_ready = 0;
        return true;
    }
    if (!positions || !radii) {
        return set_error("dynamic particle position/radius data is required when particle count is nonzero");
    }
    float max_radius = 0.0f;
    for (int i = 0; i < particle_count; ++i) {
        const float r = radii[i];
        if (std::isfinite(r)) {
            max_radius = std::max(max_radius, std::max(r, 0.0f));
        }
    }
    const float margin = std::max(solver->cfg.collision_margin, 0.0f);
    const float thickness = std::max(solver->cfg.cloth_thickness, 0.0f);
    solver->dynamic_particle_hash_cell_size = std::max(std::max(max_radius + margin, thickness), 1.0e-3f);
    if (particle_count > solver->dynamic_particle_capacity) {
        cudaFree(solver->dynamic_particle_positions);
        cudaFree(solver->dynamic_particle_radii);
        solver->dynamic_particle_positions = nullptr;
        solver->dynamic_particle_radii = nullptr;
        solver->dynamic_particle_capacity = 0;
        if (!alloc_and_copy(&solver->dynamic_particle_positions, reinterpret_cast<const Vec3*>(positions), particle_count, "dynamic particle position upload")
            || !alloc_and_copy(&solver->dynamic_particle_radii, radii, particle_count, "dynamic particle radius upload")) {
            return false;
        }
        solver->dynamic_particle_capacity = particle_count;
    } else if (!set_cuda_error(cudaMemcpy(solver->dynamic_particle_positions, positions, sizeof(Vec3) * particle_count, cudaMemcpyHostToDevice), "dynamic particle position upload")
        || !set_cuda_error(cudaMemcpy(solver->dynamic_particle_radii, radii, sizeof(float) * particle_count, cudaMemcpyHostToDevice), "dynamic particle radius upload")) {
        return false;
    }
    solver->dynamic_particle_count = particle_count;
    solver->diag.dynamic_particle_count = particle_count;
    if (!prepare_dynamic_particle_hash_buffers(solver, particle_count)) {
        return false;
    }
    return true;
}

} // namespace

extern "C" SSBL_API void* ssbl_create_solver(const SsblXpbdConfig* config, const SsblXpbdMesh* mesh) {
    g_last_error.clear();
    if (!finite_config(config) || !mesh || !mesh->positions || !mesh->inv_mass) {
        set_error("invalid ABI39 ABI41 solver create request");
        return nullptr;
    }
    auto* solver = new Abi41Solver();
    solver->cfg = *config;
    solver->cfg.vertex_count = std::max(solver->cfg.vertex_count, 0);
    solver->cfg.edge_count = std::max(solver->cfg.edge_count, 0);
    solver->cfg.triangle_count = std::max(solver->cfg.triangle_count, 0);
    solver->cfg.damping = std::isfinite(solver->cfg.damping) ? solver->cfg.damping : 1.0f;
    solver->cfg.damping = std::max(0.0f, std::min(solver->cfg.damping, 1.0f));
    if (!std::isfinite(solver->cfg.stretch_optimization_strength)) {
        solver->cfg.stretch_optimization_strength = 0.0f;
    }
    solver->cfg.stretch_optimization_strength = clamp01(solver->cfg.stretch_optimization_strength);
    solver->cfg.stretch_optimization_enabled = (
        solver->cfg.edge_count > 0
        && solver->cfg.stretch_optimization_enabled
        && solver->cfg.stretch_optimization_strength > 0.0f
    ) ? 1 : 0;
    solver->runtime_colliders.use_ground = solver->cfg.use_ground;
    solver->runtime_colliders.ground_height = solver->cfg.ground_height;
    solver->runtime_colliders.use_wall = solver->cfg.use_wall;
    std::memcpy(solver->runtime_colliders.wall_origin, solver->cfg.wall_origin, sizeof(float) * 3);
    std::memcpy(solver->runtime_colliders.wall_normal, solver->cfg.wall_normal, sizeof(float) * 3);
    solver->runtime_colliders.use_sphere = solver->cfg.use_sphere;
    std::memcpy(solver->runtime_colliders.sphere_center, solver->cfg.sphere_center, sizeof(float) * 3);
    solver->runtime_colliders.sphere_radius = solver->cfg.sphere_radius;

    int n = solver->cfg.vertex_count;
    solver->triangle_pair_capacity = std::max(n * 2, 1024);
    std::vector<Vec3> zero_vel(n, make_vec3(0.0f, 0.0f, 0.0f));
    std::vector<unsigned int> flags(n, 0u);
    for (int i = 0; i < n; ++i) {
        if (mesh->inv_mass[i] <= 0.0f) {
            flags[i] = ssbl_abi41::kPinnedOrKinematicFlag;
        }
    }
    std::vector<ReconSpring> springs;
    springs.reserve(static_cast<size_t>(solver->cfg.edge_count));
    for (int e = 0; e < solver->cfg.edge_count; ++e) {
        int i = mesh->edges ? mesh->edges[e * 2 + 0] : 0;
        int j = mesh->edges ? mesh->edges[e * 2 + 1] : 0;
        float rest = mesh->edge_rest_lengths ? mesh->edge_rest_lengths[e] : 0.0f;
        springs.push_back(ReconSpring{static_cast<unsigned int>(std::max(i, 0)), static_cast<unsigned int>(std::max(j, 0)), rest});
    }
    std::vector<ReconTriangle> triangles;
    triangles.reserve(static_cast<size_t>(solver->cfg.triangle_count));
    for (int t = 0; t < solver->cfg.triangle_count; ++t) {
        int i = mesh->triangles ? mesh->triangles[t * 3 + 0] : 0;
        int j = mesh->triangles ? mesh->triangles[t * 3 + 1] : 0;
        int k = mesh->triangles ? mesh->triangles[t * 3 + 2] : 0;
        triangles.push_back(ReconTriangle{
            static_cast<unsigned int>(std::max(i, 0)),
            static_cast<unsigned int>(std::max(j, 0)),
            static_cast<unsigned int>(std::max(k, 0))
        });
    }

    bool ok = alloc_and_copy(&solver->pos, reinterpret_cast<const Vec3*>(mesh->positions), n, "position allocation")
        && alloc_and_copy(&solver->prev, reinterpret_cast<const Vec3*>(mesh->positions), n, "previous position allocation")
        && alloc_and_copy(&solver->rest, reinterpret_cast<const Vec3*>(mesh->positions), n, "rest position allocation")
        && alloc_and_copy(&solver->vel, zero_vel.data(), n, "velocity allocation")
        && alloc_and_copy(&solver->inv_mass, mesh->inv_mass, n, "inverse mass allocation")
        && alloc_and_copy(&solver->state_flags, flags.data(), n, "state flag allocation")
        && alloc_and_copy(&solver->springs, springs.data(), solver->cfg.edge_count, "spring allocation")
        && alloc_and_copy(&solver->triangles, triangles.data(), solver->cfg.triangle_count, "triangle allocation")
        && build_surface_vertex_triangles(solver, n, triangles)
        && alloc_and_copy(&solver->triangle_pairs, static_cast<const TriangleProximityPair*>(nullptr), solver->triangle_pair_capacity, "triangle pair allocation")
        && alloc_and_copy(&solver->triangle_pair_count, static_cast<const int*>(nullptr), 1, "triangle pair count allocation")
        && alloc_and_copy(&solver->abi41_counts, static_cast<const unsigned long long*>(nullptr), kAbi41CountSlots, "recon diagnostic allocation")
        && alloc_and_copy(&solver->self_collision_counts, static_cast<const unsigned int*>(nullptr), n, "self collision count allocation")
        && alloc_and_copy(&solver->self_collision_indices, static_cast<const unsigned int*>(nullptr), n * kAbi41SelfCollisionNeighborSlots, "self collision index allocation")
        && alloc_and_copy(&solver->self_collision_radii, static_cast<const float*>(nullptr), n, "self collision radius allocation")
        && alloc_and_copy(&solver->self_accumulated_delta, static_cast<const Vec3*>(nullptr), n, "self accumulated delta allocation")
        && alloc_and_copy(&solver->self_accumulated_weight, static_cast<const float*>(nullptr), n, "self accumulated weight allocation")
        && alloc_and_copy(&solver->self_averaged_delta, static_cast<const Vec3*>(nullptr), n, "self averaged delta allocation")
        && alloc_and_copy(&solver->self_max_smoothed_delta_device, static_cast<const float*>(nullptr), 1, "self max smoothed delta allocation");
    if (ok && !prepare_self_collision_hash_buffers(solver)) {
        ok = false;
    }
    if (ok && !prepare_self_triangle_hash_buffers(solver)) {
        ok = false;
    }
    if (ok && !prepare_self_edge_hash_buffers(solver)) {
        ok = false;
    }
    if (ok && !prepare_pcg_stretch_buffers(solver, springs)) {
        ok = false;
    }
    if (!ok || !reset_abi41_counts(solver)) {
        free_solver(solver);
        return nullptr;
    }
    solver->diag.finite_flag = 1;
    const int initial_static_triangle_count = std::max(solver->cfg.static_triangle_count, 0);
    if (!upload_static_triangles(solver, mesh->static_triangles, initial_static_triangle_count)) {
        free_solver(solver);
        return nullptr;
    }
    return solver;
}

extern "C" SSBL_API int ssbl_destroy_solver(void* handle) {
    g_last_error.clear();
    free_solver(reinterpret_cast<Abi41Solver*>(handle));
    return 1;
}

extern "C" SSBL_API int ssbl_reset_solver(void* handle) {
    g_last_error.clear();
    Abi41Solver* solver = reinterpret_cast<Abi41Solver*>(handle);
    if (!solver) {
        return set_error("invalid solver handle") ? 1 : 0;
    }
    int n = solver->cfg.vertex_count;
    if (!set_cuda_error(cudaMemcpy(solver->pos, solver->rest, sizeof(Vec3) * n, cudaMemcpyDeviceToDevice), "reset positions")
        || !set_cuda_error(cudaMemcpy(solver->prev, solver->rest, sizeof(Vec3) * n, cudaMemcpyDeviceToDevice), "reset previous positions")
        || !set_cuda_error(cudaMemset(solver->vel, 0, sizeof(Vec3) * n), "reset velocities")) {
        return 0;
    }
    solver->diag = SsblXpbdDiagnostics{};
    solver->diag.finite_flag = 1;
    return reset_abi41_counts(solver) ? 1 : 0;
}

extern "C" SSBL_API int ssbl_update_pin_targets(void* handle, const int* indices, const float* positions, int count) {
    g_last_error.clear();
    return upload_pins(reinterpret_cast<Abi41Solver*>(handle), indices, positions, count) ? 1 : 0;
}

extern "C" SSBL_API int ssbl_update_runtime_colliders(void* handle, const SsblXpbdRuntimeColliders* inputs) {
    g_last_error.clear();
    Abi41Solver* solver = reinterpret_cast<Abi41Solver*>(handle);
    if (!solver || !inputs) {
        return set_error("invalid runtime collider update") ? 1 : 0;
    }
    solver->runtime_colliders = *inputs;
    return 1;
}

extern "C" SSBL_API int ssbl_update_positions(void* handle, const float* positions, int max_floats) {
    g_last_error.clear();
    Abi41Solver* solver = reinterpret_cast<Abi41Solver*>(handle);
    if (!solver || !positions || max_floats < solver->cfg.vertex_count * 3) {
        return set_error("invalid position upload") ? 1 : 0;
    }
    int bytes = sizeof(Vec3) * solver->cfg.vertex_count;
    if (!set_cuda_error(cudaMemcpy(solver->pos, positions, bytes, cudaMemcpyHostToDevice), "upload positions")
        || !set_cuda_error(cudaMemcpy(solver->prev, positions, bytes, cudaMemcpyHostToDevice), "upload previous positions")) {
        return 0;
    }
    return 1;
}

extern "C" SSBL_API int ssbl_update_static_triangles(void* handle, const float* triangles, int triangle_count) {
    g_last_error.clear();
    return upload_static_triangles(reinterpret_cast<Abi41Solver*>(handle), triangles, triangle_count) ? 1 : 0;
}

extern "C" SSBL_API int ssbl_update_dynamic_triangles(void* handle, const float* triangles, int triangle_count) {
    g_last_error.clear();
    Abi41Solver* solver = reinterpret_cast<Abi41Solver*>(handle);
    if (!solver) {
        return set_error("invalid dynamic triangle update") ? 1 : 0;
    }
    return upload_dynamic_triangles(solver, triangles, triangle_count) ? 1 : 0;
}

extern "C" SSBL_API int ssbl_update_frame_inputs(void* handle, const SsblXpbdFrameInputs* inputs) {
    g_last_error.clear();
    Abi41Solver* solver = reinterpret_cast<Abi41Solver*>(handle);
    if (!solver || !inputs) {
        return set_error("invalid frame input update") ? 1 : 0;
    }
    if (inputs->update_pin_targets
        && !upload_pins(solver, inputs->pin_indices, inputs->pin_positions, inputs->pin_count)) {
        return 0;
    }
    if (inputs->update_runtime_colliders) {
        solver->runtime_colliders = inputs->runtime_colliders;
    }
    if (inputs->update_dynamic_triangles
        && !upload_dynamic_triangles(solver, inputs->dynamic_triangles, inputs->dynamic_triangle_count)) {
        return 0;
    }
    if (inputs->update_dynamic_particles
        && !upload_dynamic_particles(
            solver,
            inputs->dynamic_particle_positions,
            inputs->dynamic_particle_radii,
            inputs->dynamic_particle_count)) {
        return 0;
    }
    if (inputs->update_static_triangles
        && !upload_static_triangles(solver, inputs->static_triangles, inputs->static_triangle_count)) {
        return 0;
    }
    if (inputs->update_force_fields
        && !upload_force_fields(
            solver,
            inputs->force_fields,
            inputs->force_field_count,
            inputs->unsupported_force_field_count)) {
        return 0;
    }
    solver->diag.force_field_count = solver->force_field_count;
    solver->diag.unsupported_force_field_count = solver->unsupported_force_field_count;
    return 1;
}

extern "C" SSBL_API int ssbl_step_solver_ex(
    void* handle,
    int substeps,
    int iterations,
    int fetch_diagnostics,
    int force_sync
) {
    g_last_error.clear();
    Abi41Solver* solver = reinterpret_cast<Abi41Solver*>(handle);
    if (!solver) {
        return set_error("invalid solver handle") ? 1 : 0;
    }
    const auto started = std::chrono::high_resolution_clock::now();
    solver->diag = SsblXpbdDiagnostics{};
    solver->diag.finite_flag = 1;
    solver->diag.static_triangle_count = solver->static_triangle_count;
    solver->diag.dynamic_triangle_count = solver->dynamic_triangle_count;
    solver->diag.dynamic_particle_count = solver->dynamic_particle_count;
    solver->diag.force_field_count = solver->force_field_count;
    solver->diag.unsupported_force_field_count = solver->unsupported_force_field_count;
    populate_static_sdf_diagnostics(solver);
    if (!reset_abi41_counts(solver)) {
        return 0;
    }
    if (!build_dynamic_triangle_hash(solver)) {
        return 0;
    }
    if (!build_dynamic_particle_hash(solver)) {
        return 0;
    }

    substeps = std::max(substeps, 1);
    iterations = std::max(iterations, 1);
    int v_blocks = block_count(solver->cfg.vertex_count);
    int e_blocks = block_count(solver->cfg.edge_count);
    int p_blocks = block_count(solver->pin_count);
    float sub_dt = solver->cfg.dt / static_cast<float>(substeps);
    for (int s = 0; s < substeps; ++s) {
        abi41_integrate_kernel<<<v_blocks, kThreads>>>(*solver, sub_dt);
        if (solver->pin_count > 0) {
            abi41_pin_project_kernel<<<p_blocks, kThreads>>>(*solver);
        }
        if (!set_cuda_error(cudaGetLastError(), "launch ABI39 recon integrate/pin")) {
            return 0;
        }
        for (int it = 0; it < iterations; ++it) {
            const bool run_stretch_pcg = (
                solver->cfg.edge_count > 0
                && solver->cfg.stretch_optimization_enabled
                && solver->cfg.stretch_optimization_strength > 0.0f
                && s == substeps - 1
                && it == iterations - 1
            );
            if (solver->cfg.edge_count > 0 && !run_stretch_pcg) {
                abi41_spring_project_kernel<<<e_blocks, kThreads>>>(*solver, sub_dt);
            }
            if (run_stretch_pcg) {
                if (!run_abi41_hard_stretch_pcg(solver, v_blocks, e_blocks)) {
                    return 0;
                }
            }
            if (solver->pin_count > 0) {
                abi41_pin_project_kernel<<<p_blocks, kThreads>>>(*solver);
            }
            abi41_analytic_collision_kernel<<<v_blocks, kThreads>>>(*solver);
            if (solver->static_sdf_ready != 0) {
                const auto static_started = std::chrono::high_resolution_clock::now();
                abi41_static_sdf_collision_kernel<<<v_blocks, kThreads>>>(*solver, sub_dt);
                solver->diag.static_collision_ms += elapsed_ms_since(static_started);
            }
            if (solver->dynamic_triangle_count > 0) {
                const auto dynamic_started = std::chrono::high_resolution_clock::now();
                if (!set_cuda_error(cudaMemset(solver->triangle_pair_count, 0, sizeof(int)), "reset dynamic triangle pair count")) {
                    return 0;
                }
                abi41_build_dynamic_triangle_pairs_kernel<<<v_blocks, kThreads>>>(*solver);
                abi41_resolve_triangle_pairs_kernel<<<block_count(solver->triangle_pair_capacity), kThreads>>>(*solver);
                solver->diag.dynamic_collision_ms += elapsed_ms_since(dynamic_started);
            }
            if (solver->dynamic_particle_count > 0) {
                const auto dynamic_particle_started = std::chrono::high_resolution_clock::now();
                abi41_dynamic_particle_collision_kernel<<<v_blocks, kThreads>>>(*solver);
                solver->diag.dynamic_particle_collision_ms += elapsed_ms_since(dynamic_particle_started);
            }
            if (solver->cfg.self_collision) {
                if (solver->self_hash_ready != 0) {
                    const auto self_hash_started = std::chrono::high_resolution_clock::now();
                    if (!build_self_collision_hash(solver)) {
                        return 0;
                    }
                    solver->diag.self_hash_ms += elapsed_ms_since(self_hash_started);
                }
                if (solver->self_triangle_hash_ready != 0) {
                    const auto self_hash_started = std::chrono::high_resolution_clock::now();
                    if (!build_self_triangle_hash(solver)) {
                        return 0;
                    }
                    solver->diag.self_hash_ms += elapsed_ms_since(self_hash_started);
                }
                if (solver->self_edge_hash_ready != 0) {
                    const auto self_hash_started = std::chrono::high_resolution_clock::now();
                    if (!build_self_edge_hash(solver)) {
                        return 0;
                    }
                    solver->diag.self_hash_ms += elapsed_ms_since(self_hash_started);
                }
                const auto self_solve_started = std::chrono::high_resolution_clock::now();
                if (!reset_self_accumulation(solver) || !build_self_neighbor_table(solver)) {
                    return 0;
                }
                abi41_set_self_collision_repulsion_kernel<<<v_blocks, kThreads>>>(*solver);
                if (solver->cfg.triangle_count > 0) {
                    if (solver->self_triangle_hash_ready != 0) {
                        abi41_soft_vertex_triangle_repulsion_hash_kernel<<<v_blocks, kThreads>>>(*solver);
                    } else {
                        abi41_soft_vertex_triangle_repulsion_kernel<<<v_blocks, kThreads>>>(*solver);
                    }
                }
                if (solver->cfg.edge_count > 0) {
                    if (solver->self_edge_hash_ready != 0) {
                        abi41_soft_edge_edge_repulsion_hash_kernel<<<e_blocks, kThreads>>>(*solver);
                    } else {
                        abi41_soft_edge_edge_repulsion_kernel<<<e_blocks, kThreads>>>(*solver);
                    }
                }
                abi41_averaging_position_kernel<<<v_blocks, kThreads>>>(*solver);
                abi41_apply_self_averaged_delta_kernel<<<v_blocks, kThreads>>>(*solver);
                solver->diag.self_solve_ms += elapsed_ms_since(self_solve_started);
            }
            if (!set_cuda_error(cudaGetLastError(), "launch ABI39 recon constraints")) {
                return 0;
            }
        }
        abi41_update_velocity_kernel<<<v_blocks, kThreads>>>(*solver, sub_dt);
        if (!set_cuda_error(cudaGetLastError(), "launch ABI39 recon velocity")) {
            return 0;
        }
    }
    if (force_sync != 0 || fetch_diagnostics != 0) {
        const auto sync_started = std::chrono::high_resolution_clock::now();
        if (!set_cuda_error(cudaDeviceSynchronize(), "ABI39 ABI41 solver step")) {
            return 0;
        }
        solver->diag.sync_ms = elapsed_ms_since(sync_started);
    }
    if (fetch_diagnostics != 0 && !fetch_abi41_counts(solver)) {
        return 0;
    }
    solver->diag.step_ms = elapsed_ms_since(started);
    solver->diag.constraints_ms = solver->diag.step_ms;
    solver->diag.resolved_contacts = solver->diag.abi41_soft_contact_count
        + solver->diag.abi41_exact_impulse_contact_count
        + solver->diag.abi41_edge_edge_contact_count
        + solver->diag.static_sdf_contact_count;
    return 1;
}

extern "C" SSBL_API int ssbl_step_solver(void* handle, int substeps, int iterations) {
    return ssbl_step_solver_ex(handle, substeps, iterations, 1, 1);
}

extern "C" SSBL_API int ssbl_download_positions(void* handle, float* out_positions, int max_floats) {
    g_last_error.clear();
    Abi41Solver* solver = reinterpret_cast<Abi41Solver*>(handle);
    if (!solver || !out_positions || max_floats < solver->cfg.vertex_count * 3) {
        return set_error("invalid position download") ? 1 : 0;
    }
    return set_cuda_error(
        cudaMemcpy(out_positions, solver->pos, sizeof(Vec3) * solver->cfg.vertex_count, cudaMemcpyDeviceToHost),
        "download ABI39 recon positions"
    ) ? 1 : 0;
}

extern "C" SSBL_API int ssbl_get_diagnostics(void* handle, SsblXpbdDiagnostics* out_diag) {
    g_last_error.clear();
    Abi41Solver* solver = reinterpret_cast<Abi41Solver*>(handle);
    if (!solver || !out_diag) {
        return set_error("invalid diagnostics request") ? 1 : 0;
    }
    *out_diag = solver->diag;
    return 1;
}

extern "C" SSBL_API unsigned int ssbl_capabilities(void) {
    return SSBL_CAP_STRETCH_OPTIMIZATION;
}

extern "C" SSBL_API const char* ssbl_last_error(void) {
    return g_last_error.c_str();
}
