#include "ssbl_xpbd_cuda.h"
#include "ssbl_abi41_cuda_types.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {

using ReconSpring = ssbl_abi41::CudaSpringPBD;
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
constexpr int kAbi41CountSlots = 11;
constexpr float kAbi41SpringRelaxation = 0.18f;

std::string g_last_error;

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
    Vec3* static_triangles = nullptr;
    int* static_triangle_bucket_counts = nullptr;
    int* static_triangle_bucket_indices = nullptr;
    int* static_triangle_cell_coords = nullptr;
    int static_triangle_count = 0;
    int static_triangle_capacity = 0;
    int static_triangle_cell_capacity = 0;
    int static_triangle_hash_bucket_count = 0;
    int static_triangle_hash_ready = 0;
    float static_triangle_hash_cell_size = 0.0f;
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
    int force_field_count = 0;
    int unsupported_force_field_count = 0;
    int* pin_indices = nullptr;
    Vec3* pin_targets = nullptr;
    int pin_count = 0;
    int pin_capacity = 0;
    SsblXpbdRuntimeColliders runtime_colliders{};
    unsigned long long* abi41_counts = nullptr;
    SsblXpbdDiagnostics diag{};
};

bool set_error(const char* message) {
    g_last_error = message ? message : "unknown ABI37 recon CUDA error";
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

__host__ __device__ float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ Vec3 cross(Vec3 a, Vec3 b) {
    return make_vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

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

__device__ void atomic_add(Vec3* dst, Vec3 value) {
    atomicAdd(&dst->x, value.x);
    atomicAdd(&dst->y, value.y);
    atomicAdd(&dst->z, value.z);
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
    Vec3 gravity = make_vec3(solver.cfg.gravity[0], solver.cfg.gravity[1], solver.cfg.gravity[2]);
    Vec3 v = add(solver.vel[i], mul(gravity, dt));
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

__global__ void abi41_build_static_triangle_hash_kernel(Abi41Solver solver) {
    const int triangle = blockIdx.x * blockDim.x + threadIdx.x;
    if (triangle >= solver.static_triangle_count
        || !solver.static_triangles
        || !solver.static_triangle_bucket_counts
        || !solver.static_triangle_bucket_indices
        || !solver.static_triangle_cell_coords
        || solver.static_triangle_hash_bucket_count <= 0) {
        return;
    }
    Vec3 a = solver.static_triangles[triangle * 3 + 0];
    Vec3 b = solver.static_triangles[triangle * 3 + 1];
    Vec3 c = solver.static_triangles[triangle * 3 + 2];
    Vec3 center = mul(add(add(a, b), c), 1.0f / 3.0f);
    const int cx = cell_coord(center.x, solver.static_triangle_hash_cell_size);
    const int cy = cell_coord(center.y, solver.static_triangle_hash_cell_size);
    const int cz = cell_coord(center.z, solver.static_triangle_hash_cell_size);
    solver.static_triangle_cell_coords[triangle * 3 + 0] = cx;
    solver.static_triangle_cell_coords[triangle * 3 + 1] = cy;
    solver.static_triangle_cell_coords[triangle * 3 + 2] = cz;

    const float margin = fmaxf(solver.cfg.collision_margin, 0.0f);
    const float min_x = fminf(a.x, fminf(b.x, c.x)) - margin;
    const float min_y = fminf(a.y, fminf(b.y, c.y)) - margin;
    const float min_z = fminf(a.z, fminf(b.z, c.z)) - margin;
    const float max_x = fmaxf(a.x, fmaxf(b.x, c.x)) + margin;
    const float max_y = fmaxf(a.y, fmaxf(b.y, c.y)) + margin;
    const float max_z = fmaxf(a.z, fmaxf(b.z, c.z)) + margin;
    const int min_cx = cell_coord(min_x, solver.static_triangle_hash_cell_size);
    const int min_cy = cell_coord(min_y, solver.static_triangle_hash_cell_size);
    const int min_cz = cell_coord(min_z, solver.static_triangle_hash_cell_size);
    const int max_cx = cell_coord(max_x, solver.static_triangle_hash_cell_size);
    const int max_cy = cell_coord(max_y, solver.static_triangle_hash_cell_size);
    const int max_cz = cell_coord(max_z, solver.static_triangle_hash_cell_size);
    for (int z = min_cz; z <= max_cz; ++z) {
        for (int y = min_cy; y <= max_cy; ++y) {
            for (int x = min_cx; x <= max_cx; ++x) {
                const unsigned int bucket = hash_cell(x, y, z, solver.static_triangle_hash_bucket_count);
                const int slot = atomicAdd(&solver.static_triangle_bucket_counts[bucket], 1);
                if (slot < kAbi41TriangleHashBucketSlots) {
                    solver.static_triangle_bucket_indices[
                        static_cast<int>(bucket) * kAbi41TriangleHashBucketSlots + slot
                    ] = triangle;
                }
            }
        }
    }
}

__device__ void consider_static_triangle_candidate(
    Abi41Solver solver,
    int triangle,
    Vec3 p,
    float margin,
    Vec3* best_delta,
    float* best_penetration,
    int* best_triangle
) {
    if (triangle < 0 || triangle >= solver.static_triangle_count) {
        return;
    }
    Vec3 a = solver.static_triangles[triangle * 3 + 0];
    Vec3 b = solver.static_triangles[triangle * 3 + 1];
    Vec3 c = solver.static_triangles[triangle * 3 + 2];
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

__global__ void abi41_build_static_triangle_pairs_kernel(Abi41Solver solver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count
        || solver.inv_mass[i] <= 0.0f
        || !solver.static_triangles
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
    if (solver.static_triangle_hash_ready != 0
        && solver.static_triangle_bucket_counts
        && solver.static_triangle_bucket_indices
        && solver.static_triangle_cell_coords
        && solver.static_triangle_hash_bucket_count > 0) {
        const int cx = cell_coord(p.x, solver.static_triangle_hash_cell_size);
        const int cy = cell_coord(p.y, solver.static_triangle_hash_cell_size);
        const int cz = cell_coord(p.z, solver.static_triangle_hash_cell_size);
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int qx = cx + dx;
                    const int qy = cy + dy;
                    const int qz = cz + dz;
                    const unsigned int bucket = hash_cell(qx, qy, qz, solver.static_triangle_hash_bucket_count);
                    const int stored = solver.static_triangle_bucket_counts[bucket];
                    const int limit = stored < kAbi41TriangleHashBucketSlots ? stored : kAbi41TriangleHashBucketSlots;
                    for (int slot = 0; slot < limit; ++slot) {
                        const int triangle = solver.static_triangle_bucket_indices[
                            static_cast<int>(bucket) * kAbi41TriangleHashBucketSlots + slot
                        ];
                        consider_static_triangle_candidate(
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
        for (int t = 0; t < solver.static_triangle_count; ++t) {
            consider_static_triangle_candidate(
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
            pair.source = 0;
            pair.delta = best_delta;
            solver.triangle_pairs[pair_index] = pair;
        } else {
            atomicAdd(&solver.abi41_counts[kAbi41CountTrianglePairOverflow], 1ull);
        }
    }
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

__device__ bool abi41_apply_soft_self_pair(Abi41Solver solver, int i, int j, float target, float onset) {
    if (j <= i || j >= solver.cfg.vertex_count || target <= 0.0f || onset <= target) {
        return false;
    }
    const float wi = solver.inv_mass[i];
    const float wj = solver.inv_mass[j];
    if (wi <= 0.0f && wj <= 0.0f) {
        return false;
    }

    const Vec3 rest_delta = sub(solver.rest[i], solver.rest[j]);
    const float rest_skip = fmaxf(onset * 1.25f, target * 2.5f);
    if (dot(rest_delta, rest_delta) < rest_skip * rest_skip) {
        return false;
    }

    const Vec3 pi = solver.pos[i];
    const Vec3 pj = solver.pos[j];
    const Vec3 d = sub(pi, pj);
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
        const float rest_len_sq = dot(rest_delta, rest_delta);
        if (rest_len_sq > kEps) {
            normal = mul(rest_delta, rsqrtf(rest_len_sq));
        }
        dist = 0.0f;
    }

    const float penetration = target - dist;
    float push = 0.0f;
    if (penetration > 0.0f) {
        push = fminf(penetration * 0.45f + target * 0.02f, target * 0.35f);
    } else {
        const float x = 1.0f - fminf(dist / onset, 1.0f);
        push = fminf(x * x * target * 0.10f, target * 0.06f);
    }
    const float total = wi + wj;
    if (push <= 0.0f || total <= 0.0f) {
        return true;
    }

    const Vec3 delta = mul(normal, push / total);
    if (wi > 0.0f) {
        atomic_add(&solver.pos[i], mul(delta, wi));
    }
    if (wj > 0.0f) {
        atomic_add(&solver.pos[j], mul(delta, -wj));
    }
    atomicAdd(&solver.abi41_counts[kAbi41CountSoftContacts], 1ull);
    return true;
}

__global__ void abi41_soft_vertex_repulsion_hash_kernel(Abi41Solver solver) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count
        || !solver.self_bucket_counts
        || !solver.self_bucket_indices
        || !solver.self_cell_coords
        || solver.self_hash_bucket_count <= 0) {
        return;
    }
    const float target = abi41_self_contact_radius(solver.cfg);
    if (target <= 0.0f) {
        return;
    }
    const float onset = fmaxf(target * 1.8f, target + kEps);
    const Vec3 p = solver.pos[i];
    const int cx = cell_coord(p.x, solver.self_hash_cell_size);
    const int cy = cell_coord(p.y, solver.self_hash_cell_size);
    const int cz = cell_coord(p.z, solver.self_hash_cell_size);
    int max_neighbors = solver.cfg.max_self_collision_neighbors;
    if (max_neighbors < 1) {
        max_neighbors = 1;
    }
    if (max_neighbors > kAbi41SelfHashBucketSlots) {
        max_neighbors = kAbi41SelfHashBucketSlots;
    }
    int accepted = 0;
    for (int dz = -1; dz <= 1 && accepted < max_neighbors; ++dz) {
        for (int dy = -1; dy <= 1 && accepted < max_neighbors; ++dy) {
            for (int dx = -1; dx <= 1 && accepted < max_neighbors; ++dx) {
                const int qx = cx + dx;
                const int qy = cy + dy;
                const int qz = cz + dz;
                const unsigned int bucket = hash_cell(qx, qy, qz, solver.self_hash_bucket_count);
                const int stored = solver.self_bucket_counts[bucket];
                const int limit = stored < kAbi41SelfHashBucketSlots ? stored : kAbi41SelfHashBucketSlots;
                for (int slot = 0; slot < limit && accepted < max_neighbors; ++slot) {
                    const int j = solver.self_bucket_indices[static_cast<int>(bucket) * kAbi41SelfHashBucketSlots + slot];
                    if (solver.self_cell_coords[j * 3 + 0] != qx
                        || solver.self_cell_coords[j * 3 + 1] != qy
                        || solver.self_cell_coords[j * 3 + 2] != qz) {
                        continue;
                    }
                    if (abi41_apply_soft_self_pair(solver, i, j, target, onset)) {
                        ++accepted;
                    }
                }
            }
        }
    }
}

__global__ void abi41_soft_vertex_repulsion_kernel(Abi41Solver solver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = solver.cfg.vertex_count;
    if (i >= n) {
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
    if (max_neighbors > kAbi41SelfHashBucketSlots) {
        max_neighbors = kAbi41SelfHashBucketSlots;
    }
    int accepted = 0;
    for (int j = i + 1; j < n && accepted < max_neighbors; ++j) {
        if (abi41_apply_soft_self_pair(solver, i, j, target, onset)) {
            ++accepted;
        }
    }
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
        atomic_add(&solver.pos[vertex], mul(delta, wv));
    }
    if (wta > 0.0f) {
        atomic_add(&solver.pos[ia], mul(delta, -wta));
    }
    if (wtb > 0.0f) {
        atomic_add(&solver.pos[ib], mul(delta, -wtb));
    }
    if (wtc > 0.0f) {
        atomic_add(&solver.pos[ic], mul(delta, -wtc));
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
        atomic_add(&solver.pos[a0], mul(delta, wa0));
    }
    if (wa1 > 0.0f) {
        atomic_add(&solver.pos[a1], mul(delta, wa1));
    }
    if (wb0 > 0.0f) {
        atomic_add(&solver.pos[b0], mul(delta, -wb0));
    }
    if (wb1 > 0.0f) {
        atomic_add(&solver.pos[b1], mul(delta, -wb1));
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

int next_power_of_two(int value) {
    int result = 1;
    value = std::max(value, 1);
    while (result < value && result < (1 << 30)) {
        result <<= 1;
    }
    return result;
}

float triangle_hash_cell_size(Abi41Solver* solver, const float* triangles, int triangle_count) {
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
    return std::max(std::max(max_extent + margin * 2.0f, thickness), 1.0e-3f);
}

bool prepare_static_triangle_hash_buffers(Abi41Solver* solver, int triangle_count) {
    solver->static_triangle_hash_ready = 0;
    if (triangle_count < kAbi41TriangleHashMinCount) {
        return true;
    }
    if (triangle_count > solver->static_triangle_cell_capacity) {
        cudaFree(solver->static_triangle_cell_coords);
        solver->static_triangle_cell_coords = nullptr;
        solver->static_triangle_cell_capacity = 0;
        if (!alloc_and_copy(&solver->static_triangle_cell_coords, static_cast<const int*>(nullptr), triangle_count * 3, "static triangle cell allocation")) {
            return false;
        }
        solver->static_triangle_cell_capacity = triangle_count;
    } else if (!solver->static_triangle_cell_coords) {
        if (!alloc_and_copy(&solver->static_triangle_cell_coords, static_cast<const int*>(nullptr), triangle_count * 3, "static triangle cell allocation")) {
            return false;
        }
        solver->static_triangle_cell_capacity = triangle_count;
    }

    const int bucket_count = next_power_of_two(std::max(kAbi41MinTriangleHashBuckets, triangle_count * 4));
    if (bucket_count != solver->static_triangle_hash_bucket_count) {
        cudaFree(solver->static_triangle_bucket_counts);
        cudaFree(solver->static_triangle_bucket_indices);
        solver->static_triangle_bucket_counts = nullptr;
        solver->static_triangle_bucket_indices = nullptr;
        solver->static_triangle_hash_bucket_count = 0;
        if (!alloc_and_copy(&solver->static_triangle_bucket_counts, static_cast<const int*>(nullptr), bucket_count, "static triangle hash count allocation")
            || !alloc_and_copy(&solver->static_triangle_bucket_indices, static_cast<const int*>(nullptr), bucket_count * kAbi41TriangleHashBucketSlots, "static triangle hash index allocation")) {
            cudaFree(solver->static_triangle_bucket_counts);
            cudaFree(solver->static_triangle_bucket_indices);
            solver->static_triangle_bucket_counts = nullptr;
            solver->static_triangle_bucket_indices = nullptr;
            return false;
        }
        solver->static_triangle_hash_bucket_count = bucket_count;
    }
    solver->static_triangle_hash_ready = 1;
    return true;
}

bool build_static_triangle_hash(Abi41Solver* solver) {
    if (!solver || solver->static_triangle_hash_ready == 0 || solver->static_triangle_count <= 0) {
        return true;
    }
    if (!set_cuda_error(
        cudaMemset(solver->static_triangle_bucket_counts, 0, sizeof(int) * solver->static_triangle_hash_bucket_count),
        "reset static triangle hash"
    )) {
        return false;
    }
    abi41_build_static_triangle_hash_kernel<<<block_count(solver->static_triangle_count), kThreads>>>(*solver);
    return set_cuda_error(cudaGetLastError(), "build static triangle hash");
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

bool reset_abi41_counts(Abi41Solver* solver) {
    if (!solver->abi41_counts) {
        return true;
    }
    return set_cuda_error(cudaMemset(solver->abi41_counts, 0, sizeof(unsigned long long) * kAbi41CountSlots), "reset recon diagnostics");
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
    solver->diag.fast_soft_repulsion_candidates = static_cast<long long>(counts[kAbi41CountSelfCandidates]);
    solver->diag.fast_soft_repulsion_applied = static_cast<long long>(counts[kAbi41CountSoftContacts]);
    solver->diag.dynamic_particle_candidate_count = static_cast<long long>(counts[kAbi41CountDynamicParticleCandidates]);
    solver->diag.dynamic_particle_contacts = static_cast<long long>(counts[kAbi41CountDynamicParticleContacts]);
    solver->diag.dynamic_particle_overflow = 0;
    solver->diag.candidate_count = static_cast<long long>(
        counts[kAbi41CountDynamicParticleCandidates] + counts[kAbi41CountTrianglePairs]
        + counts[kAbi41CountSelfCandidates]
    );
    return true;
}

void free_solver(Abi41Solver* solver) {
    if (!solver) {
        return;
    }
    cudaFree(solver->pos);
    cudaFree(solver->prev);
    cudaFree(solver->vel);
    cudaFree(solver->rest);
    cudaFree(solver->inv_mass);
    cudaFree(solver->state_flags);
    cudaFree(solver->springs);
    cudaFree(solver->triangles);
    cudaFree(solver->self_bucket_counts);
    cudaFree(solver->self_bucket_indices);
    cudaFree(solver->self_cell_coords);
    cudaFree(solver->self_triangle_bucket_counts);
    cudaFree(solver->self_triangle_bucket_indices);
    cudaFree(solver->self_triangle_cell_coords);
    cudaFree(solver->self_edge_bucket_counts);
    cudaFree(solver->self_edge_bucket_indices);
    cudaFree(solver->self_edge_cell_coords);
    cudaFree(solver->static_triangles);
    cudaFree(solver->static_triangle_bucket_counts);
    cudaFree(solver->static_triangle_bucket_indices);
    cudaFree(solver->static_triangle_cell_coords);
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
        solver->static_triangle_hash_ready = 0;
        return true;
    }
    if (!triangles) {
        return set_error("static triangle data is required when triangle count is nonzero");
    }
    solver->static_triangle_hash_cell_size = triangle_hash_cell_size(solver, triangles, triangle_count);
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
    return prepare_static_triangle_hash_buffers(solver, triangle_count);
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
        set_error("invalid ABI37 ABI41 solver create request");
        return nullptr;
    }
    auto* solver = new Abi41Solver();
    solver->cfg = *config;
    solver->cfg.vertex_count = std::max(solver->cfg.vertex_count, 0);
    solver->cfg.edge_count = std::max(solver->cfg.edge_count, 0);
    solver->cfg.triangle_count = std::max(solver->cfg.triangle_count, 0);
    solver->cfg.damping = std::isfinite(solver->cfg.damping) ? solver->cfg.damping : 1.0f;
    solver->cfg.damping = std::max(0.0f, std::min(solver->cfg.damping, 1.0f));
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
        && alloc_and_copy(&solver->static_triangles, reinterpret_cast<const Vec3*>(mesh->static_triangles), solver->cfg.static_triangle_count * 3, "static triangle allocation")
        && alloc_and_copy(&solver->triangle_pairs, static_cast<const TriangleProximityPair*>(nullptr), solver->triangle_pair_capacity, "triangle pair allocation")
        && alloc_and_copy(&solver->triangle_pair_count, static_cast<const int*>(nullptr), 1, "triangle pair count allocation")
        && alloc_and_copy(&solver->abi41_counts, static_cast<const unsigned long long*>(nullptr), kAbi41CountSlots, "recon diagnostic allocation");
    if (ok && !prepare_self_collision_hash_buffers(solver)) {
        ok = false;
    }
    if (ok && !prepare_self_triangle_hash_buffers(solver)) {
        ok = false;
    }
    if (ok && !prepare_self_edge_hash_buffers(solver)) {
        ok = false;
    }
    if (!ok || !reset_abi41_counts(solver)) {
        free_solver(solver);
        return nullptr;
    }
    solver->diag.finite_flag = 1;
    solver->static_triangle_count = solver->cfg.static_triangle_count;
    solver->static_triangle_capacity = solver->cfg.static_triangle_count;
    if (solver->static_triangle_count > 0 && mesh->static_triangles) {
        solver->static_triangle_hash_cell_size = triangle_hash_cell_size(solver, mesh->static_triangles, solver->static_triangle_count);
        if (!prepare_static_triangle_hash_buffers(solver, solver->static_triangle_count)) {
            free_solver(solver);
            return nullptr;
        }
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
    if (inputs->update_force_fields) {
        solver->force_field_count = std::max(inputs->force_field_count, 0);
        solver->unsupported_force_field_count = std::max(inputs->unsupported_force_field_count, 0);
        solver->diag.force_field_count = solver->force_field_count;
        solver->diag.unsupported_force_field_count = solver->unsupported_force_field_count;
    }
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
    if (!reset_abi41_counts(solver)) {
        return 0;
    }
    if (!build_static_triangle_hash(solver)) {
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
    float sub_dt = solver->cfg.dt / static_cast<float>(substeps);
    for (int s = 0; s < substeps; ++s) {
        abi41_integrate_kernel<<<v_blocks, kThreads>>>(*solver, sub_dt);
        if (!set_cuda_error(cudaGetLastError(), "launch ABI37 recon integrate")) {
            return 0;
        }
        for (int it = 0; it < iterations; ++it) {
            if (solver->cfg.edge_count > 0) {
                abi41_spring_project_kernel<<<e_blocks, kThreads>>>(*solver, sub_dt);
            }
            if (solver->pin_count > 0) {
                abi41_pin_project_kernel<<<block_count(solver->pin_count), kThreads>>>(*solver);
            }
            abi41_analytic_collision_kernel<<<v_blocks, kThreads>>>(*solver);
            if (solver->static_triangle_count > 0 && solver->static_triangle_count <= 4096) {
                if (!set_cuda_error(cudaMemset(solver->triangle_pair_count, 0, sizeof(int)), "reset static triangle pair count")) {
                    return 0;
                }
                abi41_build_static_triangle_pairs_kernel<<<v_blocks, kThreads>>>(*solver);
                abi41_resolve_triangle_pairs_kernel<<<block_count(solver->triangle_pair_capacity), kThreads>>>(*solver);
            }
            if (solver->dynamic_triangle_count > 0 && solver->dynamic_triangle_count <= 4096) {
                if (!set_cuda_error(cudaMemset(solver->triangle_pair_count, 0, sizeof(int)), "reset dynamic triangle pair count")) {
                    return 0;
                }
                abi41_build_dynamic_triangle_pairs_kernel<<<v_blocks, kThreads>>>(*solver);
                abi41_resolve_triangle_pairs_kernel<<<block_count(solver->triangle_pair_capacity), kThreads>>>(*solver);
            }
            if (solver->dynamic_particle_count > 0 && solver->dynamic_particle_count <= 8192) {
                abi41_dynamic_particle_collision_kernel<<<v_blocks, kThreads>>>(*solver);
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
                if (solver->self_hash_ready != 0) {
                    abi41_soft_vertex_repulsion_hash_kernel<<<v_blocks, kThreads>>>(*solver);
                } else {
                    abi41_soft_vertex_repulsion_kernel<<<v_blocks, kThreads>>>(*solver);
                }
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
                solver->diag.self_solve_ms += elapsed_ms_since(self_solve_started);
            }
            if (!set_cuda_error(cudaGetLastError(), "launch ABI37 recon constraints")) {
                return 0;
            }
        }
        abi41_update_velocity_kernel<<<v_blocks, kThreads>>>(*solver, sub_dt);
        if (!set_cuda_error(cudaGetLastError(), "launch ABI37 recon velocity")) {
            return 0;
        }
    }
    if (force_sync != 0 || fetch_diagnostics != 0) {
        const auto sync_started = std::chrono::high_resolution_clock::now();
        if (!set_cuda_error(cudaDeviceSynchronize(), "ABI37 ABI41 solver step")) {
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
        + solver->diag.abi41_edge_edge_contact_count;
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
        "download ABI37 recon positions"
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

extern "C" SSBL_API const char* ssbl_last_error(void) {
    return g_last_error.c_str();
}
