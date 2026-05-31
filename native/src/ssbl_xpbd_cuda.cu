#include "ssbl_xpbd_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

thread_local std::string g_last_error;

constexpr float kEps = 1.0e-8f;
constexpr float kProjectionRelaxation = 0.35f;
constexpr float kSelfProjectionRelaxation = 0.25f;
constexpr float kMaxSubstepMove = 0.35f;
constexpr float kMaxVelocity = 35.0f;
constexpr int kMaxStaticTriangleHashCells = 256;
constexpr int kMaxStaticVertexQueryCells = 256;
constexpr int kMaxStaticVertexCandidates = 256;
constexpr int kStaticHashTriangleThreshold = 2048;
constexpr int kStaticCollisionPasses = 4;
constexpr int kMaxDynamicTriangleHashCells = 32;
constexpr int kMaxDynamicVertexQueryCells = 64;
constexpr int kMaxDynamicVertexCandidates = 64;
constexpr int kDynamicCollisionPasses = 1;

struct Vec3 {
    float x;
    float y;
    float z;
};

struct Int2 {
    int x;
    int y;
};

struct Int3 {
    int x;
    int y;
    int z;
};

struct Solver {
    SsblXpbdConfig cfg{};
    Vec3* pos = nullptr;
    Vec3* prev = nullptr;
    Vec3* vel = nullptr;
    Vec3* rest = nullptr;
    float* inv_mass = nullptr;
    Vec3* volume_gradient = nullptr;
    float* volume_accum = nullptr;
    Int2* edges = nullptr;
    float* edge_rest = nullptr;
    int* edge_color_offsets_host = nullptr;
    Int2* bends = nullptr;
    float* bend_rest = nullptr;
    int* bend_color_offsets_host = nullptr;
    Int2* lra_edges = nullptr;
    float* lra_rest = nullptr;
    int* lra_color_offsets_host = nullptr;
    Int3* triangles = nullptr;
    Vec3* static_triangles = nullptr;
    int* static_tri_heads = nullptr;
    int* static_tri_entry_next = nullptr;
    int* static_tri_entry_index = nullptr;
    int* static_tri_entry_count = nullptr;
    int static_tri_entry_capacity = 0;
    int static_hash_table_size = 0;
    Vec3* dynamic_triangles = nullptr;
    int dynamic_triangle_count = 0;
    int dynamic_triangle_capacity = 0;
    int dynamic_expected_triangle_count = -1;
    int* dynamic_tri_heads = nullptr;
    int* dynamic_tri_entry_next = nullptr;
    int* dynamic_tri_entry_index = nullptr;
    int* dynamic_tri_entry_count = nullptr;
    int dynamic_tri_entry_capacity = 0;
    int dynamic_hash_table_size = 0;
    int* self_vert_heads = nullptr;
    int* self_vert_next = nullptr;
    int self_vert_hash_table_size = 0;
    int* pin_indices = nullptr;
    Vec3* pin_targets = nullptr;
    int pin_count = 0;
    int pin_capacity = 0;
    float* pinned_download = nullptr;
    int pinned_download_floats = 0;
};

bool set_error(const char* message) {
    g_last_error = message ? message : "unknown CUDA error";
    return false;
}

bool set_cuda_error(cudaError_t err, const char* prefix) {
    if (err == cudaSuccess) {
        return true;
    }
    g_last_error = std::string(prefix) + ": " + cudaGetErrorString(err);
    return false;
}

template <typename T>
bool alloc_and_copy(T** dst, const T* src, int count, const char* label) {
    if (count <= 0) {
        *dst = nullptr;
        return true;
    }
    if (src == nullptr) {
        return set_error(label);
    }
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(dst), sizeof(T) * count);
    if (!set_cuda_error(err, "cudaMalloc")) {
        return false;
    }
    err = cudaMemcpy(*dst, src, sizeof(T) * count, cudaMemcpyHostToDevice);
    return set_cuda_error(err, "cudaMemcpy");
}

bool copy_host_offsets(int** dst, const int* src, int offset_count, const char* label) {
    if (offset_count <= 0) {
        *dst = nullptr;
        return true;
    }
    if (src == nullptr) {
        return set_error(label);
    }
    *dst = new int[offset_count];
    std::memcpy(*dst, src, sizeof(int) * offset_count);
    return true;
}

__device__ Vec3 add(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ Vec3 sub(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ Vec3 mul(Vec3 a, float s) {
    return {a.x * s, a.y * s, a.z * s};
}

__device__ float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ Vec3 cross(Vec3 a, Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

__device__ float norm(Vec3 a) {
    return sqrtf(fmaxf(dot(a, a), kEps));
}

__device__ Vec3 normalize(Vec3 a) {
    float len = norm(a);
    return {a.x / len, a.y / len, a.z / len};
}

__device__ bool finite_vec(Vec3 a) {
    return isfinite(a.x) && isfinite(a.y) && isfinite(a.z);
}

__device__ void atomic_add(Vec3* dst, Vec3 delta) {
    atomicAdd(&dst->x, delta.x);
    atomicAdd(&dst->y, delta.y);
    atomicAdd(&dst->z, delta.z);
}

__device__ int fast_floor_to_int(float value) {
    int i = static_cast<int>(value);
    return i > value ? i - 1 : i;
}

__device__ int cell_coord(float value, float cell_size) {
    return fast_floor_to_int(value / cell_size);
}

__device__ int hash_cell(int x, int y, int z, int table_size) {
    unsigned int h = static_cast<unsigned int>(x) * 73856093u
        ^ static_cast<unsigned int>(y) * 19349663u
        ^ static_cast<unsigned int>(z) * 83492791u;
    return static_cast<int>(h % static_cast<unsigned int>(table_size));
}

__device__ float self_cell_size(Solver solver) {
    float thickness = fmaxf(solver.cfg.cloth_thickness, solver.cfg.collision_margin);
    return fmaxf(thickness * 4.0f, 1.0e-3f);
}

__device__ float static_cell_size(Solver solver) {
    return fmaxf(solver.cfg.collision_margin * 2.0f, 1.25e-2f);
}

__device__ bool same_or_rest_neighbor(Solver solver, int a, int b) {
    if (a == b) {
        return true;
    }
    Vec3 rest_delta = sub(solver.rest[a], solver.rest[b]);
    float rest_dist = norm(rest_delta);
    float thickness = fmaxf(solver.cfg.cloth_thickness, solver.cfg.collision_margin);
    return rest_dist <= fmaxf(thickness * 5.0f, 1.0e-5f);
}


__global__ void integrate_kernel(Solver solver, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count) {
        return;
    }
    solver.prev[i] = solver.pos[i];
    if (solver.inv_mass[i] <= 0.0f) {
        solver.vel[i] = {0.0f, 0.0f, 0.0f};
        solver.pos[i] = solver.rest[i];
        return;
    }
    Vec3 g{solver.cfg.gravity[0], solver.cfg.gravity[1], solver.cfg.gravity[2]};
    Vec3 v = add(solver.vel[i], mul(g, dt));
    v = mul(v, solver.cfg.damping);
    solver.vel[i] = v;
    solver.pos[i] = add(solver.pos[i], mul(v, dt));
}

__global__ void edge_project_kernel(Solver solver, float dt) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= solver.cfg.edge_count) {
        return;
    }
    Int2 edge = solver.edges[e];
    int i = edge.x;
    int j = edge.y;
    float wi = solver.inv_mass[i];
    float wj = solver.inv_mass[j];
    float weight = wi + wj;
    if (weight <= 0.0f) {
        return;
    }
    Vec3 delta = sub(solver.pos[j], solver.pos[i]);
    float len = norm(delta);
    float c = len - solver.edge_rest[e];
    float alpha = solver.cfg.stretch_compliance / fmaxf(dt * dt, kEps);
    float dlambda = -c / (weight + alpha);
    Vec3 corr = mul(delta, kProjectionRelaxation * dlambda / len);
    atomic_add(&solver.pos[i], mul(corr, -wi));
    atomic_add(&solver.pos[j], mul(corr, wj));
}

__global__ void bend_project_kernel(Solver solver, float dt) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= solver.cfg.bend_count) {
        return;
    }
    Int2 pair = solver.bends[b];
    int i = pair.x;
    int j = pair.y;
    float wi = solver.inv_mass[i];
    float wj = solver.inv_mass[j];
    float weight = wi + wj;
    if (weight <= 0.0f) {
        return;
    }
    Vec3 delta = sub(solver.pos[j], solver.pos[i]);
    float len = norm(delta);
    float c = len - solver.bend_rest[b];
    float alpha = solver.cfg.bend_compliance / fmaxf(dt * dt, kEps);
    float dlambda = -c / (weight + alpha);
    Vec3 corr = mul(delta, kProjectionRelaxation * dlambda / len);
    atomic_add(&solver.pos[i], mul(corr, -wi));
    atomic_add(&solver.pos[j], mul(corr, wj));
}

__global__ void edge_project_range_kernel(Solver solver, float dt, int start, int count) {
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) {
        return;
    }
    int e = start + local;
    Int2 edge = solver.edges[e];
    int i = edge.x;
    int j = edge.y;
    float wi = solver.inv_mass[i];
    float wj = solver.inv_mass[j];
    float weight = wi + wj;
    if (weight <= 0.0f) {
        return;
    }
    Vec3 delta = sub(solver.pos[j], solver.pos[i]);
    float len = norm(delta);
    float c = len - solver.edge_rest[e];
    float alpha = solver.cfg.stretch_compliance / fmaxf(dt * dt, kEps);
    float dlambda = -c / (weight + alpha);
    Vec3 corr = mul(delta, kProjectionRelaxation * dlambda / len);
    solver.pos[i] = add(solver.pos[i], mul(corr, -wi));
    solver.pos[j] = add(solver.pos[j], mul(corr, wj));
}

__global__ void bend_project_range_kernel(Solver solver, float dt, int start, int count) {
    int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) {
        return;
    }
    int b = start + local;
    Int2 pair = solver.bends[b];
    int i = pair.x;
    int j = pair.y;
    float wi = solver.inv_mass[i];
    float wj = solver.inv_mass[j];
    float weight = wi + wj;
    if (weight <= 0.0f) {
        return;
    }
    Vec3 delta = sub(solver.pos[j], solver.pos[i]);
    float len = norm(delta);
    float c = len - solver.bend_rest[b];
    float alpha = solver.cfg.bend_compliance / fmaxf(dt * dt, kEps);
    float dlambda = -c / (weight + alpha);
    Vec3 corr = mul(delta, kProjectionRelaxation * dlambda / len);
    solver.pos[i] = add(solver.pos[i], mul(corr, -wi));
    solver.pos[j] = add(solver.pos[j], mul(corr, wj));
}

__global__ void lra_project_kernel(Solver solver, float dt) {
    int cidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cidx >= solver.cfg.lra_count) {
        return;
    }
    Int2 pair = solver.lra_edges[cidx];
    int i = pair.x;
    int j = pair.y;
    float wi = solver.inv_mass[i];
    float wj = solver.inv_mass[j];
    float weight = wi + wj;
    if (weight <= 0.0f) {
        return;
    }
    Vec3 delta = sub(solver.pos[j], solver.pos[i]);
    float len = norm(delta);
    float c = len - solver.lra_rest[cidx];
    if (c <= 0.0f) {
        return;
    }
    float alpha = solver.cfg.lra_compliance / fmaxf(dt * dt, kEps);
    float dlambda = -c / (weight + alpha);
    Vec3 corr = mul(delta, kProjectionRelaxation * dlambda / len);
    atomic_add(&solver.pos[i], mul(corr, -wi));
    atomic_add(&solver.pos[j], mul(corr, wj));
}

__global__ void pin_project_kernel(Solver solver) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= solver.pin_count) {
        return;
    }
    int i = solver.pin_indices[p];
    if (i < 0 || i >= solver.cfg.vertex_count) {
        return;
    }
    solver.pos[i] = solver.pin_targets[p];
    solver.vel[i] = {0.0f, 0.0f, 0.0f};
}

__global__ void volume_accumulate_kernel(Solver solver) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= solver.cfg.triangle_count) {
        return;
    }
    Int3 tri = solver.triangles[t];
    Vec3 a = solver.pos[tri.x];
    Vec3 b = solver.pos[tri.y];
    Vec3 c = solver.pos[tri.z];
    atomicAdd(&solver.volume_accum[0], dot(a, cross(b, c)) / 6.0f);
    atomic_add(&solver.volume_gradient[tri.x], mul(cross(b, c), 1.0f / 6.0f));
    atomic_add(&solver.volume_gradient[tri.y], mul(cross(c, a), 1.0f / 6.0f));
    atomic_add(&solver.volume_gradient[tri.z], mul(cross(a, b), 1.0f / 6.0f));
}

__global__ void volume_denominator_kernel(Solver solver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count || solver.inv_mass[i] <= 0.0f) {
        return;
    }
    Vec3 grad = solver.volume_gradient[i];
    if (!finite_vec(grad)) {
        return;
    }
    atomicAdd(&solver.volume_accum[1], solver.inv_mass[i] * dot(grad, grad));
}

__global__ void volume_project_kernel(Solver solver, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count || solver.inv_mass[i] <= 0.0f) {
        return;
    }
    float strength = fmaxf(solver.cfg.pressure_strength, 0.0f);
    if (strength <= 0.0f) {
        return;
    }
    float denom = solver.volume_accum[1];
    if (denom <= kEps || fabsf(solver.cfg.rest_volume) <= kEps) {
        return;
    }
    float target = solver.cfg.rest_volume * solver.cfg.volume_target_scale;
    float c = solver.volume_accum[0] - target;
    float alpha = fmaxf(solver.cfg.volume_compliance, 0.0f) / fmaxf(dt * dt, kEps);
    float dlambda = -c / (denom + alpha);
    Vec3 grad = solver.volume_gradient[i];
    if (!finite_vec(grad)) {
        return;
    }
    Vec3 correction = mul(grad, solver.inv_mass[i] * dlambda * strength);
    solver.pos[i] = add(solver.pos[i], correction);
}

__global__ void analytic_collision_kernel(Solver solver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count || solver.inv_mass[i] <= 0.0f) {
        return;
    }
    Vec3 p = solver.pos[i];
    Vec3 prev = solver.prev[i];
    float margin = solver.cfg.collision_margin;
    if (solver.cfg.use_ground && p.z < solver.cfg.ground_height + margin) {
        float target_z = solver.cfg.ground_height + margin;
        float correction = target_z - p.z;
        p.z = target_z;
        prev.z += correction;
    }
    if (solver.cfg.use_wall) {
        Vec3 o{solver.cfg.wall_origin[0], solver.cfg.wall_origin[1], solver.cfg.wall_origin[2]};
        Vec3 n = normalize({solver.cfg.wall_normal[0], solver.cfg.wall_normal[1], solver.cfg.wall_normal[2]});
        float d = dot(sub(p, o), n);
        if (d < margin) {
            Vec3 correction = mul(n, margin - d);
            p = add(p, correction);
            prev = add(prev, correction);
        }
    }
    if (solver.cfg.use_sphere) {
        Vec3 c{solver.cfg.sphere_center[0], solver.cfg.sphere_center[1], solver.cfg.sphere_center[2]};
        Vec3 delta = sub(p, c);
        if (dot(delta, delta) <= kEps) {
            delta = {0.0f, 0.0f, 1.0f};
        }
        float len = norm(delta);
        float radius = solver.cfg.sphere_radius + margin;
        if (len < radius) {
            Vec3 projected = add(c, mul(delta, radius / len));
            Vec3 correction = sub(projected, p);
            p = projected;
            prev = add(prev, correction);
        }
    }
    solver.pos[i] = p;
    solver.prev[i] = prev;
}

__device__ Vec3 closest_point_on_triangle(Vec3 p, Vec3 a, Vec3 b, Vec3 c) {
    Vec3 ab = sub(b, a);
    Vec3 ac = sub(c, a);
    Vec3 ap = sub(p, a);
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    Vec3 bp = sub(p, b);
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        return add(a, mul(ab, v));
    }

    Vec3 cp = sub(p, c);
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        return add(a, mul(ac, w));
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, mul(sub(c, b), w));
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return add(a, add(mul(ab, v), mul(ac, w)));
}

__device__ bool static_triangle_contact_candidate(
    Vec3 a,
    Vec3 b,
    Vec3 c,
    float margin,
    Vec3 p,
    Vec3 prev,
    Vec3* projected_out,
    float* score_out
) {
    Vec3 normal = normalize(cross(sub(b, a), sub(c, a)));
    if (!finite_vec(normal)) {
        return false;
    }

    Vec3 closest = closest_point_on_triangle(p, a, b, c);
    Vec3 delta = sub(p, closest);
    float d = norm(delta);

    if (d < margin) {
        float delta_sq = dot(delta, delta);
        if (delta_sq > 1.0e-12f) {
            *projected_out = add(closest, mul(delta, margin / sqrtf(delta_sq)));
        } else {
            float signed_now = dot(delta, normal);
            float signed_prev = dot(sub(prev, closest), normal);
            float side = signed_prev >= 0.0f ? 1.0f : -1.0f;
            if (fabsf(signed_prev) <= kEps && fabsf(signed_now) > kEps) {
                side = signed_now >= 0.0f ? 1.0f : -1.0f;
            }
            *projected_out = add(closest, mul(normal, side * margin));
        }
        *score_out = d;
        return true;
    }

    float signed_prev = dot(sub(prev, a), normal);
    float signed_now = dot(sub(p, a), normal);
    float denom = signed_prev - signed_now;
    if (signed_prev * signed_now < 0.0f && fabsf(denom) > kEps) {
        float t = signed_prev / denom;
        if (t >= 0.0f && t <= 1.0f) {
            Vec3 hit = add(prev, mul(sub(p, prev), t));
            Vec3 closest_hit = closest_point_on_triangle(hit, a, b, c);
            if (norm(sub(hit, closest_hit)) <= fmaxf(margin * 2.0f, 1.0e-4f)) {
                float side = signed_prev >= 0.0f ? 1.0f : -1.0f;
                *projected_out = add(closest_hit, mul(normal, side * margin));
                *score_out = -1.0f + t;
                return true;
            }
        }
    }
    return false;
}

__global__ void static_collision_kernel(Solver solver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count || solver.inv_mass[i] <= 0.0f) {
        return;
    }
    Vec3 p = solver.pos[i];
    Vec3 prev = solver.prev[i];
    float margin = solver.cfg.collision_margin;
    bool found = false;
    float best_score = 1.0e30f;
    Vec3 best_projected = p;
    for (int t = 0; t < solver.cfg.static_triangle_count; ++t) {
        Vec3 a = solver.static_triangles[t * 3 + 0];
        Vec3 b = solver.static_triangles[t * 3 + 1];
        Vec3 c = solver.static_triangles[t * 3 + 2];
        Vec3 projected = p;
        float score = 1.0e30f;
        if (static_triangle_contact_candidate(a, b, c, margin, p, prev, &projected, &score) && score < best_score) {
            found = true;
            best_score = score;
            best_projected = projected;
        }
    }
    if (found) {
        p = best_projected;
        prev = p;
    }
    solver.pos[i] = p;
    solver.prev[i] = prev;
}

__global__ void build_static_triangle_hash_kernel(Solver solver) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= solver.cfg.static_triangle_count) {
        return;
    }
    Vec3 a = solver.static_triangles[t * 3 + 0];
    Vec3 b = solver.static_triangles[t * 3 + 1];
    Vec3 c = solver.static_triangles[t * 3 + 2];
    float cell_size = static_cell_size(solver);
    float margin = solver.cfg.collision_margin;
    float expand = fmaxf(margin * 2.0f, cell_size * 0.5f);
    int min_x = cell_coord(fminf(fminf(a.x, b.x), c.x) - expand, cell_size);
    int min_y = cell_coord(fminf(fminf(a.y, b.y), c.y) - expand, cell_size);
    int min_z = cell_coord(fminf(fminf(a.z, b.z), c.z) - expand, cell_size);
    int max_x = cell_coord(fmaxf(fmaxf(a.x, b.x), c.x) + expand, cell_size);
    int max_y = cell_coord(fmaxf(fmaxf(a.y, b.y), c.y) + expand, cell_size);
    int max_z = cell_coord(fmaxf(fmaxf(a.z, b.z), c.z) + expand, cell_size);
    int inserted = 0;
    for (int z = min_z; z <= max_z && inserted < kMaxDynamicTriangleHashCells; ++z) {
        for (int y = min_y; y <= max_y && inserted < kMaxDynamicTriangleHashCells; ++y) {
            for (int x = min_x; x <= max_x && inserted < kMaxDynamicTriangleHashCells; ++x) {
                int entry = atomicAdd(solver.static_tri_entry_count, 1);
                if (entry >= solver.static_tri_entry_capacity) {
                    return;
                }
                int hash = hash_cell(x, y, z, solver.static_hash_table_size);
                solver.static_tri_entry_index[entry] = t;
                solver.static_tri_entry_next[entry] = atomicExch(&solver.static_tri_heads[hash], entry);
                ++inserted;
            }
        }
    }
}

__global__ void static_collision_hashed_kernel(Solver solver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count || solver.inv_mass[i] <= 0.0f) {
        return;
    }
    Vec3 p = solver.pos[i];
    Vec3 prev = solver.prev[i];
    float margin = solver.cfg.collision_margin;
    float cell_size = static_cell_size(solver);
    float expand = fmaxf(margin * 2.0f, 1.0e-4f);
    int min_x = cell_coord(fminf(p.x, prev.x) - expand, cell_size);
    int min_y = cell_coord(fminf(p.y, prev.y) - expand, cell_size);
    int min_z = cell_coord(fminf(p.z, prev.z) - expand, cell_size);
    int max_x = cell_coord(fmaxf(p.x, prev.x) + expand, cell_size);
    int max_y = cell_coord(fmaxf(p.y, prev.y) + expand, cell_size);
    int max_z = cell_coord(fmaxf(p.z, prev.z) + expand, cell_size);
    int queried = 0;
    int visited[kMaxDynamicVertexCandidates];
    int visited_count = 0;
    bool found = false;
    float best_score = 1.0e30f;
    Vec3 best_projected = p;
    for (int z = min_z; z <= max_z && queried < kMaxStaticVertexQueryCells; ++z) {
        for (int y = min_y; y <= max_y && queried < kMaxStaticVertexQueryCells; ++y) {
            for (int x = min_x; x <= max_x && queried < kMaxStaticVertexQueryCells; ++x) {
                int hash = hash_cell(x, y, z, solver.static_hash_table_size);
                int entry = solver.static_tri_heads[hash];
                while (entry >= 0 && visited_count < kMaxStaticVertexCandidates) {
                    int t = solver.static_tri_entry_index[entry];
                    bool duplicate = false;
                    for (int j = 0; j < visited_count; ++j) {
                        if (visited[j] == t) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate) {
                        visited[visited_count] = t;
                        ++visited_count;
                        Vec3 a = solver.static_triangles[t * 3 + 0];
                        Vec3 b = solver.static_triangles[t * 3 + 1];
                        Vec3 c = solver.static_triangles[t * 3 + 2];
                        Vec3 projected = p;
                        float score = 1.0e30f;
                        if (static_triangle_contact_candidate(a, b, c, margin, p, prev, &projected, &score) && score < best_score) {
                            found = true;
                            best_score = score;
                            best_projected = projected;
                        }
                    }
                    entry = solver.static_tri_entry_next[entry];
                }
                ++queried;
            }
        }
    }
    if (found) {
        p = best_projected;
        prev = p;
    }
    solver.pos[i] = p;
    solver.prev[i] = prev;
}

__global__ void dynamic_collision_kernel(Solver solver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count || solver.inv_mass[i] <= 0.0f) {
        return;
    }
    Vec3 p = solver.pos[i];
    Vec3 prev = solver.prev[i];
    float margin = solver.cfg.collision_margin;
    bool found = false;
    float best_score = 1.0e30f;
    Vec3 best_projected = p;
    for (int t = 0; t < solver.dynamic_triangle_count; ++t) {
        Vec3 a = solver.dynamic_triangles[t * 3 + 0];
        Vec3 b = solver.dynamic_triangles[t * 3 + 1];
        Vec3 c = solver.dynamic_triangles[t * 3 + 2];
        Vec3 projected = p;
        float score = 1.0e30f;
        if (static_triangle_contact_candidate(a, b, c, margin, p, prev, &projected, &score) && score < best_score) {
            found = true;
            best_score = score;
            best_projected = projected;
        }
    }
    if (found) {
        p = best_projected;
        prev = p;
    }
    solver.pos[i] = p;
    solver.prev[i] = prev;
}

__global__ void build_dynamic_triangle_hash_kernel(Solver solver) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= solver.dynamic_triangle_count) {
        return;
    }
    Vec3 a = solver.dynamic_triangles[t * 3 + 0];
    Vec3 b = solver.dynamic_triangles[t * 3 + 1];
    Vec3 c = solver.dynamic_triangles[t * 3 + 2];
    float cell_size = static_cell_size(solver);
    float margin = solver.cfg.collision_margin;
    float expand = fmaxf(margin * 2.0f, cell_size * 0.5f);
    int min_x = cell_coord(fminf(fminf(a.x, b.x), c.x) - expand, cell_size);
    int min_y = cell_coord(fminf(fminf(a.y, b.y), c.y) - expand, cell_size);
    int min_z = cell_coord(fminf(fminf(a.z, b.z), c.z) - expand, cell_size);
    int max_x = cell_coord(fmaxf(fmaxf(a.x, b.x), c.x) + expand, cell_size);
    int max_y = cell_coord(fmaxf(fmaxf(a.y, b.y), c.y) + expand, cell_size);
    int max_z = cell_coord(fmaxf(fmaxf(a.z, b.z), c.z) + expand, cell_size);
    int inserted = 0;
    for (int z = min_z; z <= max_z && inserted < kMaxStaticTriangleHashCells; ++z) {
        for (int y = min_y; y <= max_y && inserted < kMaxStaticTriangleHashCells; ++y) {
            for (int x = min_x; x <= max_x && inserted < kMaxStaticTriangleHashCells; ++x) {
                int entry = atomicAdd(solver.dynamic_tri_entry_count, 1);
                if (entry >= solver.dynamic_tri_entry_capacity) {
                    return;
                }
                int hash = hash_cell(x, y, z, solver.dynamic_hash_table_size);
                solver.dynamic_tri_entry_index[entry] = t;
                solver.dynamic_tri_entry_next[entry] = atomicExch(&solver.dynamic_tri_heads[hash], entry);
                ++inserted;
            }
        }
    }
}

__global__ void dynamic_collision_hashed_kernel(Solver solver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count || solver.inv_mass[i] <= 0.0f) {
        return;
    }
    Vec3 p = solver.pos[i];
    Vec3 prev = solver.prev[i];
    float margin = solver.cfg.collision_margin;
    float cell_size = static_cell_size(solver);
    float expand = fmaxf(margin * 2.0f, 1.0e-4f);
    int min_x = cell_coord(fminf(p.x, prev.x) - expand, cell_size);
    int min_y = cell_coord(fminf(p.y, prev.y) - expand, cell_size);
    int min_z = cell_coord(fminf(p.z, prev.z) - expand, cell_size);
    int max_x = cell_coord(fmaxf(p.x, prev.x) + expand, cell_size);
    int max_y = cell_coord(fmaxf(p.y, prev.y) + expand, cell_size);
    int max_z = cell_coord(fmaxf(p.z, prev.z) + expand, cell_size);
    int queried = 0;
    int visited[kMaxStaticVertexCandidates];
    int visited_count = 0;
    bool found = false;
    float best_score = 1.0e30f;
    Vec3 best_projected = p;
    for (int z = min_z; z <= max_z && queried < kMaxDynamicVertexQueryCells; ++z) {
        for (int y = min_y; y <= max_y && queried < kMaxDynamicVertexQueryCells; ++y) {
            for (int x = min_x; x <= max_x && queried < kMaxDynamicVertexQueryCells; ++x) {
                int hash = hash_cell(x, y, z, solver.dynamic_hash_table_size);
                int entry = solver.dynamic_tri_heads[hash];
                while (entry >= 0 && visited_count < kMaxDynamicVertexCandidates) {
                    int t = solver.dynamic_tri_entry_index[entry];
                    bool duplicate = false;
                    for (int j = 0; j < visited_count; ++j) {
                        if (visited[j] == t) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate) {
                        visited[visited_count] = t;
                        ++visited_count;
                        Vec3 a = solver.dynamic_triangles[t * 3 + 0];
                        Vec3 b = solver.dynamic_triangles[t * 3 + 1];
                        Vec3 c = solver.dynamic_triangles[t * 3 + 2];
                        Vec3 projected = p;
                        float score = 1.0e30f;
                        if (static_triangle_contact_candidate(a, b, c, margin, p, prev, &projected, &score) && score < best_score) {
                            found = true;
                            best_score = score;
                            best_projected = projected;
                        }
                    }
                    entry = solver.dynamic_tri_entry_next[entry];
                }
                ++queried;
            }
        }
    }
    if (found) {
        p = best_projected;
        prev = p;
    }
    solver.pos[i] = p;
    solver.prev[i] = prev;
}

__global__ void build_self_vertex_hash_kernel(Solver solver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count) {
        return;
    }
    Vec3 p = solver.pos[i];
    float cell_size = self_cell_size(solver);
    int cx = cell_coord(p.x, cell_size);
    int cy = cell_coord(p.y, cell_size);
    int cz = cell_coord(p.z, cell_size);
    int hash = hash_cell(cx, cy, cz, solver.self_vert_hash_table_size);
    solver.self_vert_next[i] = atomicExch(&solver.self_vert_heads[hash], i);
}

__global__ void self_particle_collision_kernel(Solver solver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count || solver.inv_mass[i] <= 0.0f) {
        return;
    }
    Vec3 p = solver.pos[i];
    float wi = solver.inv_mass[i];
    float margin = solver.cfg.collision_margin;
    float thickness = fmaxf(solver.cfg.cloth_thickness, margin);
    float cell_size = self_cell_size(solver);
    int max_neighbors = solver.cfg.max_self_collision_neighbors > 1 ? solver.cfg.max_self_collision_neighbors : 1;
    int candidates = 0;
    int scanned = 0;
    int scan_limit = max_neighbors * 8;
    int base_x = cell_coord(p.x, cell_size);
    int base_y = cell_coord(p.y, cell_size);
    int base_z = cell_coord(p.z, cell_size);

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int hash = hash_cell(base_x + dx, base_y + dy, base_z + dz, solver.self_vert_hash_table_size);
                int j = solver.self_vert_heads[hash];
                while (j >= 0 && candidates < max_neighbors && scanned < scan_limit) {
                    ++scanned;
                    if (j > i && solver.inv_mass[j] > 0.0f && !same_or_rest_neighbor(solver, i, j)) {
                        ++candidates;
                        Vec3 q = solver.pos[j];
                        Vec3 delta = sub(p, q);
                        float d2 = dot(delta, delta);
                        if (d2 < thickness * thickness && d2 > kEps) {
                            float d = sqrtf(d2);
                            Vec3 normal = mul(delta, 1.0f / d);
                            float wj = solver.inv_mass[j];
                            float total = wi + wj;
                            if (total > 0.0f) {
                                Vec3 correction = mul(normal, kSelfProjectionRelaxation * (thickness - d) / total);
                                atomic_add(&solver.pos[i], mul(correction, wi));
                                atomic_add(&solver.pos[j], mul(correction, -wj));
                            }
                        }
                    }
                    j = solver.self_vert_next[j];
                }
            }
        }
    }
}

__global__ void update_velocity_kernel(Solver solver, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count) {
        return;
    }
    if (solver.inv_mass[i] <= 0.0f) {
        solver.vel[i] = {0.0f, 0.0f, 0.0f};
        return;
    }
    Vec3 v = mul(sub(solver.pos[i], solver.prev[i]), 1.0f / fmaxf(dt, kEps));
    if (!finite_vec(v)) {
        solver.pos[i] = solver.rest[i];
        solver.prev[i] = solver.rest[i];
        solver.vel[i] = {0.0f, 0.0f, 0.0f};
        return;
    }
    float speed = norm(v);
    if (speed > kMaxVelocity) {
        v = mul(v, kMaxVelocity / speed);
        solver.pos[i] = add(solver.prev[i], mul(v, dt));
    }
    solver.vel[i] = v;
}

__global__ void sanitize_positions_kernel(Solver solver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count) {
        return;
    }
    Vec3 p = solver.pos[i];
    Vec3 prev = solver.prev[i];
    if (!finite_vec(p) || !finite_vec(prev)) {
        solver.pos[i] = solver.rest[i];
        solver.prev[i] = solver.rest[i];
        solver.vel[i] = {0.0f, 0.0f, 0.0f};
        return;
    }
    if (solver.inv_mass[i] <= 0.0f) {
        return;
    }
    Vec3 step_delta = sub(p, prev);
    float step_len = norm(step_delta);
    if (step_len > kMaxSubstepMove) {
        solver.pos[i] = add(prev, mul(step_delta, kMaxSubstepMove / step_len));
    }
}

int block_count(int count) {
    return (count + 255) / 256;
}

int next_power_of_two(int value) {
    int result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

bool allocate_dynamic_triangle_collision(Solver* solver, int triangle_count) {
    if (triangle_count <= 0) {
        solver->dynamic_triangle_count = 0;
        return true;
    }
    if (solver->dynamic_expected_triangle_count >= 0 && solver->dynamic_expected_triangle_count != triangle_count) {
        return set_error("dynamic collider triangle count changed; fixed topology is required");
    }
    if (solver->dynamic_expected_triangle_count < 0) {
        solver->dynamic_expected_triangle_count = triangle_count;
    }
    if (triangle_count <= solver->dynamic_triangle_capacity
        && solver->dynamic_triangles
        && solver->dynamic_tri_heads
        && solver->dynamic_tri_entry_next
        && solver->dynamic_tri_entry_index
        && solver->dynamic_tri_entry_count) {
        solver->dynamic_triangle_count = triangle_count;
        return true;
    }

    cudaFree(solver->dynamic_triangles);
    cudaFree(solver->dynamic_tri_heads);
    cudaFree(solver->dynamic_tri_entry_next);
    cudaFree(solver->dynamic_tri_entry_index);
    cudaFree(solver->dynamic_tri_entry_count);
    solver->dynamic_triangles = nullptr;
    solver->dynamic_tri_heads = nullptr;
    solver->dynamic_tri_entry_next = nullptr;
    solver->dynamic_tri_entry_index = nullptr;
    solver->dynamic_tri_entry_count = nullptr;

    solver->dynamic_triangle_capacity = triangle_count;
    solver->dynamic_triangle_count = triangle_count;
    solver->dynamic_hash_table_size = next_power_of_two(std::max(1024, triangle_count * 4));
    solver->dynamic_tri_entry_capacity = std::max(1, triangle_count * kMaxDynamicTriangleHashCells);

    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&solver->dynamic_triangles), sizeof(Vec3) * triangle_count * 3);
    if (!set_cuda_error(err, "dynamic collider triangle allocation")) {
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&solver->dynamic_tri_heads), sizeof(int) * solver->dynamic_hash_table_size);
    if (!set_cuda_error(err, "dynamic collider hash allocation")) {
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&solver->dynamic_tri_entry_next), sizeof(int) * solver->dynamic_tri_entry_capacity);
    if (!set_cuda_error(err, "dynamic collider link allocation")) {
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&solver->dynamic_tri_entry_index), sizeof(int) * solver->dynamic_tri_entry_capacity);
    if (!set_cuda_error(err, "dynamic collider entry allocation")) {
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&solver->dynamic_tri_entry_count), sizeof(int));
    return set_cuda_error(err, "dynamic collider counter allocation");
}

bool rebuild_dynamic_triangle_hash(Solver* solver) {
    if (!solver || solver->dynamic_triangle_count <= 0) {
        return true;
    }
    if (!solver->dynamic_tri_heads
        || !solver->dynamic_tri_entry_next
        || !solver->dynamic_tri_entry_index
        || !solver->dynamic_tri_entry_count) {
        return set_error("dynamic collider hash is not allocated");
    }
    cudaMemset(solver->dynamic_tri_heads, 0xff, sizeof(int) * solver->dynamic_hash_table_size);
    cudaMemset(solver->dynamic_tri_entry_count, 0, sizeof(int));
    build_dynamic_triangle_hash_kernel<<<block_count(solver->dynamic_triangle_count), 256>>>(*solver);
    return set_cuda_error(cudaDeviceSynchronize(), "rebuild dynamic collision hash");
}

void free_solver(Solver* solver) {
    if (!solver) {
        return;
    }
    cudaFree(solver->pos);
    cudaFree(solver->prev);
    cudaFree(solver->vel);
    cudaFree(solver->rest);
    cudaFree(solver->inv_mass);
    cudaFree(solver->volume_gradient);
    cudaFree(solver->volume_accum);
    cudaFree(solver->edges);
    cudaFree(solver->edge_rest);
    delete[] solver->edge_color_offsets_host;
    cudaFree(solver->bends);
    cudaFree(solver->bend_rest);
    delete[] solver->bend_color_offsets_host;
    cudaFree(solver->lra_edges);
    cudaFree(solver->lra_rest);
    delete[] solver->lra_color_offsets_host;
    cudaFree(solver->triangles);
    cudaFree(solver->static_triangles);
    cudaFree(solver->static_tri_heads);
    cudaFree(solver->static_tri_entry_next);
    cudaFree(solver->static_tri_entry_index);
    cudaFree(solver->static_tri_entry_count);
    cudaFree(solver->dynamic_triangles);
    cudaFree(solver->dynamic_tri_heads);
    cudaFree(solver->dynamic_tri_entry_next);
    cudaFree(solver->dynamic_tri_entry_index);
    cudaFree(solver->dynamic_tri_entry_count);
    cudaFree(solver->self_vert_heads);
    cudaFree(solver->self_vert_next);
    cudaFree(solver->pin_indices);
    cudaFree(solver->pin_targets);
    cudaFreeHost(solver->pinned_download);
    delete solver;
}

}  // namespace

extern "C" SSBL_API void* ssbl_create_solver(const SsblXpbdConfig* config, const SsblXpbdMesh* mesh) {
    g_last_error.clear();
    if (!config || !mesh) {
        set_error("missing solver config or mesh");
        return nullptr;
    }
    if (config->vertex_count <= 0 || !mesh->positions || !mesh->inv_mass) {
        set_error("invalid cloth vertex data");
        return nullptr;
    }

    Solver* solver = new Solver();
    solver->cfg = *config;
    // Removed self-collision modes above fast. Clamp stale callers to the
    // remaining particle-hash path.
    solver->cfg.self_collision_mode = solver->cfg.self_collision
        ? std::min(std::max(solver->cfg.self_collision_mode, 0), 1)
        : 0;

    const int vertex_count = solver->cfg.vertex_count;
    std::vector<Vec3> host_pos(vertex_count);
    std::memcpy(host_pos.data(), mesh->positions, sizeof(float) * vertex_count * 3);
    std::vector<Vec3> zero_vel(vertex_count, Vec3{0.0f, 0.0f, 0.0f});

    bool ok = true;
    ok = ok && alloc_and_copy(&solver->pos, host_pos.data(), vertex_count, "missing positions");
    ok = ok && alloc_and_copy(&solver->prev, host_pos.data(), vertex_count, "missing positions");
    ok = ok && alloc_and_copy(&solver->rest, host_pos.data(), vertex_count, "missing positions");
    ok = ok && alloc_and_copy(&solver->vel, zero_vel.data(), vertex_count, "missing velocity buffer");
    ok = ok && alloc_and_copy(&solver->inv_mass, mesh->inv_mass, vertex_count, "missing inverse masses");
    if (ok && config->use_volume_pressure) {
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&solver->volume_gradient), sizeof(Vec3) * vertex_count);
        ok = ok && set_cuda_error(err, "volume gradient allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->volume_accum), sizeof(float) * 2);
        ok = ok && set_cuda_error(err, "volume accumulator allocation");
    }
    if (ok) {
        solver->pinned_download_floats = vertex_count * 3;
        cudaError_t err = cudaMallocHost(
            reinterpret_cast<void**>(&solver->pinned_download),
            sizeof(float) * solver->pinned_download_floats
        );
        ok = ok && set_cuda_error(err, "pinned download allocation");
    }
    ok = ok && alloc_and_copy(&solver->edges, reinterpret_cast<const Int2*>(mesh->edges), config->edge_count, "missing edges");
    ok = ok && alloc_and_copy(&solver->edge_rest, mesh->edge_rest_lengths, config->edge_count, "missing edge rest lengths");
    ok = ok && copy_host_offsets(&solver->edge_color_offsets_host, mesh->edge_color_offsets, config->edge_color_count + 1, "missing edge color offsets");
    ok = ok && alloc_and_copy(&solver->bends, reinterpret_cast<const Int2*>(mesh->bends), config->bend_count, "missing bends");
    ok = ok && alloc_and_copy(&solver->bend_rest, mesh->bend_rest_lengths, config->bend_count, "missing bend rest lengths");
    ok = ok && copy_host_offsets(&solver->bend_color_offsets_host, mesh->bend_color_offsets, config->bend_color_count + 1, "missing bend color offsets");
    ok = ok && alloc_and_copy(&solver->lra_edges, reinterpret_cast<const Int2*>(mesh->lra_edges), config->lra_count, "missing LRA edges");
    ok = ok && alloc_and_copy(&solver->lra_rest, mesh->lra_rest_lengths, config->lra_count, "missing LRA rest lengths");
    ok = ok && copy_host_offsets(&solver->lra_color_offsets_host, mesh->lra_color_offsets, config->lra_color_count + 1, "missing LRA color offsets");
    ok = ok && alloc_and_copy(&solver->triangles, reinterpret_cast<const Int3*>(mesh->triangles), config->triangle_count, "missing triangles");
    ok = ok && alloc_and_copy(&solver->static_triangles, reinterpret_cast<const Vec3*>(mesh->static_triangles), config->static_triangle_count * 3, "missing static triangles");

    if (ok && config->static_triangle_count > 0) {
        solver->static_hash_table_size = next_power_of_two(std::max(2048, config->static_triangle_count * 8));
        solver->static_tri_entry_capacity = std::max(1, config->static_triangle_count * kMaxStaticTriangleHashCells);
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&solver->static_tri_heads), sizeof(int) * solver->static_hash_table_size);
        ok = ok && set_cuda_error(err, "static collision hash allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->static_tri_entry_next), sizeof(int) * solver->static_tri_entry_capacity);
        ok = ok && set_cuda_error(err, "static collision link allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->static_tri_entry_index), sizeof(int) * solver->static_tri_entry_capacity);
        ok = ok && set_cuda_error(err, "static collision entry allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->static_tri_entry_count), sizeof(int));
        ok = ok && set_cuda_error(err, "static collision counter allocation");
        if (ok) {
            cudaMemset(solver->static_tri_heads, 0xff, sizeof(int) * solver->static_hash_table_size);
            cudaMemset(solver->static_tri_entry_count, 0, sizeof(int));
            build_static_triangle_hash_kernel<<<block_count(config->static_triangle_count), 256>>>(*solver);
            ok = ok && set_cuda_error(cudaDeviceSynchronize(), "static collision hash build");
        }
    }

    if (ok && config->self_collision && config->vertex_count > 0) {
        solver->self_vert_hash_table_size = next_power_of_two(std::max(1024, config->vertex_count * 2));
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&solver->self_vert_heads), sizeof(int) * solver->self_vert_hash_table_size);
        ok = ok && set_cuda_error(err, "self vertex hash allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->self_vert_next), sizeof(int) * config->vertex_count);
        ok = ok && set_cuda_error(err, "self vertex link allocation");
    }

    if (!ok) {
        free_solver(solver);
        return nullptr;
    }
    return solver;
}

extern "C" SSBL_API int ssbl_destroy_solver(void* handle) {
    g_last_error.clear();
    free_solver(reinterpret_cast<Solver*>(handle));
    return 1;
}

extern "C" SSBL_API int ssbl_reset_solver(void* handle) {
    g_last_error.clear();
    Solver* solver = reinterpret_cast<Solver*>(handle);
    if (!solver) {
        return set_error("invalid solver handle") ? 1 : 0;
    }
    int n = solver->cfg.vertex_count;
    if (!set_cuda_error(cudaMemcpy(solver->pos, solver->rest, sizeof(Vec3) * n, cudaMemcpyDeviceToDevice), "reset positions")) {
        return 0;
    }
    if (!set_cuda_error(cudaMemset(solver->vel, 0, sizeof(Vec3) * n), "reset velocities")) {
        return 0;
    }
    return 1;
}

extern "C" SSBL_API int ssbl_update_pin_targets(void* handle, const int* indices, const float* positions, int count) {
    g_last_error.clear();
    Solver* solver = reinterpret_cast<Solver*>(handle);
    if (!solver) {
        return set_error("invalid solver handle") ? 1 : 0;
    }
    if (count <= 0) {
        solver->pin_count = 0;
        return 1;
    }
    if (!indices || !positions) {
        return set_error("missing pin targets") ? 1 : 0;
    }
    if (count > solver->pin_capacity) {
        cudaFree(solver->pin_indices);
        cudaFree(solver->pin_targets);
        solver->pin_indices = nullptr;
        solver->pin_targets = nullptr;
        solver->pin_capacity = count;
        if (!set_cuda_error(cudaMalloc(reinterpret_cast<void**>(&solver->pin_indices), sizeof(int) * count), "pin index allocation")) {
            return 0;
        }
        if (!set_cuda_error(cudaMalloc(reinterpret_cast<void**>(&solver->pin_targets), sizeof(Vec3) * count), "pin target allocation")) {
            return 0;
        }
    }
    solver->pin_count = count;
    if (!set_cuda_error(cudaMemcpy(solver->pin_indices, indices, sizeof(int) * count, cudaMemcpyHostToDevice), "pin index upload")) {
        return 0;
    }
    if (!set_cuda_error(cudaMemcpy(solver->pin_targets, positions, sizeof(float) * count * 3, cudaMemcpyHostToDevice), "pin target upload")) {
        return 0;
    }
    return 1;
}

extern "C" SSBL_API int ssbl_update_runtime_colliders(void* handle, const SsblXpbdRuntimeColliders* inputs) {
    g_last_error.clear();
    Solver* solver = reinterpret_cast<Solver*>(handle);
    if (!solver || !inputs) {
        return set_error("invalid runtime collider update") ? 1 : 0;
    }
    solver->cfg.use_ground = inputs->use_ground;
    solver->cfg.ground_height = inputs->ground_height;
    solver->cfg.use_wall = inputs->use_wall;
    std::memcpy(solver->cfg.wall_origin, inputs->wall_origin, sizeof(float) * 3);
    std::memcpy(solver->cfg.wall_normal, inputs->wall_normal, sizeof(float) * 3);
    solver->cfg.use_sphere = inputs->use_sphere;
    std::memcpy(solver->cfg.sphere_center, inputs->sphere_center, sizeof(float) * 3);
    solver->cfg.sphere_radius = inputs->sphere_radius;
    return 1;
}

extern "C" SSBL_API int ssbl_update_static_triangles(void* handle, const float* triangles, int triangle_count) {
    g_last_error.clear();
    Solver* solver = reinterpret_cast<Solver*>(handle);
    if (!solver) {
        return set_error("invalid static collider update") ? 1 : 0;
    }
    if (triangle_count != solver->cfg.static_triangle_count) {
        return set_error("static collider triangle count changed; fixed topology is required") ? 1 : 0;
    }
    if (triangle_count <= 0) {
        return 1;
    }
    if (!triangles || !solver->static_triangles) {
        return set_error("missing static collider triangles") ? 1 : 0;
    }
    if (!set_cuda_error(
        cudaMemcpy(solver->static_triangles, triangles, sizeof(float) * triangle_count * 9, cudaMemcpyHostToDevice),
        "upload static triangles"
    )) {
        return 0;
    }
    if (solver->static_tri_heads
        && solver->static_tri_entry_next
        && solver->static_tri_entry_index
        && solver->static_tri_entry_count) {
        cudaMemset(solver->static_tri_heads, 0xff, sizeof(int) * solver->static_hash_table_size);
        cudaMemset(solver->static_tri_entry_count, 0, sizeof(int));
        build_static_triangle_hash_kernel<<<block_count(triangle_count), 256>>>(*solver);
        if (!set_cuda_error(cudaDeviceSynchronize(), "rebuild static collision hash")) {
            return 0;
        }
    }
    return 1;
}

extern "C" SSBL_API int ssbl_update_dynamic_triangles(void* handle, const float* triangles, int triangle_count) {
    g_last_error.clear();
    Solver* solver = reinterpret_cast<Solver*>(handle);
    if (!solver) {
        return set_error("invalid dynamic collider update") ? 1 : 0;
    }
    if (triangle_count <= 0) {
        solver->dynamic_triangle_count = 0;
        return 1;
    }
    if (!triangles) {
        return set_error("missing dynamic collider triangles") ? 1 : 0;
    }
    if (!allocate_dynamic_triangle_collision(solver, triangle_count)) {
        return 0;
    }
    if (!set_cuda_error(
        cudaMemcpy(solver->dynamic_triangles, triangles, sizeof(float) * triangle_count * 9, cudaMemcpyHostToDevice),
        "upload dynamic triangles"
    )) {
        return 0;
    }
    return rebuild_dynamic_triangle_hash(solver) ? 1 : 0;
}

extern "C" SSBL_API int ssbl_step_solver(void* handle, int substeps, int iterations) {
    g_last_error.clear();
    Solver* solver = reinterpret_cast<Solver*>(handle);
    if (!solver) {
        return set_error("invalid solver handle") ? 1 : 0;
    }
    substeps = std::max(substeps, 1);
    iterations = std::max(iterations, 1);
    float sub_dt = solver->cfg.dt / static_cast<float>(substeps);
    int v_blocks = block_count(solver->cfg.vertex_count);
    int e_blocks = block_count(solver->cfg.edge_count);
    int b_blocks = block_count(solver->cfg.bend_count);
    int lra_blocks = block_count(solver->cfg.lra_count);
    int t_blocks = block_count(solver->cfg.triangle_count);
    int p_blocks = block_count(solver->pin_count);

    for (int s = 0; s < substeps; ++s) {
        integrate_kernel<<<v_blocks, 256>>>(*solver, sub_dt);
        if (solver->pin_count > 0) {
            pin_project_kernel<<<p_blocks, 256>>>(*solver);
        }
        for (int it = 0; it < iterations; ++it) {
            if (solver->cfg.edge_count > 0) {
                if (solver->cfg.edge_color_count > 0 && solver->edge_color_offsets_host) {
                    for (int color = 0; color < solver->cfg.edge_color_count; ++color) {
                        int start = solver->edge_color_offsets_host[color];
                        int count = solver->edge_color_offsets_host[color + 1] - start;
                        if (count > 0) {
                            edge_project_range_kernel<<<block_count(count), 256>>>(*solver, sub_dt, start, count);
                        }
                    }
                } else {
                    edge_project_kernel<<<e_blocks, 256>>>(*solver, sub_dt);
                }
            }
            if (solver->cfg.bend_count > 0) {
                if (solver->cfg.bend_color_count > 0 && solver->bend_color_offsets_host) {
                    for (int color = 0; color < solver->cfg.bend_color_count; ++color) {
                        int start = solver->bend_color_offsets_host[color];
                        int count = solver->bend_color_offsets_host[color + 1] - start;
                        if (count > 0) {
                            bend_project_range_kernel<<<block_count(count), 256>>>(*solver, sub_dt, start, count);
                        }
                    }
                } else {
                    bend_project_kernel<<<b_blocks, 256>>>(*solver, sub_dt);
                }
            }
            if (solver->cfg.lra_count > 0) {
                lra_project_kernel<<<lra_blocks, 256>>>(*solver, sub_dt);
            }
            if (solver->cfg.use_volume_pressure && solver->volume_gradient && solver->volume_accum && solver->cfg.triangle_count > 0) {
                cudaMemset(solver->volume_gradient, 0, sizeof(Vec3) * solver->cfg.vertex_count);
                cudaMemset(solver->volume_accum, 0, sizeof(float) * 2);
                volume_accumulate_kernel<<<t_blocks, 256>>>(*solver);
                volume_denominator_kernel<<<v_blocks, 256>>>(*solver);
                volume_project_kernel<<<v_blocks, 256>>>(*solver, sub_dt);
            }
            if (solver->pin_count > 0) {
                pin_project_kernel<<<p_blocks, 256>>>(*solver);
            }
            analytic_collision_kernel<<<v_blocks, 256>>>(*solver);
            if (solver->cfg.static_triangle_count > 0) {
                for (int static_pass = 0; static_pass < kStaticCollisionPasses; ++static_pass) {
                    if (solver->cfg.static_triangle_count > kStaticHashTriangleThreshold
                        && solver->static_tri_heads
                        && solver->static_tri_entry_next
                        && solver->static_tri_entry_index
                        && solver->static_tri_entry_count) {
                        static_collision_hashed_kernel<<<v_blocks, 256>>>(*solver);
                    } else {
                        static_collision_kernel<<<v_blocks, 256>>>(*solver);
                    }
                }
            }
            if (solver->dynamic_triangle_count > 0) {
                for (int dynamic_pass = 0; dynamic_pass < kDynamicCollisionPasses; ++dynamic_pass) {
                    if (solver->dynamic_triangle_count > kStaticHashTriangleThreshold
                        && solver->dynamic_tri_heads
                        && solver->dynamic_tri_entry_next
                        && solver->dynamic_tri_entry_index
                        && solver->dynamic_tri_entry_count) {
                        dynamic_collision_hashed_kernel<<<v_blocks, 256>>>(*solver);
                    } else {
                        dynamic_collision_kernel<<<v_blocks, 256>>>(*solver);
                    }
                }
            }
            int interval = std::max(solver->cfg.self_collision_interval, 1);
            bool run_self_collision = solver->cfg.self_collision
                && it == iterations - 1
                && (((s + 1) % interval) == 0 || s == substeps - 1);
            if (run_self_collision && solver->self_vert_heads && solver->self_vert_next) {
                cudaMemset(solver->self_vert_heads, 0xff, sizeof(int) * solver->self_vert_hash_table_size);
                build_self_vertex_hash_kernel<<<v_blocks, 256>>>(*solver);
                self_particle_collision_kernel<<<v_blocks, 256>>>(*solver);
            }
            sanitize_positions_kernel<<<v_blocks, 256>>>(*solver);
        }
        update_velocity_kernel<<<v_blocks, 256>>>(*solver, sub_dt);
    }

    return set_cuda_error(cudaDeviceSynchronize(), "solver step") ? 1 : 0;
}

extern "C" SSBL_API int ssbl_download_positions(void* handle, float* out_positions, int max_floats) {
    g_last_error.clear();
    Solver* solver = reinterpret_cast<Solver*>(handle);
    if (!solver || !out_positions) {
        return set_error("invalid download request") ? 1 : 0;
    }
    int needed = solver->cfg.vertex_count * 3;
    if (max_floats < needed) {
        return set_error("download buffer is too small") ? 1 : 0;
    }
    if (solver->pinned_download && solver->pinned_download_floats >= needed) {
        if (!set_cuda_error(
            cudaMemcpy(solver->pinned_download, solver->pos, sizeof(float) * needed, cudaMemcpyDeviceToHost),
            "download positions"
        )) {
            return 0;
        }
        std::memcpy(out_positions, solver->pinned_download, sizeof(float) * needed);
        return 1;
    }
    return set_cuda_error(cudaMemcpy(out_positions, solver->pos, sizeof(float) * needed, cudaMemcpyDeviceToHost), "download positions") ? 1 : 0;
}

extern "C" SSBL_API const char* ssbl_last_error(void) {
    return g_last_error.c_str();
}
