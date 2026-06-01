#include "ssbl_xpbd_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cfloat>
#include <string>
#include <vector>

namespace {

thread_local std::string g_last_error;

constexpr float kEps = 1.0e-8f;
constexpr float kProjectionRelaxation = 0.35f;
constexpr float kSelfProjectionRelaxation = 0.40f;
constexpr float kSelfRecoveryProjectionRelaxation = 0.70f;
constexpr float kMaxSubstepMove = 0.35f;
constexpr float kMaxVelocity = 35.0f;
constexpr int kSelfCollisionPasses = 2;
constexpr int kSelfSurfaceSamplesPerTriangleDefault = 7;
constexpr int kSelfSurfaceSamplesPerTriangleReduced = 4;
constexpr int kSelfRecoveryPassLimit = 2;
constexpr int kSelfRecoveryCcdClampThreshold = 8;
constexpr int kLargeMeshSelfVertexThreshold = 80000;
constexpr int kLargeMeshSelfTriangleThreshold = 150000;
constexpr float kSelfCoarseDistanceMultiplier = 1.5f;
constexpr float kSelfApproachEps = 1.0e-5f;
constexpr float kSelfContactDistanceEdgeP10Scale = 0.60f;
constexpr int kMaxStaticTriangleHashCells = 256;
constexpr int kMaxStaticVertexQueryCells = 256;
constexpr int kMaxStaticVertexCandidates = 256;
constexpr int kStaticHashTriangleThreshold = 2048;
constexpr int kStaticCollisionPasses = 4;
constexpr int kMaxDynamicTriangleHashCells = 32;
constexpr int kMaxDynamicVertexQueryCells = 64;
constexpr int kMaxDynamicVertexCandidates = 64;
constexpr int kDynamicCollisionPasses = 1;
constexpr int kDiagCountSlots = 4;
constexpr int kDiagCandidateCount = 0;
constexpr int kDiagResolvedContacts = 1;
constexpr int kDiagCcdClampCount = 2;
constexpr int kDiagFiniteFlag = 3;

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
    Vec3* pos_backup = nullptr;
    Vec3* prev_backup = nullptr;
    Vec3* vel_backup = nullptr;
    Vec3* rest = nullptr;
    float* inv_mass = nullptr;
    Vec3* volume_gradient = nullptr;
    float* volume_accum = nullptr;
    Int2* edges = nullptr;
    float* edge_rest = nullptr;
    int* edge_color_offsets_host = nullptr;
    int* vertex_neighbor_offsets = nullptr;
    int* vertex_neighbors = nullptr;
    int vertex_neighbor_count = 0;
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
    int* self_sample_heads = nullptr;
    int* self_sample_next = nullptr;
    int self_sample_hash_table_size = 0;
    int self_sample_count = 0;
    int self_samples_per_triangle = kSelfSurfaceSamplesPerTriangleDefault;
    int self_recovery_mode = 0;
    long long self_collision_run_count = 0;
    float self_contact_distance_value = 0.0f;
    int* pin_indices = nullptr;
    Vec3* pin_targets = nullptr;
    int pin_count = 0;
    int pin_capacity = 0;
    float* pinned_download = nullptr;
    int pinned_download_floats = 0;
    unsigned long long* diag_counts = nullptr;
    float* diag_min_gap = nullptr;
    unsigned long long* probe_counts = nullptr;
    float* probe_min_gap = nullptr;
    unsigned long long* diag_counts_backup = nullptr;
    float* diag_min_gap_backup = nullptr;
    SsblXpbdDiagnostics diag{};
    SsblXpbdDiagnostics probe_diag{};
    float pending_hash_build_ms = 0.0f;
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

float elapsed_ms_since(const std::chrono::high_resolution_clock::time_point& start) {
    const auto now = std::chrono::high_resolution_clock::now();
    return static_cast<float>(std::chrono::duration<double, std::milli>(now - start).count());
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

bool build_vertex_neighbors(Solver* solver, const SsblXpbdConfig* config, const SsblXpbdMesh* mesh) {
    if (!solver || !config || !mesh || config->vertex_count <= 0 || config->edge_count <= 0 || !mesh->edges) {
        return true;
    }
    const int vertex_count = config->vertex_count;
    const int edge_count = config->edge_count;
    const Int2* edges = reinterpret_cast<const Int2*>(mesh->edges);
    std::vector<int> counts(vertex_count, 0);
    for (int i = 0; i < edge_count; ++i) {
        const int a = edges[i].x;
        const int b = edges[i].y;
        if (a < 0 || a >= vertex_count || b < 0 || b >= vertex_count || a == b) {
            continue;
        }
        ++counts[a];
        ++counts[b];
    }
    std::vector<int> offsets(vertex_count + 1, 0);
    for (int i = 0; i < vertex_count; ++i) {
        offsets[i + 1] = offsets[i] + counts[i];
    }
    std::vector<int> cursor(offsets.begin(), offsets.end());
    std::vector<int> neighbors(offsets.back(), -1);
    for (int i = 0; i < edge_count; ++i) {
        const int a = edges[i].x;
        const int b = edges[i].y;
        if (a < 0 || a >= vertex_count || b < 0 || b >= vertex_count || a == b) {
            continue;
        }
        neighbors[cursor[a]++] = b;
        neighbors[cursor[b]++] = a;
    }
    solver->vertex_neighbor_count = static_cast<int>(neighbors.size());
    bool ok = alloc_and_copy(&solver->vertex_neighbor_offsets, offsets.data(), vertex_count + 1, "missing vertex neighbor offsets");
    ok = ok && alloc_and_copy(&solver->vertex_neighbors, neighbors.data(), solver->vertex_neighbor_count, "missing vertex neighbor data");
    return ok;
}

float compute_self_contact_distance(const SsblXpbdConfig* config, const SsblXpbdMesh* mesh) {
    if (!config) {
        return 1.0e-3f;
    }
    float requested = std::max(config->cloth_thickness, 1.0e-3f);
    if (!mesh || !mesh->edge_rest_lengths || config->edge_count <= 0) {
        return requested;
    }
    std::vector<float> edge_lengths(mesh->edge_rest_lengths, mesh->edge_rest_lengths + config->edge_count);
    if (edge_lengths.empty()) {
        return requested;
    }
    size_t p10_index = static_cast<size_t>(std::floor(double(edge_lengths.size() - 1) * 0.10));
    std::nth_element(edge_lengths.begin(), edge_lengths.begin() + p10_index, edge_lengths.end());
    float edge_p10 = std::max(edge_lengths[p10_index], 1.0e-3f);
    float capped = edge_p10 * kSelfContactDistanceEdgeP10Scale;
    float lower = std::max(config->collision_margin * 0.5f, 1.0e-3f);
    return std::clamp(capped, lower, requested);
}

int self_fast_surface_sample_count_per_triangle(const Solver* solver) {
    if (!solver) {
        return kSelfSurfaceSamplesPerTriangleReduced;
    }
    return std::min(
        std::max(solver->self_samples_per_triangle, 1),
        kSelfSurfaceSamplesPerTriangleReduced
    );
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

__device__ void atomic_min_float(float* address, float value) {
    if (!address || !isfinite(value)) {
        return;
    }
    int* address_as_int = reinterpret_cast<int*>(address);
    int old_bits = *address_as_int;
    while (__int_as_float(old_bits) > value) {
        int assumed = old_bits;
        old_bits = atomicCAS(address_as_int, assumed, __float_as_int(value));
        if (old_bits == assumed) {
            break;
        }
    }
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

__device__ float external_contact_distance(Solver solver) {
    return fmaxf(solver.cfg.cloth_thickness, solver.cfg.collision_margin);
}

__device__ float self_contact_distance(Solver solver) {
    return fmaxf(solver.self_contact_distance_value, 1.0e-3f);
}

__device__ float self_cell_size(Solver solver) {
    float thickness = self_contact_distance(solver);
    return fmaxf(thickness * 2.0f, 1.0e-3f);
}

__device__ float static_cell_size(Solver solver) {
    return fmaxf(external_contact_distance(solver) * 2.0f, 1.25e-2f);
}

__device__ void diag_note_gap(Solver solver, float gap) {
    atomic_min_float(solver.diag_min_gap, gap);
}

__device__ void diag_note_effective_candidate(Solver solver) {
    if (solver.diag_counts) {
        atomicAdd(&solver.diag_counts[kDiagCandidateCount], 1ull);
    }
}

__device__ void diag_note_candidate(Solver solver, float gap) {
    diag_note_gap(solver, gap);
    diag_note_effective_candidate(solver);
}

__device__ void diag_note_resolved(Solver solver) {
    if (solver.diag_counts) {
        atomicAdd(&solver.diag_counts[kDiagResolvedContacts], 1ull);
    }
}

__device__ void diag_note_ccd(Solver solver) {
    if (solver.diag_counts) {
        atomicAdd(&solver.diag_counts[kDiagCcdClampCount], 1ull);
    }
}

__device__ void diag_note_nonfinite(Solver solver) {
    if (solver.diag_counts) {
        atomicExch(&solver.diag_counts[kDiagFiniteFlag], 0ull);
    }
}

__device__ bool same_or_one_ring_neighbor(Solver solver, int a, int b) {
    if (a == b) {
        return true;
    }
    if (!solver.vertex_neighbor_offsets || !solver.vertex_neighbors) {
        return false;
    }
    int start = solver.vertex_neighbor_offsets[a];
    int end = solver.vertex_neighbor_offsets[a + 1];
    for (int idx = start; idx < end; ++idx) {
        if (solver.vertex_neighbors[idx] == b) {
            return true;
        }
    }
    return false;
}

__device__ void self_surface_sample_weights(int kind, float* wa, float* wb, float* wc) {
    if (kind == 1) {
        *wa = 0.5f;
        *wb = 0.5f;
        *wc = 0.0f;
    } else if (kind == 2) {
        *wa = 0.0f;
        *wb = 0.5f;
        *wc = 0.5f;
    } else if (kind == 3) {
        *wa = 0.5f;
        *wb = 0.0f;
        *wc = 0.5f;
    } else if (kind == 4) {
        *wa = 2.0f / 3.0f;
        *wb = 1.0f / 6.0f;
        *wc = 1.0f / 6.0f;
    } else if (kind == 5) {
        *wa = 1.0f / 6.0f;
        *wb = 2.0f / 3.0f;
        *wc = 1.0f / 6.0f;
    } else if (kind == 6) {
        *wa = 1.0f / 6.0f;
        *wb = 1.0f / 6.0f;
        *wc = 2.0f / 3.0f;
    } else {
        *wa = 1.0f / 3.0f;
        *wb = 1.0f / 3.0f;
        *wc = 1.0f / 3.0f;
    }
}

__device__ Vec3 weighted_triangle_point(Vec3 a, Vec3 b, Vec3 c, float wa, float wb, float wc) {
    return add(add(mul(a, wa), mul(b, wb)), mul(c, wc));
}

__device__ Vec3 stable_triangle_normal(Vec3 a, Vec3 b, Vec3 c, Vec3 ra, Vec3 rb, Vec3 rc) {
    Vec3 n = cross(sub(b, a), sub(c, a));
    if (dot(n, n) <= kEps) {
        n = cross(sub(rb, ra), sub(rc, ra));
    }
    if (dot(n, n) <= kEps) {
        return {0.0f, 0.0f, 1.0f};
    }
    return normalize(n);
}

__device__ Vec3 self_collision_normal(
    Vec3 delta,
    Vec3 surface_normal,
    Vec3 previous_delta,
    float thickness,
    float* distance_out
) {
    float d2 = dot(delta, delta);
    if (d2 <= kEps) {
        float previous_signed = dot(previous_delta, surface_normal);
        float side = previous_signed >= 0.0f ? 1.0f : -1.0f;
        *distance_out = 0.0f;
        return mul(surface_normal, side);
    }

    float d = sqrtf(fmaxf(d2, 0.0f));
    float signed_distance = dot(delta, surface_normal);
    float abs_signed = fabsf(signed_distance);
    float normal_component_ratio = abs_signed / fmaxf(d, kEps);
    if (normal_component_ratio < 0.35f || abs_signed < thickness * 0.20f) {
        float side = signed_distance >= 0.0f ? 1.0f : -1.0f;
        if (abs_signed <= kEps) {
            float previous_signed = dot(previous_delta, surface_normal);
            side = previous_signed >= 0.0f ? 1.0f : -1.0f;
        }
        *distance_out = abs_signed;
        return mul(surface_normal, side);
    }

    *distance_out = d;
    return mul(delta, 1.0f / d);
}

__device__ bool self_coarse_distance_ok(float distance, float thickness) {
    return distance <= thickness * kSelfCoarseDistanceMultiplier;
}

__device__ bool self_is_approaching(Vec3 delta, Vec3 previous_delta, Vec3 normal) {
    float previous_sep = dot(previous_delta, normal);
    float current_sep = dot(delta, normal);
    return current_sep <= previous_sep - kSelfApproachEps;
}

__device__ float self_projection_relaxation(Solver solver) {
    return solver.self_recovery_mode ? kSelfRecoveryProjectionRelaxation : kSelfProjectionRelaxation;
}

__device__ int self_sample_triangle_index(Solver solver, int sample) {
    int samples_per_triangle = solver.self_samples_per_triangle > 0 ? solver.self_samples_per_triangle : 1;
    return sample / samples_per_triangle;
}

__device__ int self_sample_kind(Solver solver, int sample, int tri_index) {
    int samples_per_triangle = solver.self_samples_per_triangle > 0 ? solver.self_samples_per_triangle : 1;
    return sample - tri_index * samples_per_triangle;
}

__device__ bool rest_surface_neighbor(
    Solver solver,
    int vertex,
    Int3 tri,
    float wa,
    float wb,
    float wc
) {
    if (vertex == tri.x || vertex == tri.y || vertex == tri.z) {
        return true;
    }
    if (same_or_one_ring_neighbor(solver, vertex, tri.x)
        || same_or_one_ring_neighbor(solver, vertex, tri.y)
        || same_or_one_ring_neighbor(solver, vertex, tri.z)) {
        return true;
    }
    float thickness = self_contact_distance(solver);
    float close_threshold = fmaxf(thickness * 0.5f, 1.0e-5f);
    Vec3 rest_sample = weighted_triangle_point(
        solver.rest[tri.x],
        solver.rest[tri.y],
        solver.rest[tri.z],
        wa,
        wb,
        wc
    );
    return norm(sub(solver.rest[vertex], rest_sample)) <= close_threshold
        || norm(sub(solver.rest[vertex], solver.rest[tri.x])) <= close_threshold
        || norm(sub(solver.rest[vertex], solver.rest[tri.y])) <= close_threshold
        || norm(sub(solver.rest[vertex], solver.rest[tri.z])) <= close_threshold;
}

__device__ bool triangles_share_vertex(Int3 a, Int3 b) {
    return a.x == b.x || a.x == b.y || a.x == b.z
        || a.y == b.x || a.y == b.y || a.y == b.z
        || a.z == b.x || a.z == b.y || a.z == b.z;
}

__device__ float weighted_inv_mass(Solver solver, Int3 tri, float wa, float wb, float wc) {
    return wa * wa * solver.inv_mass[tri.x]
        + wb * wb * solver.inv_mass[tri.y]
        + wc * wc * solver.inv_mass[tri.z];
}

__device__ bool rest_samples_neighbor(
    Solver solver,
    Int3 a,
    float aa,
    float ab,
    float ac,
    Int3 b,
    float ba,
    float bb,
    float bc
) {
    if (triangles_share_vertex(a, b)) {
        return true;
    }
    float thickness = self_contact_distance(solver);
    float close_threshold = fmaxf(thickness * 0.35f, 1.0e-5f);
    Vec3 rest_a = weighted_triangle_point(
        solver.rest[a.x],
        solver.rest[a.y],
        solver.rest[a.z],
        aa,
        ab,
        ac
    );
    Vec3 rest_b = weighted_triangle_point(
        solver.rest[b.x],
        solver.rest[b.y],
        solver.rest[b.z],
        ba,
        bb,
        bc
    );
    return norm(sub(rest_a, rest_b)) <= close_threshold;
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
        diag_note_candidate(solver, p.z - target_z);
        diag_note_resolved(solver);
        float correction = target_z - p.z;
        p.z = target_z;
        prev.z += correction;
    }
    if (solver.cfg.use_wall) {
        Vec3 o{solver.cfg.wall_origin[0], solver.cfg.wall_origin[1], solver.cfg.wall_origin[2]};
        Vec3 n = normalize({solver.cfg.wall_normal[0], solver.cfg.wall_normal[1], solver.cfg.wall_normal[2]});
        float d = dot(sub(p, o), n);
        if (d < margin) {
            diag_note_candidate(solver, d - margin);
            diag_note_resolved(solver);
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
            diag_note_candidate(solver, len - radius);
            diag_note_resolved(solver);
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
    float contact_distance,
    Vec3 p,
    Vec3 prev,
    Vec3* projected_out,
    float* score_out,
    float* gap_out,
    int* used_ccd_out
) {
    Vec3 normal = normalize(cross(sub(b, a), sub(c, a)));
    if (!finite_vec(normal)) {
        return false;
    }

    Vec3 closest = closest_point_on_triangle(p, a, b, c);
    Vec3 delta = sub(p, closest);
    float d = norm(delta);

    if (d < contact_distance) {
        float delta_sq = dot(delta, delta);
        if (delta_sq > 1.0e-12f) {
            *projected_out = add(closest, mul(delta, contact_distance / sqrtf(delta_sq)));
        } else {
            float signed_now = dot(delta, normal);
            float signed_prev = dot(sub(prev, closest), normal);
            float side = signed_prev >= 0.0f ? 1.0f : -1.0f;
            if (fabsf(signed_prev) <= kEps && fabsf(signed_now) > kEps) {
                side = signed_now >= 0.0f ? 1.0f : -1.0f;
            }
            *projected_out = add(closest, mul(normal, side * contact_distance));
        }
        *score_out = d;
        *gap_out = d - contact_distance;
        *used_ccd_out = 0;
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
            if (norm(sub(hit, closest_hit)) <= fmaxf(contact_distance * 2.0f, 1.0e-4f)) {
                float side = signed_prev >= 0.0f ? 1.0f : -1.0f;
                *projected_out = add(closest_hit, mul(normal, side * contact_distance));
                *score_out = -1.0f + t;
                *gap_out = -contact_distance;
                *used_ccd_out = 1;
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
    float contact_distance = external_contact_distance(solver);
    bool found = false;
    float best_score = 1.0e30f;
    Vec3 best_projected = p;
    for (int t = 0; t < solver.cfg.static_triangle_count; ++t) {
        Vec3 a = solver.static_triangles[t * 3 + 0];
        Vec3 b = solver.static_triangles[t * 3 + 1];
        Vec3 c = solver.static_triangles[t * 3 + 2];
        Vec3 projected = p;
        float score = 1.0e30f;
        float gap = FLT_MAX;
        int used_ccd = 0;
        if (static_triangle_contact_candidate(
                a, b, c, contact_distance, p, prev, &projected, &score, &gap, &used_ccd
            )) {
            diag_note_candidate(solver, gap);
            if (used_ccd) {
                diag_note_ccd(solver);
            }
        }
        if (score < best_score && score < 1.0e30f) {
            found = true;
            best_score = score;
            best_projected = projected;
        }
    }
    if (found) {
        diag_note_resolved(solver);
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
    float contact_distance = external_contact_distance(solver);
    float expand = fmaxf(contact_distance * 2.0f, cell_size * 0.5f);
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
    float contact_distance = external_contact_distance(solver);
    float cell_size = static_cell_size(solver);
    float expand = fmaxf(contact_distance * 2.0f, 1.0e-4f);
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
                        float gap = FLT_MAX;
                        int used_ccd = 0;
                        if (static_triangle_contact_candidate(
                                a, b, c, contact_distance, p, prev, &projected, &score, &gap, &used_ccd
                            )) {
                            diag_note_candidate(solver, gap);
                            if (used_ccd) {
                                diag_note_ccd(solver);
                            }
                        }
                        if (score < best_score && score < 1.0e30f) {
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
        diag_note_resolved(solver);
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
    float contact_distance = external_contact_distance(solver);
    bool found = false;
    float best_score = 1.0e30f;
    Vec3 best_projected = p;
    for (int t = 0; t < solver.dynamic_triangle_count; ++t) {
        Vec3 a = solver.dynamic_triangles[t * 3 + 0];
        Vec3 b = solver.dynamic_triangles[t * 3 + 1];
        Vec3 c = solver.dynamic_triangles[t * 3 + 2];
        Vec3 projected = p;
        float score = 1.0e30f;
        float gap = FLT_MAX;
        int used_ccd = 0;
        if (static_triangle_contact_candidate(
                a, b, c, contact_distance, p, prev, &projected, &score, &gap, &used_ccd
            )) {
            diag_note_candidate(solver, gap);
            if (used_ccd) {
                diag_note_ccd(solver);
            }
        }
        if (score < best_score && score < 1.0e30f) {
            found = true;
            best_score = score;
            best_projected = projected;
        }
    }
    if (found) {
        diag_note_resolved(solver);
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
    float contact_distance = external_contact_distance(solver);
    float expand = fmaxf(contact_distance * 2.0f, cell_size * 0.5f);
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
    float contact_distance = external_contact_distance(solver);
    float cell_size = static_cell_size(solver);
    float expand = fmaxf(contact_distance * 2.0f, 1.0e-4f);
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
                        float gap = FLT_MAX;
                        int used_ccd = 0;
                        if (static_triangle_contact_candidate(
                                a, b, c, contact_distance, p, prev, &projected, &score, &gap, &used_ccd
                            )) {
                            diag_note_candidate(solver, gap);
                            if (used_ccd) {
                                diag_note_ccd(solver);
                            }
                        }
                        if (score < best_score && score < 1.0e30f) {
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
        diag_note_resolved(solver);
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
    float thickness = self_contact_distance(solver);
    float cell_size = self_cell_size(solver);
    int max_neighbors = solver.cfg.max_self_collision_neighbors > 1 ? solver.cfg.max_self_collision_neighbors : 1;
    if (solver.self_recovery_mode) {
        max_neighbors = min(max_neighbors * 2, 256);
    }
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
                    if (j > i && solver.inv_mass[j] > 0.0f && !same_or_one_ring_neighbor(solver, i, j)) {
                        Vec3 q = solver.pos[j];
                        Vec3 delta = sub(p, q);
                        float d2 = dot(delta, delta);
                        float d = sqrtf(fmaxf(d2, 0.0f));
                        if (!self_coarse_distance_ok(d, thickness)) {
                            j = solver.self_vert_next[j];
                            continue;
                        }
                        float gap = d - thickness;
                        diag_note_gap(solver, gap);
                        Vec3 normal;
                        float contact_distance = d;
                        if (d2 > kEps) {
                            normal = mul(delta, 1.0f / fmaxf(d, kEps));
                        } else {
                            Vec3 rest_delta = sub(solver.rest[i], solver.rest[j]);
                            if (dot(rest_delta, rest_delta) > kEps) {
                                normal = normalize(rest_delta);
                            } else {
                                normal = {0.0f, 0.0f, 1.0f};
                            }
                            contact_distance = 0.0f;
                        }
                        Vec3 previous_delta = sub(solver.prev[i], solver.prev[j]);
                        bool allow_recovery_projection = solver.self_recovery_mode && gap < 0.0f;
                        if (!allow_recovery_projection && !self_is_approaching(delta, previous_delta, normal)) {
                            j = solver.self_vert_next[j];
                            continue;
                        }
                        diag_note_effective_candidate(solver);
                        ++candidates;
                        if (gap < 0.0f) {
                            float wj = solver.inv_mass[j];
                            float total = wi + wj;
                            if (total > 0.0f) {
                                Vec3 correction = mul(normal, self_projection_relaxation(solver) * (thickness - contact_distance) / total);
                                atomic_add(&solver.pos[i], mul(correction, wi));
                                atomic_add(&solver.pos[j], mul(correction, -wj));
                                diag_note_resolved(solver);
                            }
                        }
                    }
                    j = solver.self_vert_next[j];
                }
            }
        }
    }
}

__global__ void build_self_surface_sample_hash_kernel(Solver solver) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= solver.self_sample_count) {
        return;
    }
    int tri_index = self_sample_triangle_index(solver, sample);
    int kind = self_sample_kind(solver, sample, tri_index);
    if (tri_index >= solver.cfg.triangle_count) {
        return;
    }
    Int3 tri = solver.triangles[tri_index];
    float wa;
    float wb;
    float wc;
    self_surface_sample_weights(kind, &wa, &wb, &wc);
    Vec3 p = weighted_triangle_point(solver.pos[tri.x], solver.pos[tri.y], solver.pos[tri.z], wa, wb, wc);
    if (!finite_vec(p)) {
        return;
    }
    float cell_size = self_cell_size(solver);
    int cx = cell_coord(p.x, cell_size);
    int cy = cell_coord(p.y, cell_size);
    int cz = cell_coord(p.z, cell_size);
    int hash = hash_cell(cx, cy, cz, solver.self_sample_hash_table_size);
    solver.self_sample_next[sample] = atomicExch(&solver.self_sample_heads[hash], sample);
}

__global__ void self_vertex_surface_collision_kernel(Solver solver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count || solver.inv_mass[i] <= 0.0f) {
        return;
    }
    Vec3 p = solver.pos[i];
    if (!finite_vec(p)) {
        return;
    }
    float wi = solver.inv_mass[i];
    float margin = solver.cfg.collision_margin;
    float thickness = self_contact_distance(solver);
    float cell_size = self_cell_size(solver);
    int max_neighbors = solver.cfg.max_self_collision_neighbors > 1 ? solver.cfg.max_self_collision_neighbors : 1;
    if (solver.self_recovery_mode) {
        max_neighbors = min(max_neighbors * 2, 256);
    }
    int candidates = 0;
    int scanned = 0;
    int scan_limit = max_neighbors * 32;
    int base_x = cell_coord(p.x, cell_size);
    int base_y = cell_coord(p.y, cell_size);
    int base_z = cell_coord(p.z, cell_size);

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int hash = hash_cell(base_x + dx, base_y + dy, base_z + dz, solver.self_sample_hash_table_size);
                int sample = solver.self_sample_heads[hash];
                while (sample >= 0 && candidates < max_neighbors && scanned < scan_limit) {
                    ++scanned;
                    int tri_index = self_sample_triangle_index(solver, sample);
                    int kind = self_sample_kind(solver, sample, tri_index);
                    Int3 tri = solver.triangles[tri_index];
                    float wa;
                    float wb;
                    float wc;
                    self_surface_sample_weights(kind, &wa, &wb, &wc);
                    if (!rest_surface_neighbor(solver, i, tri, wa, wb, wc)) {
                        Vec3 a = solver.pos[tri.x];
                        Vec3 b = solver.pos[tri.y];
                        Vec3 c = solver.pos[tri.z];
                        Vec3 sample_pos = weighted_triangle_point(a, b, c, wa, wb, wc);
                        Vec3 delta = sub(p, sample_pos);
                        float d2 = dot(delta, delta);
                        float d_linear = sqrtf(fmaxf(d2, 0.0f));
                        if (!self_coarse_distance_ok(d_linear, thickness)) {
                            sample = solver.self_sample_next[sample];
                            continue;
                        }
                        float gap = d_linear - thickness;
                        diag_note_gap(solver, gap);
                        Vec3 surface_normal = stable_triangle_normal(
                            a,
                            b,
                            c,
                            solver.rest[tri.x],
                            solver.rest[tri.y],
                            solver.rest[tri.z]
                        );
                        Vec3 prev_sample = weighted_triangle_point(
                            solver.prev[tri.x],
                            solver.prev[tri.y],
                            solver.prev[tri.z],
                            wa,
                            wb,
                            wc
                        );
                        Vec3 previous_delta = sub(solver.prev[i], prev_sample);
                        float d = 0.0f;
                        Vec3 normal = self_collision_normal(
                            delta,
                            surface_normal,
                            previous_delta,
                            thickness,
                            &d
                        );
                        bool allow_recovery_projection = solver.self_recovery_mode && gap < 0.0f;
                        if (!allow_recovery_projection && !self_is_approaching(delta, previous_delta, normal)) {
                            sample = solver.self_sample_next[sample];
                            continue;
                        }
                        diag_note_effective_candidate(solver);
                        ++candidates;
                        if (gap < 0.0f && dot(previous_delta, surface_normal) * dot(delta, surface_normal) < 0.0f) {
                            diag_note_ccd(solver);
                        }
                        if (gap < 0.0f) {
                            float wx = solver.inv_mass[tri.x];
                            float wy = solver.inv_mass[tri.y];
                            float wz = solver.inv_mass[tri.z];
                            float sample_weight = weighted_inv_mass(solver, tri, wa, wb, wc);
                            float total = wi + sample_weight;
                            if (total > 0.0f) {
                                Vec3 correction = mul(normal, self_projection_relaxation(solver) * (thickness - d) / total);
                                atomic_add(&solver.pos[i], mul(correction, wi));
                                atomic_add(&solver.pos[tri.x], mul(correction, -wx * wa));
                                atomic_add(&solver.pos[tri.y], mul(correction, -wy * wb));
                                atomic_add(&solver.pos[tri.z], mul(correction, -wz * wc));
                                diag_note_resolved(solver);
                            }
                        }
                    }
                    sample = solver.self_sample_next[sample];
                }
            }
        }
    }
}

__global__ void self_surface_sample_collision_kernel(Solver solver) {
    int sample_a = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_a >= solver.self_sample_count) {
        return;
    }
    int tri_index_a = self_sample_triangle_index(solver, sample_a);
    int kind_a = self_sample_kind(solver, sample_a, tri_index_a);
    Int3 tri_a = solver.triangles[tri_index_a];
    float aa;
    float ab;
    float ac;
    self_surface_sample_weights(kind_a, &aa, &ab, &ac);
    Vec3 a0 = solver.pos[tri_a.x];
    Vec3 a1 = solver.pos[tri_a.y];
    Vec3 a2 = solver.pos[tri_a.z];
    Vec3 p = weighted_triangle_point(a0, a1, a2, aa, ab, ac);
    if (!finite_vec(p)) {
        return;
    }
    float sample_weight_a = weighted_inv_mass(solver, tri_a, aa, ab, ac);
    if (sample_weight_a <= 0.0f) {
        return;
    }
    float margin = solver.cfg.collision_margin;
    float thickness = self_contact_distance(solver);
    float cell_size = self_cell_size(solver);
    int max_neighbors = solver.cfg.max_self_collision_neighbors > 1 ? solver.cfg.max_self_collision_neighbors : 1;
    if (solver.self_recovery_mode) {
        max_neighbors = min(max_neighbors * 2, 256);
    }
    int candidates = 0;
    int scanned = 0;
    int scan_limit = max_neighbors * 32;
    int base_x = cell_coord(p.x, cell_size);
    int base_y = cell_coord(p.y, cell_size);
    int base_z = cell_coord(p.z, cell_size);

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int hash = hash_cell(base_x + dx, base_y + dy, base_z + dz, solver.self_sample_hash_table_size);
                int sample_b = solver.self_sample_heads[hash];
                while (sample_b >= 0 && candidates < max_neighbors && scanned < scan_limit) {
                    ++scanned;
                    if (sample_b > sample_a) {
                        int tri_index_b = self_sample_triangle_index(solver, sample_b);
                        int kind_b = self_sample_kind(solver, sample_b, tri_index_b);
                        Int3 tri_b = solver.triangles[tri_index_b];
                        float ba;
                        float bb;
                        float bc;
                        self_surface_sample_weights(kind_b, &ba, &bb, &bc);
                        if (!rest_samples_neighbor(solver, tri_a, aa, ab, ac, tri_b, ba, bb, bc)) {
                            Vec3 b0 = solver.pos[tri_b.x];
                            Vec3 b1 = solver.pos[tri_b.y];
                            Vec3 b2 = solver.pos[tri_b.z];
                            Vec3 q = weighted_triangle_point(b0, b1, b2, ba, bb, bc);
                            Vec3 delta = sub(p, q);
                            float d2 = dot(delta, delta);
                            float d_linear = sqrtf(fmaxf(d2, 0.0f));
                            if (!self_coarse_distance_ok(d_linear, thickness)) {
                                sample_b = solver.self_sample_next[sample_b];
                                continue;
                            }
                            float gap = d_linear - thickness;
                            diag_note_gap(solver, gap);
                            Vec3 surface_normal = stable_triangle_normal(
                                a0,
                                a1,
                                a2,
                                solver.rest[tri_a.x],
                                solver.rest[tri_a.y],
                                solver.rest[tri_a.z]
                            );
                            Vec3 prev_a = weighted_triangle_point(
                                solver.prev[tri_a.x],
                                solver.prev[tri_a.y],
                                solver.prev[tri_a.z],
                                aa,
                                ab,
                                ac
                            );
                            Vec3 prev_b = weighted_triangle_point(
                                solver.prev[tri_b.x],
                                solver.prev[tri_b.y],
                                solver.prev[tri_b.z],
                                ba,
                                bb,
                                bc
                            );
                            Vec3 previous_delta = sub(prev_a, prev_b);
                            float d = 0.0f;
                            Vec3 normal = self_collision_normal(
                                delta,
                                surface_normal,
                                previous_delta,
                                thickness,
                                &d
                            );
                            bool allow_recovery_projection = solver.self_recovery_mode && gap < 0.0f;
                            if (!allow_recovery_projection && !self_is_approaching(delta, previous_delta, normal)) {
                                sample_b = solver.self_sample_next[sample_b];
                                continue;
                            }
                            diag_note_effective_candidate(solver);
                            ++candidates;
                            if (gap < 0.0f && dot(previous_delta, surface_normal) * dot(delta, surface_normal) < 0.0f) {
                                diag_note_ccd(solver);
                            }
                            if (gap < 0.0f) {
                                float sample_weight_b = weighted_inv_mass(solver, tri_b, ba, bb, bc);
                                float total = sample_weight_a + sample_weight_b;
                                if (total > 0.0f) {
                                    Vec3 correction = mul(normal, self_projection_relaxation(solver) * (thickness - d) / total);
                                    atomic_add(&solver.pos[tri_a.x], mul(correction, solver.inv_mass[tri_a.x] * aa));
                                    atomic_add(&solver.pos[tri_a.y], mul(correction, solver.inv_mass[tri_a.y] * ab));
                                    atomic_add(&solver.pos[tri_a.z], mul(correction, solver.inv_mass[tri_a.z] * ac));
                                    atomic_add(&solver.pos[tri_b.x], mul(correction, -solver.inv_mass[tri_b.x] * ba));
                                    atomic_add(&solver.pos[tri_b.y], mul(correction, -solver.inv_mass[tri_b.y] * bb));
                                    atomic_add(&solver.pos[tri_b.z], mul(correction, -solver.inv_mass[tri_b.z] * bc));
                                    diag_note_resolved(solver);
                                }
                            }
                        }
                    }
                    sample_b = solver.self_sample_next[sample_b];
                }
            }
        }
    }
}

__global__ void probe_self_particle_collision_kernel(Solver solver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count || solver.inv_mass[i] <= 0.0f) {
        return;
    }
    Vec3 p = solver.pos[i];
    if (!finite_vec(p)) {
        diag_note_nonfinite(solver);
        return;
    }
    float thickness = self_contact_distance(solver);
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
                    if (j > i && solver.inv_mass[j] > 0.0f && !same_or_one_ring_neighbor(solver, i, j)) {
                        Vec3 q = solver.pos[j];
                        Vec3 delta = sub(p, q);
                        float d2 = dot(delta, delta);
                        float d = sqrtf(fmaxf(d2, 0.0f));
                        if (!self_coarse_distance_ok(d, thickness)) {
                            j = solver.self_vert_next[j];
                            continue;
                        }
                        float gap = d - thickness;
                        diag_note_gap(solver, gap);
                        Vec3 normal;
                        if (d2 > kEps) {
                            normal = mul(delta, 1.0f / fmaxf(d, kEps));
                        } else {
                            Vec3 rest_delta = sub(solver.rest[i], solver.rest[j]);
                            normal = dot(rest_delta, rest_delta) > kEps ? normalize(rest_delta) : Vec3{0.0f, 0.0f, 1.0f};
                        }
                        Vec3 previous_delta = sub(solver.prev[i], solver.prev[j]);
                        if (!self_is_approaching(delta, previous_delta, normal)) {
                            j = solver.self_vert_next[j];
                            continue;
                        }
                        diag_note_effective_candidate(solver);
                        ++candidates;
                    }
                    j = solver.self_vert_next[j];
                }
            }
        }
    }
}

__global__ void probe_self_vertex_surface_collision_kernel(Solver solver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count || solver.inv_mass[i] <= 0.0f) {
        return;
    }
    Vec3 p = solver.pos[i];
    if (!finite_vec(p)) {
        diag_note_nonfinite(solver);
        return;
    }
    float thickness = self_contact_distance(solver);
    float cell_size = self_cell_size(solver);
    int max_neighbors = solver.cfg.max_self_collision_neighbors > 1 ? solver.cfg.max_self_collision_neighbors : 1;
    int candidates = 0;
    int scanned = 0;
    int scan_limit = max_neighbors * 32;
    int base_x = cell_coord(p.x, cell_size);
    int base_y = cell_coord(p.y, cell_size);
    int base_z = cell_coord(p.z, cell_size);

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int hash = hash_cell(base_x + dx, base_y + dy, base_z + dz, solver.self_sample_hash_table_size);
                int sample = solver.self_sample_heads[hash];
                while (sample >= 0 && candidates < max_neighbors && scanned < scan_limit) {
                    ++scanned;
                    int tri_index = self_sample_triangle_index(solver, sample);
                    int kind = self_sample_kind(solver, sample, tri_index);
                    Int3 tri = solver.triangles[tri_index];
                    float wa;
                    float wb;
                    float wc;
                    self_surface_sample_weights(kind, &wa, &wb, &wc);
                    if (!rest_surface_neighbor(solver, i, tri, wa, wb, wc)) {
                        Vec3 a = solver.pos[tri.x];
                        Vec3 b = solver.pos[tri.y];
                        Vec3 c = solver.pos[tri.z];
                        Vec3 sample_pos = weighted_triangle_point(a, b, c, wa, wb, wc);
                        Vec3 delta = sub(p, sample_pos);
                        float d_linear = sqrtf(fmaxf(dot(delta, delta), 0.0f));
                        if (!self_coarse_distance_ok(d_linear, thickness)) {
                            sample = solver.self_sample_next[sample];
                            continue;
                        }
                        float gap = d_linear - thickness;
                        diag_note_gap(solver, gap);
                        Vec3 surface_normal = stable_triangle_normal(
                            a,
                            b,
                            c,
                            solver.rest[tri.x],
                            solver.rest[tri.y],
                            solver.rest[tri.z]
                        );
                        Vec3 prev_sample = weighted_triangle_point(
                            solver.prev[tri.x],
                            solver.prev[tri.y],
                            solver.prev[tri.z],
                            wa,
                            wb,
                            wc
                        );
                        Vec3 previous_delta = sub(solver.prev[i], prev_sample);
                        float contact_distance = 0.0f;
                        Vec3 normal = self_collision_normal(delta, surface_normal, previous_delta, thickness, &contact_distance);
                        if (!self_is_approaching(delta, previous_delta, normal)) {
                            sample = solver.self_sample_next[sample];
                            continue;
                        }
                        diag_note_effective_candidate(solver);
                        ++candidates;
                        if (gap < 0.0f && dot(previous_delta, surface_normal) * dot(delta, surface_normal) < 0.0f) {
                            diag_note_ccd(solver);
                        }
                    }
                    sample = solver.self_sample_next[sample];
                }
            }
        }
    }
}

__global__ void probe_self_surface_sample_collision_kernel(Solver solver) {
    int sample_a = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample_a >= solver.self_sample_count) {
        return;
    }
    int tri_index_a = self_sample_triangle_index(solver, sample_a);
    int kind_a = self_sample_kind(solver, sample_a, tri_index_a);
    Int3 tri_a = solver.triangles[tri_index_a];
    float aa;
    float ab;
    float ac;
    self_surface_sample_weights(kind_a, &aa, &ab, &ac);
    Vec3 a0 = solver.pos[tri_a.x];
    Vec3 a1 = solver.pos[tri_a.y];
    Vec3 a2 = solver.pos[tri_a.z];
    Vec3 p = weighted_triangle_point(a0, a1, a2, aa, ab, ac);
    if (!finite_vec(p)) {
        diag_note_nonfinite(solver);
        return;
    }
    if (weighted_inv_mass(solver, tri_a, aa, ab, ac) <= 0.0f) {
        return;
    }
    float thickness = self_contact_distance(solver);
    float cell_size = self_cell_size(solver);
    int max_neighbors = solver.cfg.max_self_collision_neighbors > 1 ? solver.cfg.max_self_collision_neighbors : 1;
    int candidates = 0;
    int scanned = 0;
    int scan_limit = max_neighbors * 32;
    int base_x = cell_coord(p.x, cell_size);
    int base_y = cell_coord(p.y, cell_size);
    int base_z = cell_coord(p.z, cell_size);

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int hash = hash_cell(base_x + dx, base_y + dy, base_z + dz, solver.self_sample_hash_table_size);
                int sample_b = solver.self_sample_heads[hash];
                while (sample_b >= 0 && candidates < max_neighbors && scanned < scan_limit) {
                    ++scanned;
                    if (sample_b > sample_a) {
                        int tri_index_b = self_sample_triangle_index(solver, sample_b);
                        int kind_b = self_sample_kind(solver, sample_b, tri_index_b);
                        Int3 tri_b = solver.triangles[tri_index_b];
                        float ba;
                        float bb;
                        float bc;
                        self_surface_sample_weights(kind_b, &ba, &bb, &bc);
                        if (!rest_samples_neighbor(solver, tri_a, aa, ab, ac, tri_b, ba, bb, bc)) {
                            Vec3 b0 = solver.pos[tri_b.x];
                            Vec3 b1 = solver.pos[tri_b.y];
                            Vec3 b2 = solver.pos[tri_b.z];
                            Vec3 q = weighted_triangle_point(b0, b1, b2, ba, bb, bc);
                            Vec3 delta = sub(p, q);
                            float d_linear = sqrtf(fmaxf(dot(delta, delta), 0.0f));
                            if (!self_coarse_distance_ok(d_linear, thickness)) {
                                sample_b = solver.self_sample_next[sample_b];
                                continue;
                            }
                            float gap = d_linear - thickness;
                            diag_note_gap(solver, gap);
                            Vec3 surface_normal = stable_triangle_normal(
                                a0,
                                a1,
                                a2,
                                solver.rest[tri_a.x],
                                solver.rest[tri_a.y],
                                solver.rest[tri_a.z]
                            );
                            Vec3 prev_a = weighted_triangle_point(
                                solver.prev[tri_a.x],
                                solver.prev[tri_a.y],
                                solver.prev[tri_a.z],
                                aa,
                                ab,
                                ac
                            );
                            Vec3 prev_b = weighted_triangle_point(
                                solver.prev[tri_b.x],
                                solver.prev[tri_b.y],
                                solver.prev[tri_b.z],
                                ba,
                                bb,
                                bc
                            );
                            Vec3 previous_delta = sub(prev_a, prev_b);
                            float contact_distance = 0.0f;
                            Vec3 normal = self_collision_normal(delta, surface_normal, previous_delta, thickness, &contact_distance);
                            if (!self_is_approaching(delta, previous_delta, normal)) {
                                sample_b = solver.self_sample_next[sample_b];
                                continue;
                            }
                            diag_note_effective_candidate(solver);
                            ++candidates;
                            if (gap < 0.0f && dot(previous_delta, surface_normal) * dot(delta, surface_normal) < 0.0f) {
                                diag_note_ccd(solver);
                            }
                        }
                    }
                    sample_b = solver.self_sample_next[sample_b];
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
        diag_note_nonfinite(solver);
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
        diag_note_nonfinite(solver);
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
    const auto hash_start = std::chrono::high_resolution_clock::now();
    cudaMemset(solver->dynamic_tri_heads, 0xff, sizeof(int) * solver->dynamic_hash_table_size);
    cudaMemset(solver->dynamic_tri_entry_count, 0, sizeof(int));
    build_dynamic_triangle_hash_kernel<<<block_count(solver->dynamic_triangle_count), 256>>>(*solver);
    if (!set_cuda_error(cudaDeviceSynchronize(), "rebuild dynamic collision hash")) {
        return false;
    }
    solver->pending_hash_build_ms += elapsed_ms_since(hash_start);
    return true;
}

bool valid_min_gap(float value) {
    return std::isfinite(value) && value < 1.0e29f;
}

bool reset_diagnostics_buffers(
    unsigned long long* counts_buffer,
    float* min_gap_buffer,
    SsblXpbdDiagnostics* host_diag,
    const char* count_label,
    const char* gap_label
) {
    if (!counts_buffer || !min_gap_buffer || !host_diag) {
        return set_error("diagnostic buffers are not allocated");
    }
    const unsigned long long counts[kDiagCountSlots] = {0ull, 0ull, 0ull, 1ull};
    const float min_gap = FLT_MAX;
    if (!set_cuda_error(
            cudaMemcpy(counts_buffer, counts, sizeof(counts), cudaMemcpyHostToDevice),
            count_label)) {
        return false;
    }
    if (!set_cuda_error(
            cudaMemcpy(min_gap_buffer, &min_gap, sizeof(float), cudaMemcpyHostToDevice),
            gap_label)) {
        return false;
    }
    *host_diag = {};
    host_diag->finite_flag = 1;
    host_diag->min_gap = FLT_MAX;
    return true;
}

bool fetch_diagnostics_buffers(
    unsigned long long* counts_buffer,
    float* min_gap_buffer,
    SsblXpbdDiagnostics* host_diag,
    const char* count_label,
    const char* gap_label
) {
    if (!counts_buffer || !min_gap_buffer || !host_diag) {
        return set_error("diagnostic buffers are not allocated");
    }
    unsigned long long counts[kDiagCountSlots] = {0ull, 0ull, 0ull, 1ull};
    float min_gap = FLT_MAX;
    if (!set_cuda_error(
            cudaMemcpy(counts, counts_buffer, sizeof(counts), cudaMemcpyDeviceToHost),
            count_label)) {
        return false;
    }
    if (!set_cuda_error(
            cudaMemcpy(&min_gap, min_gap_buffer, sizeof(float), cudaMemcpyDeviceToHost),
            gap_label)) {
        return false;
    }
    host_diag->candidate_count = static_cast<long long>(counts[kDiagCandidateCount]);
    host_diag->resolved_contacts = static_cast<long long>(counts[kDiagResolvedContacts]);
    host_diag->ccd_clamp_count = static_cast<long long>(counts[kDiagCcdClampCount]);
    host_diag->finite_flag = counts[kDiagFiniteFlag] != 0ull ? 1 : 0;
    host_diag->min_gap = min_gap;
    return true;
}

bool reset_step_diagnostics(Solver* solver) {
    if (!solver) {
        return set_error("invalid solver");
    }
    return reset_diagnostics_buffers(
        solver->diag_counts,
        solver->diag_min_gap,
        &solver->diag,
        "reset diagnostic counts",
        "reset diagnostic min gap"
    );
}

bool fetch_step_diagnostics(Solver* solver) {
    if (!solver) {
        return set_error("invalid solver");
    }
    return fetch_diagnostics_buffers(
        solver->diag_counts,
        solver->diag_min_gap,
        &solver->diag,
        "download diagnostic counts",
        "download diagnostic min gap"
    );
}

bool reset_probe_diagnostics(Solver* solver) {
    if (!solver) {
        return set_error("invalid solver");
    }
    return reset_diagnostics_buffers(
        solver->probe_counts,
        solver->probe_min_gap,
        &solver->probe_diag,
        "reset probe counts",
        "reset probe min gap"
    );
}

bool fetch_probe_diagnostics(Solver* solver) {
    if (!solver) {
        return set_error("invalid solver");
    }
    return fetch_diagnostics_buffers(
        solver->probe_counts,
        solver->probe_min_gap,
        &solver->probe_diag,
        "download probe counts",
        "download probe min gap"
    );
}

bool backup_solver_state(Solver* solver) {
    if (!solver || !solver->pos_backup || !solver->prev_backup || !solver->vel_backup) {
        return set_error("solver state backup is not allocated");
    }
    const size_t bytes = sizeof(Vec3) * solver->cfg.vertex_count;
    if (!set_cuda_error(cudaMemcpy(solver->pos_backup, solver->pos, bytes, cudaMemcpyDeviceToDevice), "backup positions")) {
        return false;
    }
    if (!set_cuda_error(cudaMemcpy(solver->prev_backup, solver->prev, bytes, cudaMemcpyDeviceToDevice), "backup previous positions")) {
        return false;
    }
    return set_cuda_error(cudaMemcpy(solver->vel_backup, solver->vel, bytes, cudaMemcpyDeviceToDevice), "backup velocities");
}

bool restore_solver_state(Solver* solver) {
    if (!solver || !solver->pos_backup || !solver->prev_backup || !solver->vel_backup) {
        return set_error("solver state backup is not allocated");
    }
    const size_t bytes = sizeof(Vec3) * solver->cfg.vertex_count;
    if (!set_cuda_error(cudaMemcpy(solver->pos, solver->pos_backup, bytes, cudaMemcpyDeviceToDevice), "restore positions")) {
        return false;
    }
    if (!set_cuda_error(cudaMemcpy(solver->prev, solver->prev_backup, bytes, cudaMemcpyDeviceToDevice), "restore previous positions")) {
        return false;
    }
    return set_cuda_error(cudaMemcpy(solver->vel, solver->vel_backup, bytes, cudaMemcpyDeviceToDevice), "restore velocities");
}

bool backup_step_diagnostics_state(Solver* solver) {
    if (!solver || !solver->diag_counts_backup || !solver->diag_min_gap_backup) {
        return set_error("diagnostic backup buffers are not allocated");
    }
    if (!set_cuda_error(
            cudaMemcpy(
                solver->diag_counts_backup,
                solver->diag_counts,
                sizeof(unsigned long long) * kDiagCountSlots,
                cudaMemcpyDeviceToDevice
            ),
            "backup diagnostic counts")) {
        return false;
    }
    return set_cuda_error(
        cudaMemcpy(solver->diag_min_gap_backup, solver->diag_min_gap, sizeof(float), cudaMemcpyDeviceToDevice),
        "backup diagnostic min gap"
    );
}

bool restore_step_diagnostics_state(Solver* solver) {
    if (!solver || !solver->diag_counts_backup || !solver->diag_min_gap_backup) {
        return set_error("diagnostic backup buffers are not allocated");
    }
    if (!set_cuda_error(
            cudaMemcpy(
                solver->diag_counts,
                solver->diag_counts_backup,
                sizeof(unsigned long long) * kDiagCountSlots,
                cudaMemcpyDeviceToDevice
            ),
            "restore diagnostic counts")) {
        return false;
    }
    return set_cuda_error(
        cudaMemcpy(solver->diag_min_gap, solver->diag_min_gap_backup, sizeof(float), cudaMemcpyDeviceToDevice),
        "restore diagnostic min gap"
    );
}

bool run_self_collision_pass(Solver* solver, int v_blocks, bool recovery_mode, bool run_surface_sample_pairs) {
    if (!solver || !solver->self_vert_heads || !solver->self_vert_next) {
        return true;
    }
    Solver collision_solver = *solver;
    collision_solver.self_recovery_mode = recovery_mode ? 1 : 0;
    if (!recovery_mode) {
        collision_solver.self_samples_per_triangle = self_fast_surface_sample_count_per_triangle(solver);
        collision_solver.self_sample_count = solver->cfg.triangle_count * collision_solver.self_samples_per_triangle;
    }
    cudaMemset(solver->self_vert_heads, 0xff, sizeof(int) * solver->self_vert_hash_table_size);
    build_self_vertex_hash_kernel<<<v_blocks, 256>>>(collision_solver);
    self_particle_collision_kernel<<<v_blocks, 256>>>(collision_solver);
    if (solver->self_sample_heads
        && solver->self_sample_next
        && collision_solver.self_sample_count > 0) {
        cudaMemset(solver->self_sample_heads, 0xff, sizeof(int) * solver->self_sample_hash_table_size);
        build_self_surface_sample_hash_kernel<<<block_count(collision_solver.self_sample_count), 256>>>(collision_solver);
        self_vertex_surface_collision_kernel<<<v_blocks, 256>>>(collision_solver);
        if (run_surface_sample_pairs) {
            cudaMemset(solver->self_sample_heads, 0xff, sizeof(int) * solver->self_sample_hash_table_size);
            build_self_surface_sample_hash_kernel<<<block_count(collision_solver.self_sample_count), 256>>>(collision_solver);
            self_surface_sample_collision_kernel<<<block_count(collision_solver.self_sample_count), 256>>>(collision_solver);
        }
    }
    sanitize_positions_kernel<<<v_blocks, 256>>>(collision_solver);
    return set_cuda_error(cudaGetLastError(), "launch self collision pass");
}

bool probe_self_collision(Solver* solver, int v_blocks, SsblXpbdDiagnostics* out_diag) {
    if (!solver || !out_diag) {
        return set_error("invalid self-collision probe");
    }
    if (!solver->self_vert_heads || !solver->self_vert_next) {
        *out_diag = {};
        out_diag->finite_flag = 1;
        out_diag->min_gap = FLT_MAX;
        return true;
    }
    Solver probe_solver = *solver;
    probe_solver.diag_counts = solver->probe_counts;
    probe_solver.diag_min_gap = solver->probe_min_gap;
    probe_solver.self_samples_per_triangle = self_fast_surface_sample_count_per_triangle(solver);
    probe_solver.self_sample_count = solver->cfg.triangle_count * probe_solver.self_samples_per_triangle;
    if (!reset_probe_diagnostics(solver)) {
        return false;
    }
    cudaMemset(solver->self_vert_heads, 0xff, sizeof(int) * solver->self_vert_hash_table_size);
    build_self_vertex_hash_kernel<<<v_blocks, 256>>>(probe_solver);
    probe_self_particle_collision_kernel<<<v_blocks, 256>>>(probe_solver);
    if (solver->self_sample_heads
        && solver->self_sample_next
        && probe_solver.self_sample_count > 0) {
        cudaMemset(solver->self_sample_heads, 0xff, sizeof(int) * solver->self_sample_hash_table_size);
        build_self_surface_sample_hash_kernel<<<block_count(probe_solver.self_sample_count), 256>>>(probe_solver);
        probe_self_vertex_surface_collision_kernel<<<v_blocks, 256>>>(probe_solver);
    }
    if (!set_cuda_error(cudaGetLastError(), "launch self-collision probe")) {
        return false;
    }
    if (!fetch_probe_diagnostics(solver)) {
        return false;
    }
    *out_diag = solver->probe_diag;
    return true;
}

bool self_probe_triggers_recovery(const SsblXpbdDiagnostics& diag, float cloth_thickness) {
    const float trigger_gap = -0.25f * cloth_thickness;
    return (valid_min_gap(diag.min_gap) && diag.min_gap < trigger_gap)
        || diag.ccd_clamp_count >= kSelfRecoveryCcdClampThreshold;
}

bool self_probe_triggers_retry(const SsblXpbdDiagnostics& diag, float cloth_thickness) {
    const float retry_gap = -0.5f * cloth_thickness;
    return valid_min_gap(diag.min_gap) && diag.min_gap < retry_gap;
}

bool run_substep(
    Solver* solver,
    float sub_dt,
    int iterations,
    bool run_self_collision,
    bool run_volume_pressure,
    int v_blocks,
    int e_blocks,
    int b_blocks,
    int lra_blocks,
    int t_blocks,
    int p_blocks,
    long long* recovery_passes_total,
    long long* local_retry_total,
    bool allow_retry
) {
    if (!solver) {
        return set_error("invalid solver");
    }
    if (allow_retry && run_self_collision) {
        if (!backup_solver_state(solver) || !backup_step_diagnostics_state(solver)) {
            return false;
        }
    }

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
        if (run_volume_pressure && solver->volume_gradient && solver->volume_accum && solver->cfg.triangle_count > 0) {
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
        if (run_self_collision && it == iterations - 1) {
            ++solver->self_collision_run_count;
            int surface_pair_interval = std::max(solver->cfg.self_surface_pair_interval, 1);
            bool run_surface_pairs = (solver->self_collision_run_count % surface_pair_interval) == 0;
            for (int self_pass = 0; self_pass < kSelfCollisionPasses; ++self_pass) {
                bool run_surface_sample_pairs = run_surface_pairs && (self_pass == kSelfCollisionPasses - 1);
                if (!run_self_collision_pass(solver, v_blocks, false, run_surface_sample_pairs)) {
                    return false;
                }
            }

            int probe_interval = std::max(solver->cfg.self_probe_interval, 1);
            bool run_probe = (solver->self_collision_run_count % probe_interval) == 0;
            if (!run_probe) {
                sanitize_positions_kernel<<<v_blocks, 256>>>(*solver);
                continue;
            }

            SsblXpbdDiagnostics probe_diag{};
            if (!probe_self_collision(solver, v_blocks, &probe_diag)) {
                return false;
            }
            float cloth_thickness = std::max(solver->cfg.cloth_thickness, 1.0e-4f);
            int extra_recovery_passes = 0;
            while (extra_recovery_passes < kSelfRecoveryPassLimit
                && self_probe_triggers_recovery(probe_diag, cloth_thickness)) {
                if (!run_self_collision_pass(solver, v_blocks, true, true)) {
                    return false;
                }
                ++extra_recovery_passes;
                if (recovery_passes_total) {
                    ++(*recovery_passes_total);
                }
                if (!probe_self_collision(solver, v_blocks, &probe_diag)) {
                    return false;
                }
            }

            if (allow_retry && self_probe_triggers_retry(probe_diag, cloth_thickness)) {
                if (local_retry_total) {
                    ++(*local_retry_total);
                }
                if (!restore_solver_state(solver) || !restore_step_diagnostics_state(solver)) {
                    return false;
                }
                float half_dt = sub_dt * 0.5f;
                if (!run_substep(
                        solver,
                        half_dt,
                        iterations,
                        run_self_collision,
                        run_volume_pressure,
                        v_blocks,
                        e_blocks,
                        b_blocks,
                        lra_blocks,
                        t_blocks,
                        p_blocks,
                        recovery_passes_total,
                        local_retry_total,
                        false)) {
                    return false;
                }
                return run_substep(
                    solver,
                    half_dt,
                    iterations,
                    run_self_collision,
                    run_volume_pressure,
                    v_blocks,
                    e_blocks,
                    b_blocks,
                    lra_blocks,
                    t_blocks,
                    p_blocks,
                    recovery_passes_total,
                    local_retry_total,
                    false
                );
            }
        }
        sanitize_positions_kernel<<<v_blocks, 256>>>(*solver);
    }
    update_velocity_kernel<<<v_blocks, 256>>>(*solver, sub_dt);
    return set_cuda_error(cudaGetLastError(), "launch substep");
}

void free_solver(Solver* solver) {
    if (!solver) {
        return;
    }
    cudaFree(solver->pos);
    cudaFree(solver->prev);
    cudaFree(solver->vel);
    cudaFree(solver->pos_backup);
    cudaFree(solver->prev_backup);
    cudaFree(solver->vel_backup);
    cudaFree(solver->rest);
    cudaFree(solver->inv_mass);
    cudaFree(solver->volume_gradient);
    cudaFree(solver->volume_accum);
    cudaFree(solver->edges);
    cudaFree(solver->edge_rest);
    delete[] solver->edge_color_offsets_host;
    cudaFree(solver->vertex_neighbor_offsets);
    cudaFree(solver->vertex_neighbors);
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
    cudaFree(solver->self_sample_heads);
    cudaFree(solver->self_sample_next);
    cudaFree(solver->pin_indices);
    cudaFree(solver->pin_targets);
    cudaFreeHost(solver->pinned_download);
    cudaFree(solver->diag_counts);
    cudaFree(solver->diag_min_gap);
    cudaFree(solver->probe_counts);
    cudaFree(solver->probe_min_gap);
    cudaFree(solver->diag_counts_backup);
    cudaFree(solver->diag_min_gap_backup);
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
    solver->self_contact_distance_value = compute_self_contact_distance(config, mesh);
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
    if (ok) {
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&solver->pos_backup), sizeof(Vec3) * vertex_count);
        ok = ok && set_cuda_error(err, "position backup allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->prev_backup), sizeof(Vec3) * vertex_count);
        ok = ok && set_cuda_error(err, "previous-position backup allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->vel_backup), sizeof(Vec3) * vertex_count);
        ok = ok && set_cuda_error(err, "velocity backup allocation");
    }
    if (ok) {
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&solver->diag_counts), sizeof(unsigned long long) * kDiagCountSlots);
        ok = ok && set_cuda_error(err, "diagnostic count allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->diag_min_gap), sizeof(float));
        ok = ok && set_cuda_error(err, "diagnostic gap allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->probe_counts), sizeof(unsigned long long) * kDiagCountSlots);
        ok = ok && set_cuda_error(err, "probe count allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->probe_min_gap), sizeof(float));
        ok = ok && set_cuda_error(err, "probe gap allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->diag_counts_backup), sizeof(unsigned long long) * kDiagCountSlots);
        ok = ok && set_cuda_error(err, "diagnostic backup count allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->diag_min_gap_backup), sizeof(float));
        ok = ok && set_cuda_error(err, "diagnostic backup gap allocation");
        solver->diag.finite_flag = 1;
        solver->diag.min_gap = FLT_MAX;
        solver->probe_diag.finite_flag = 1;
        solver->probe_diag.min_gap = FLT_MAX;
    }
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
    ok = ok && build_vertex_neighbors(solver, config, mesh);
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
            const auto hash_start = std::chrono::high_resolution_clock::now();
            cudaMemset(solver->static_tri_heads, 0xff, sizeof(int) * solver->static_hash_table_size);
            cudaMemset(solver->static_tri_entry_count, 0, sizeof(int));
            build_static_triangle_hash_kernel<<<block_count(config->static_triangle_count), 256>>>(*solver);
            ok = ok && set_cuda_error(cudaDeviceSynchronize(), "static collision hash build");
            if (ok) {
                solver->pending_hash_build_ms += elapsed_ms_since(hash_start);
            }
        }
    }

    if (ok && config->self_collision && config->vertex_count > 0) {
        solver->self_samples_per_triangle =
            (config->vertex_count > kLargeMeshSelfVertexThreshold || config->triangle_count > kLargeMeshSelfTriangleThreshold)
            ? kSelfSurfaceSamplesPerTriangleReduced
            : kSelfSurfaceSamplesPerTriangleDefault;
        solver->self_vert_hash_table_size = next_power_of_two(std::max(1024, config->vertex_count * 2));
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&solver->self_vert_heads), sizeof(int) * solver->self_vert_hash_table_size);
        ok = ok && set_cuda_error(err, "self vertex hash allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->self_vert_next), sizeof(int) * config->vertex_count);
        ok = ok && set_cuda_error(err, "self vertex link allocation");
        if (ok && config->triangle_count > 0) {
            solver->self_sample_count = config->triangle_count * solver->self_samples_per_triangle;
            solver->self_sample_hash_table_size = next_power_of_two(std::max(1024, solver->self_sample_count * 2));
            err = cudaMalloc(reinterpret_cast<void**>(&solver->self_sample_heads), sizeof(int) * solver->self_sample_hash_table_size);
            ok = ok && set_cuda_error(err, "self surface sample hash allocation");
            err = cudaMalloc(reinterpret_cast<void**>(&solver->self_sample_next), sizeof(int) * solver->self_sample_count);
            ok = ok && set_cuda_error(err, "self surface sample link allocation");
        }
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
        const auto hash_start = std::chrono::high_resolution_clock::now();
        cudaMemset(solver->static_tri_heads, 0xff, sizeof(int) * solver->static_hash_table_size);
        cudaMemset(solver->static_tri_entry_count, 0, sizeof(int));
        build_static_triangle_hash_kernel<<<block_count(triangle_count), 256>>>(*solver);
        if (!set_cuda_error(cudaDeviceSynchronize(), "rebuild static collision hash")) {
            return 0;
        }
        solver->pending_hash_build_ms += elapsed_ms_since(hash_start);
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
    if (!reset_step_diagnostics(solver)) {
        return 0;
    }
    substeps = std::max(substeps, 1);
    iterations = std::max(iterations, 1);
    const auto step_start = std::chrono::high_resolution_clock::now();
    const float pending_hash_build_ms = solver->pending_hash_build_ms;
    solver->pending_hash_build_ms = 0.0f;
    float sub_dt = solver->cfg.dt / static_cast<float>(substeps);
    int v_blocks = block_count(solver->cfg.vertex_count);
    int e_blocks = block_count(solver->cfg.edge_count);
    int b_blocks = block_count(solver->cfg.bend_count);
    int lra_blocks = block_count(solver->cfg.lra_count);
    int t_blocks = block_count(solver->cfg.triangle_count);
    int p_blocks = block_count(solver->pin_count);
    long long recovery_passes_total = 0;
    long long local_retry_total = 0;

    for (int s = 0; s < substeps; ++s) {
        int interval = std::max(solver->cfg.self_collision_interval, 1);
        bool run_self_collision = solver->cfg.self_collision
            && (((s + 1) % interval) == 0 || s == substeps - 1);
        int volume_interval = std::max(solver->cfg.volume_solve_interval, 1);
        bool run_volume_pressure = solver->cfg.use_volume_pressure
            && (((s + 1) % volume_interval) == 0 || s == substeps - 1);
        if (!run_substep(
                solver,
                sub_dt,
                iterations,
                run_self_collision,
                run_volume_pressure,
                v_blocks,
                e_blocks,
                b_blocks,
                lra_blocks,
                t_blocks,
                p_blocks,
                &recovery_passes_total,
                &local_retry_total,
                true)) {
            return 0;
        }
    }

    if (!set_cuda_error(cudaDeviceSynchronize(), "solver step")) {
        return 0;
    }
    solver->diag.step_ms = elapsed_ms_since(step_start);
    solver->diag.hash_build_ms = pending_hash_build_ms;
    if (!fetch_step_diagnostics(solver)) {
        return 0;
    }
    solver->diag.recovery_passes = recovery_passes_total;
    solver->diag.local_retry_count = local_retry_total;
    return 1;
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

extern "C" SSBL_API int ssbl_get_diagnostics(void* handle, SsblXpbdDiagnostics* out_diag) {
    g_last_error.clear();
    Solver* solver = reinterpret_cast<Solver*>(handle);
    if (!solver || !out_diag) {
        return set_error("invalid diagnostics request") ? 1 : 0;
    }
    *out_diag = solver->diag;
    return 1;
}

extern "C" SSBL_API const char* ssbl_last_error(void) {
    return g_last_error.c_str();
}
