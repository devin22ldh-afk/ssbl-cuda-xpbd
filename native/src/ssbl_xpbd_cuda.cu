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
constexpr float kSelfRecoveryProjectionRelaxation = 0.55f;
constexpr float kFastSelfProjectionRelaxation = 0.85f;
constexpr float kFastSelfCleanupProjectionRelaxation = 1.35f;
constexpr float kFastSelfRecoveryProjectionRelaxation = 0.90f;
constexpr float kSelfRecoveryVelocityDamping = 0.65f;
constexpr float kSelfCorrectionMaxDisplacementScale = 0.45f;
constexpr float kSelfRecoveryMaxDisplacementScale = 0.50f;
constexpr float kMaxSubstepMove = 0.35f;
constexpr float kMaxVelocity = 35.0f;
constexpr int kSelfCollisionPasses = 2;
constexpr int kSelfSurfaceSamplesPerTriangleDefault = 7;
constexpr int kSelfSurfaceSamplesPerTriangleReduced = 4;
constexpr int kSelfRecoveryPassLimit = 2;
constexpr int kLargeMeshSelfVertexThreshold = 80000;
constexpr int kLargeMeshSelfTriangleThreshold = 150000;
constexpr float kSelfCoarseDistanceMultiplier = 2.5f;
constexpr float kSelfApproachEps = 1.0e-5f;
constexpr float kSelfContactDistanceEdgeP10Scale = 0.60f;
constexpr int kFastSelfCollisionPasses = 2;
constexpr int kFastSelfTriangleCleanupRuns = 0;
constexpr int kSelfSleepTargetVerticesPerRegion = 512;
constexpr int kSelfSleepMaxRegions = 1024;
constexpr int kSelfSourceFull = 0;
constexpr int kSelfSourceActive = 1;
constexpr int kSelfSourceSuspect = 2;
constexpr int kSelfCollisionModeFast = 1;
constexpr int kSelfCollisionModeStrict = 2;
constexpr int kSelfCompactionCountSlots = 6;
constexpr int kSelfCompactionActiveVertices = 0;
constexpr int kSelfCompactionSuspectVertices = 1;
constexpr int kSelfCompactionActiveSamples = 2;
constexpr int kSelfCompactionSuspectSamples = 3;
constexpr int kSelfCompactionActiveRegions = 4;
constexpr int kSelfCompactionSuspectRegions = 5;
constexpr int kSelfVsPairCountSlots = 2;
constexpr int kSelfVsPairCount = 0;
constexpr int kSelfVsPairOverflow = 1;
constexpr int kSelfVsPairCapacityMin = 262144;
constexpr int kSelfVsPairCapacityMax = 8388608;
constexpr int kMaxSelfTriangleHashCells = 32;
constexpr int kMaxSelfVertexTriangleQueryCells = 64;
constexpr int kMaxSelfVertexTriangleCandidates = 128;
constexpr int kSelfTriangleEntryCapacityMax = 8388608;
constexpr int kJitterCountSlots = 2;
constexpr int kJitterStabilizedCount = 0;
constexpr int kJitterRejectedCount = 1;
constexpr int kJitterScoreThreshold = 2;
constexpr float kJitterDeltaThresholdScale = 0.35f;
constexpr float kJitterCorrectionScale = 0.25f;
constexpr float kJitterCorrectionLimitScale = 0.20f;
constexpr float kJitterStretchResidualTolerance = 1.005f;
constexpr float kJitterAreaResidualTolerance = 1.005f;
constexpr float kJitterAreaPeakTolerance = 1.0005f;
constexpr float kJitterRecoveryOpposeLimit = -0.25f;
constexpr int kMaxForceFields = 64;
constexpr int kForceFieldWind = 1;
constexpr int kForceFieldForce = 2;
constexpr int kForceFieldVortex = 3;
constexpr int kForceFieldTurbulence = 4;
constexpr int kForceFieldCharge = 5;
constexpr int kForceFieldHarmonic = 6;
constexpr int kForceFieldLennardJ = 7;
constexpr int kForceFieldMagnet = 8;
constexpr int kForceFieldDrag = 9;
constexpr int kForceFieldTexture = 10;
constexpr float kMaxForceFieldAcceleration = 5000.0f;
constexpr int kMaxStaticTriangleHashCells = 256;
constexpr int kMaxStaticVertexQueryCells = 256;
constexpr int kMaxStaticVertexCandidates = 256;
constexpr int kStaticHashTriangleThreshold = 2048;
constexpr int kStaticCollisionPasses = 4;
constexpr int kMaxDynamicTriangleHashCells = 32;
constexpr int kMaxDynamicVertexQueryCells = 64;
constexpr int kMaxDynamicVertexCandidates = 64;
constexpr int kMaxDynamicParticleQueryCells = 125;
constexpr int kMaxDynamicParticleNeighbors = 64;
constexpr float kDynamicParticleCollisionRelaxation = 0.8f;
constexpr float kDynamicParticleMaxCorrectionScale = 1.0f;
constexpr int kMaxDynamicCollisionPasses = 2;
constexpr int kDynamicCollisionTwoPassTriangleLimit = 128;
constexpr long long kDynamicCollisionTwoPassWorkLimit = 50000000;
constexpr float kExternalCollisionRelaxation = 0.75f;
constexpr float kExternalCollisionCcdRelaxation = 0.75f;
constexpr float kStaticCollisionMaxCorrectionScale = 1.0f;
constexpr float kStaticCollisionCcdMaxCorrectionScale = 1.0f;
constexpr float kDynamicCollisionMaxCorrectionScale = 0.5f;
constexpr float kDynamicCollisionCcdMaxCorrectionScale = 1.0f;
constexpr float kExternalCollisionMinCorrection = 1.0e-4f;
constexpr float kExternalCollisionInwardVelocityDamping = 1.0f;
constexpr int kExternalContactKindStatic = 1;
constexpr int kExternalContactKindDynamic = 2;
constexpr int kExternalContactProbeLimit = 16;
constexpr int kExternalContactMaxAge = 8;
constexpr int kExternalContactCapacityMin = 4096;
constexpr int kExternalContactCapacityMax = 1048576;
constexpr float kExternalContactWarmStartScale = 0.35f;
constexpr int kDiagCountSlots = 15;
constexpr int kDiagCandidateCount = 0;
constexpr int kDiagResolvedContacts = 1;
constexpr int kDiagCcdClampCount = 2;
constexpr int kDiagFiniteFlag = 3;
constexpr int kDiagSelfSkippedSources = 4;
constexpr int kDiagSelfActiveRegions = 5;
constexpr int kDiagSelfSleepingRegions = 6;
constexpr int kDiagExternalContactCacheHits = 7;
constexpr int kDiagExternalContactCacheMisses = 8;
constexpr int kDiagExternalContactCacheOverflow = 9;
constexpr int kDiagExternalFrictionCorrections = 10;
constexpr int kDiagExternalContactCacheCount = 11;
constexpr int kDiagDynamicParticleCandidates = 12;
constexpr int kDiagDynamicParticleContacts = 13;
constexpr int kDiagDynamicParticleOverflow = 14;

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

struct ExternalContact {
    unsigned long long key;
    Vec3 normal;
    Vec3 barycentric;
    float lambda_n;
    Vec3 lambda_t;
    int age;
    int active;
};

struct DynamicParticle {
    Vec3 position;
    float radius;
    float inv_mass;
    int slot_id;
    int phase;
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
    int* volume_vertex_offsets = nullptr;
    int* volume_vertex_triangles = nullptr;
    int volume_vertex_triangle_count = 0;
    float* volume_partial_values = nullptr;
    float* volume_partial_denominators = nullptr;
    int volume_partial_capacity = 0;
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
    int static_collider_complex = 0;
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
    DynamicParticle* dynamic_particles = nullptr;
    int dynamic_particle_count = 0;
    int dynamic_particle_capacity = 0;
    float dynamic_particle_max_radius = 0.0f;
    int* dynamic_particle_heads = nullptr;
    int* dynamic_particle_next = nullptr;
    int* dynamic_particle_index = nullptr;
    int* dynamic_particle_count_buffer = nullptr;
    int dynamic_particle_hash_table_size = 0;
    ExternalContact* external_contacts = nullptr;
    int external_contact_capacity = 0;
    SsblXpbdForceField* force_fields = nullptr;
    int force_field_count = 0;
    int force_field_capacity = 0;
    int unsupported_force_field_count = 0;
    int* self_vert_heads = nullptr;
    int* self_vert_next = nullptr;
    int self_vert_hash_table_size = 0;
    int* self_edge_heads = nullptr;
    int* self_edge_next = nullptr;
    int self_edge_hash_table_size = 0;
    int* self_sample_heads = nullptr;
    int* self_sample_next = nullptr;
    int* self_sample_hash_dirty = nullptr;
    int* self_tri_heads = nullptr;
    int* self_tri_entry_next = nullptr;
    int* self_tri_entry_index = nullptr;
    int* self_tri_entry_count = nullptr;
    int self_tri_hash_table_size = 0;
    int self_tri_entry_capacity = 0;
    int* self_recovery_touched = nullptr;
    Vec3* self_recovery_delta = nullptr;
    int* self_sleep_vertex_regions = nullptr;
    int* self_sleep_triangle_regions = nullptr;
    Vec3* self_sleep_prev_pos = nullptr;
    int* self_sleep_region_still_frames = nullptr;
    int* self_sleep_region_sleeping = nullptr;
    int* self_sleep_region_motion = nullptr;
    int* self_sleep_region_touched = nullptr;
    int* self_sleep_region_vertex_counts = nullptr;
    int* self_sleep_has_sleeping = nullptr;
    int self_sleep_region_count = 0;
    int self_sleep_dim_x = 1;
    int self_sleep_dim_y = 1;
    int self_sleep_dim_z = 1;
    int self_sleep_force_active = 0;
    long long self_sleep_frame_count = 0;
    int* self_active_vertices = nullptr;
    int* self_suspect_vertices = nullptr;
    int* self_active_vertex_flags = nullptr;
    int* self_suspect_vertex_flags = nullptr;
    int* self_active_samples = nullptr;
    int* self_suspect_samples = nullptr;
    int* self_compaction_counts = nullptr;
    int self_active_vertex_count = 0;
    int self_suspect_vertex_count = 0;
    int self_active_sample_count = 0;
    int self_suspect_sample_count = 0;
    int self_suspect_region_count = 0;
    int self_compaction_used = 0;
    int self_compaction_samples_per_triangle = 0;
    int self_source_mode = kSelfSourceFull;
    long long self_full_recovery_fallbacks = 0;
    Int2* self_vs_pairs = nullptr;
    int* self_vs_pair_counts = nullptr;
    int self_vs_pair_capacity = 0;
    int self_vs_pair_count = 0;
    int self_vs_pair_overflow = 0;
    int self_vs_pair_current_overflow = 0;
    int self_vs_pair_compaction_used = 0;
    int self_vs_pair_valid = 0;
    int self_sample_hash_table_size = 0;
    int self_sample_count = 0;
    int self_sample_hash_valid = 0;
    int self_samples_per_triangle = kSelfSurfaceSamplesPerTriangleDefault;
    int self_recovery_mode = 0;
    int self_cleanup_mode = 0;
    long long self_collision_run_count = 0;
    float self_contact_distance_value = 0.0f;
    Vec3* jitter_frame_start_pos = nullptr;
    Vec3* jitter_prev_delta = nullptr;
    int* jitter_score = nullptr;
    unsigned long long* jitter_counts = nullptr;
    float* jitter_max_correction = nullptr;
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

enum TimingSlot {
    kTimingConstraints = 0,
    kTimingVolume,
    kTimingAnalyticCollision,
    kTimingStaticCollision,
    kTimingDynamicCollision,
    kTimingDynamicParticleCollision,
    kTimingSelfHash,
    kTimingSelfSolve,
    kTimingSelfProbe,
    kTimingSelfRecovery,
    kTimingSelfVsPairBuild,
    kTimingSelfVsPairProjectSolve,
    kTimingSelfVsPairProjectProbe,
    kTimingSelfVsPairProjectRecovery,
    kTimingCount
};

struct TimedSegment {
    int slot = 0;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
};

float* timing_field(SsblXpbdDiagnostics* diag, int slot) {
    if (!diag) {
        return nullptr;
    }
    switch (slot) {
        case kTimingConstraints:
            return &diag->constraints_ms;
        case kTimingVolume:
            return &diag->volume_ms;
        case kTimingAnalyticCollision:
            return &diag->analytic_collision_ms;
        case kTimingStaticCollision:
            return &diag->static_collision_ms;
        case kTimingDynamicCollision:
            return &diag->dynamic_collision_ms;
        case kTimingDynamicParticleCollision:
            return &diag->dynamic_particle_collision_ms;
        case kTimingSelfHash:
            return &diag->self_hash_ms;
        case kTimingSelfSolve:
            return &diag->self_solve_ms;
        case kTimingSelfProbe:
            return &diag->self_probe_ms;
        case kTimingSelfRecovery:
            return &diag->self_recovery_ms;
        case kTimingSelfVsPairBuild:
            return &diag->self_vs_pair_build_ms;
        case kTimingSelfVsPairProjectSolve:
        case kTimingSelfVsPairProjectProbe:
        case kTimingSelfVsPairProjectRecovery:
            return &diag->self_vs_pair_project_ms;
        default:
            return nullptr;
    }
}

float* timing_primary_field(SsblXpbdDiagnostics* diag, int slot) {
    if (!diag) {
        return nullptr;
    }
    switch (slot) {
        case kTimingSelfVsPairProjectSolve:
            return &diag->self_solve_ms;
        case kTimingSelfVsPairProjectRecovery:
            return &diag->self_recovery_ms;
        case kTimingDynamicParticleCollision:
            return &diag->dynamic_collision_ms;
        default:
            return nullptr;
    }
}

bool begin_timed_segment(std::vector<TimedSegment>* timings, int slot, TimedSegment* segment, const char* label) {
    if (!timings || !segment) {
        return true;
    }
    *segment = {};
    segment->slot = slot;
    cudaError_t err = cudaEventCreate(&segment->start);
    if (!set_cuda_error(err, label)) {
        return false;
    }
    err = cudaEventCreate(&segment->stop);
    if (!set_cuda_error(err, label)) {
        cudaEventDestroy(segment->start);
        segment->start = nullptr;
        return false;
    }
    err = cudaEventRecord(segment->start, 0);
    if (!set_cuda_error(err, label)) {
        cudaEventDestroy(segment->start);
        cudaEventDestroy(segment->stop);
        *segment = {};
        return false;
    }
    return true;
}

bool end_timed_segment(std::vector<TimedSegment>* timings, TimedSegment* segment, const char* label) {
    if (!timings || !segment || !segment->start || !segment->stop) {
        return true;
    }
    cudaError_t err = cudaEventRecord(segment->stop, 0);
    if (!set_cuda_error(err, label)) {
        cudaEventDestroy(segment->start);
        cudaEventDestroy(segment->stop);
        *segment = {};
        return false;
    }
    timings->push_back(*segment);
    *segment = {};
    return true;
}

void destroy_timing_records(std::vector<TimedSegment>* timings) {
    if (!timings) {
        return;
    }
    for (TimedSegment& segment : *timings) {
        if (segment.start) {
            cudaEventDestroy(segment.start);
        }
        if (segment.stop) {
            cudaEventDestroy(segment.stop);
        }
    }
    timings->clear();
}

bool collect_timing_records(Solver* solver, std::vector<TimedSegment>* timings) {
    if (!solver || !timings) {
        return true;
    }
    for (TimedSegment& segment : *timings) {
        float elapsed = 0.0f;
        cudaError_t err = cudaEventElapsedTime(&elapsed, segment.start, segment.stop);
        if (!set_cuda_error(err, "collect native timing")) {
            destroy_timing_records(timings);
            return false;
        }
        float* field = timing_field(&solver->diag, segment.slot);
        if (field) {
            *field += elapsed;
        }
        float* primary_field = timing_primary_field(&solver->diag, segment.slot);
        if (primary_field) {
            *primary_field += elapsed;
        }
    }
    destroy_timing_records(timings);
    return true;
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

bool build_volume_vertex_triangles(Solver* solver, const SsblXpbdConfig* config, const SsblXpbdMesh* mesh) {
    if (!solver || !config || !mesh || config->vertex_count <= 0 || config->triangle_count <= 0 || !mesh->triangles) {
        return true;
    }
    const int vertex_count = config->vertex_count;
    const int triangle_count = config->triangle_count;
    const Int3* triangles = reinterpret_cast<const Int3*>(mesh->triangles);
    std::vector<int> counts(vertex_count, 0);
    for (int t = 0; t < triangle_count; ++t) {
        const Int3 tri = triangles[t];
        if (tri.x >= 0 && tri.x < vertex_count) {
            ++counts[tri.x];
        }
        if (tri.y >= 0 && tri.y < vertex_count) {
            ++counts[tri.y];
        }
        if (tri.z >= 0 && tri.z < vertex_count) {
            ++counts[tri.z];
        }
    }
    std::vector<int> offsets(vertex_count + 1, 0);
    for (int i = 0; i < vertex_count; ++i) {
        offsets[i + 1] = offsets[i] + counts[i];
    }
    std::vector<int> cursor(offsets.begin(), offsets.end());
    std::vector<int> incident(offsets.back(), -1);
    for (int t = 0; t < triangle_count; ++t) {
        const Int3 tri = triangles[t];
        if (tri.x >= 0 && tri.x < vertex_count) {
            incident[cursor[tri.x]++] = t;
        }
        if (tri.y >= 0 && tri.y < vertex_count) {
            incident[cursor[tri.y]++] = t;
        }
        if (tri.z >= 0 && tri.z < vertex_count) {
            incident[cursor[tri.z]++] = t;
        }
    }
    solver->volume_vertex_triangle_count = static_cast<int>(incident.size());
    bool ok = alloc_and_copy(&solver->volume_vertex_offsets, offsets.data(), vertex_count + 1, "missing volume vertex offsets");
    ok = ok && alloc_and_copy(
        &solver->volume_vertex_triangles,
        incident.data(),
        solver->volume_vertex_triangle_count,
        "missing volume incident triangles"
    );
    return ok;
}

int self_sleep_region_index_for_position(
    Vec3 p,
    Vec3 min_pos,
    Vec3 extent,
    int dim_x,
    int dim_y,
    int dim_z
) {
    auto axis_index = [](float value, float min_value, float axis_extent, int dim) {
        if (dim <= 1 || axis_extent <= kEps) {
            return 0;
        }
        float normalized = (value - min_value) / axis_extent;
        int cell = static_cast<int>(std::floor(normalized * static_cast<float>(dim)));
        return std::clamp(cell, 0, dim - 1);
    };
    int x = axis_index(p.x, min_pos.x, extent.x, dim_x);
    int y = axis_index(p.y, min_pos.y, extent.y, dim_y);
    int z = axis_index(p.z, min_pos.z, extent.z, dim_z);
    return x + y * dim_x + z * dim_x * dim_y;
}

bool build_self_sleep_regions(
    Solver* solver,
    const SsblXpbdConfig* config,
    const SsblXpbdMesh* mesh,
    const std::vector<Vec3>& host_pos
) {
    if (!solver || !config || !mesh || !solver->cfg.self_sleep_enabled || config->vertex_count <= 0) {
        return true;
    }

    Vec3 min_pos = host_pos[0];
    Vec3 max_pos = host_pos[0];
    for (const Vec3& p : host_pos) {
        min_pos.x = std::min(min_pos.x, p.x);
        min_pos.y = std::min(min_pos.y, p.y);
        min_pos.z = std::min(min_pos.z, p.z);
        max_pos.x = std::max(max_pos.x, p.x);
        max_pos.y = std::max(max_pos.y, p.y);
        max_pos.z = std::max(max_pos.z, p.z);
    }
    Vec3 extent{
        std::max(max_pos.x - min_pos.x, 1.0e-6f),
        std::max(max_pos.y - min_pos.y, 1.0e-6f),
        std::max(max_pos.z - min_pos.z, 1.0e-6f)
    };
    int target_regions = std::clamp(
        (config->vertex_count + kSelfSleepTargetVerticesPerRegion - 1) / kSelfSleepTargetVerticesPerRegion,
        1,
        kSelfSleepMaxRegions
    );
    int dim_x = 1;
    int dim_y = 1;
    int dim_z = 1;
    while (dim_x * dim_y * dim_z < target_regions) {
        float score_x = extent.x / static_cast<float>(dim_x);
        float score_y = extent.y / static_cast<float>(dim_y);
        float score_z = extent.z / static_cast<float>(dim_z);
        if (score_x >= score_y && score_x >= score_z) {
            ++dim_x;
        } else if (score_y >= score_z) {
            ++dim_y;
        } else {
            ++dim_z;
        }
    }
    int region_count = std::min(dim_x * dim_y * dim_z, kSelfSleepMaxRegions);
    solver->self_sleep_dim_x = dim_x;
    solver->self_sleep_dim_y = dim_y;
    solver->self_sleep_dim_z = dim_z;
    std::vector<int> vertex_regions(config->vertex_count, 0);
    std::vector<int> region_vertex_counts(region_count, 0);
    for (int i = 0; i < config->vertex_count; ++i) {
        vertex_regions[i] = std::clamp(
            self_sleep_region_index_for_position(host_pos[i], min_pos, extent, dim_x, dim_y, dim_z),
            0,
            region_count - 1
        );
        ++region_vertex_counts[vertex_regions[i]];
    }

    std::vector<int> triangle_regions(std::max(config->triangle_count, 0), 0);
    if (config->triangle_count > 0) {
        if (!mesh->triangles) {
            return set_error("missing self sleep triangle data");
        }
        const Int3* triangles = reinterpret_cast<const Int3*>(mesh->triangles);
        for (int t = 0; t < config->triangle_count; ++t) {
            Int3 tri = triangles[t];
            if (tri.x >= 0 && tri.x < config->vertex_count
                && tri.y >= 0 && tri.y < config->vertex_count
                && tri.z >= 0 && tri.z < config->vertex_count) {
                Vec3 centroid{
                    (host_pos[tri.x].x + host_pos[tri.y].x + host_pos[tri.z].x) / 3.0f,
                    (host_pos[tri.x].y + host_pos[tri.y].y + host_pos[tri.z].y) / 3.0f,
                    (host_pos[tri.x].z + host_pos[tri.y].z + host_pos[tri.z].z) / 3.0f
                };
                triangle_regions[t] = std::clamp(
                    self_sleep_region_index_for_position(centroid, min_pos, extent, dim_x, dim_y, dim_z),
                    0,
                    region_count - 1
                );
            }
        }
    }

    solver->self_sleep_region_count = region_count;
    bool ok = true;
    ok = ok && alloc_and_copy(&solver->self_sleep_vertex_regions, vertex_regions.data(), config->vertex_count, "missing self sleep vertex regions");
    ok = ok && alloc_and_copy(&solver->self_sleep_triangle_regions, triangle_regions.data(), config->triangle_count, "missing self sleep triangle regions");
    ok = ok && alloc_and_copy(&solver->self_sleep_prev_pos, host_pos.data(), config->vertex_count, "missing self sleep previous positions");
    ok = ok && alloc_and_copy(&solver->self_sleep_region_vertex_counts, region_vertex_counts.data(), region_count, "missing self sleep region counts");
    if (ok) {
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&solver->self_sleep_region_still_frames), sizeof(int) * region_count);
        ok = ok && set_cuda_error(err, "self sleep still-frame allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->self_sleep_region_sleeping), sizeof(int) * region_count);
        ok = ok && set_cuda_error(err, "self sleep state allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->self_sleep_region_motion), sizeof(int) * region_count);
        ok = ok && set_cuda_error(err, "self sleep motion allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->self_sleep_region_touched), sizeof(int) * region_count);
        ok = ok && set_cuda_error(err, "self sleep touch allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->self_sleep_has_sleeping), sizeof(int));
        ok = ok && set_cuda_error(err, "self sleep summary allocation");
    }
    if (ok) {
        ok = ok && set_cuda_error(cudaMemset(solver->self_sleep_region_still_frames, 0, sizeof(int) * region_count), "self sleep still-frame reset");
        ok = ok && set_cuda_error(cudaMemset(solver->self_sleep_region_sleeping, 0, sizeof(int) * region_count), "self sleep state reset");
        ok = ok && set_cuda_error(cudaMemset(solver->self_sleep_region_motion, 0, sizeof(int) * region_count), "self sleep motion reset");
        ok = ok && set_cuda_error(cudaMemset(solver->self_sleep_region_touched, 0, sizeof(int) * region_count), "self sleep touch reset");
        ok = ok && set_cuda_error(cudaMemset(solver->self_sleep_has_sleeping, 0, sizeof(int)), "self sleep summary reset");
    }
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
        return kSelfSurfaceSamplesPerTriangleDefault;
    }
    return std::min(
        std::max(solver->self_samples_per_triangle, 1),
        kSelfSurfaceSamplesPerTriangleDefault
    );
}

int static_collider_is_complex(const SsblXpbdConfig* config, const SsblXpbdMesh* mesh) {
    if (!config || !mesh || !mesh->static_triangles || config->static_triangle_count < 4) {
        return 0;
    }
    const Vec3* triangles = reinterpret_cast<const Vec3*>(mesh->static_triangles);
    Vec3 reference{0.0f, 0.0f, 0.0f};
    bool have_reference = false;
    for (int t = 0; t < config->static_triangle_count; ++t) {
        Vec3 a = triangles[t * 3 + 0];
        Vec3 b = triangles[t * 3 + 1];
        Vec3 c = triangles[t * 3 + 2];
        Vec3 ab{b.x - a.x, b.y - a.y, b.z - a.z};
        Vec3 ac{c.x - a.x, c.y - a.y, c.z - a.z};
        Vec3 normal{
            ab.y * ac.z - ab.z * ac.y,
            ab.z * ac.x - ab.x * ac.z,
            ab.x * ac.y - ab.y * ac.x,
        };
        float len_sq = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;
        if (!std::isfinite(len_sq) || len_sq <= 1.0e-12f) {
            continue;
        }
        float inv_len = 1.0f / std::sqrt(len_sq);
        normal.x *= inv_len;
        normal.y *= inv_len;
        normal.z *= inv_len;
        if (!have_reference) {
            reference = normal;
            have_reference = true;
            continue;
        }
        float alignment = std::fabs(reference.x * normal.x + reference.y * normal.y + reference.z * normal.z);
        if (alignment < 0.985f) {
            return 1;
        }
    }
    return 0;
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

__device__ float clamp01(float value) {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

__device__ Vec3 array_vec3(const float values[3]) {
    return {values[0], values[1], values[2]};
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
    float len = sqrtf(fmaxf(dot(value, value), 0.0f));
    if (!isfinite(len) || len <= kMaxForceFieldAcceleration) {
        return isfinite(len) ? value : Vec3{0.0f, 0.0f, 0.0f};
    }
    return mul(value, kMaxForceFieldAcceleration / fmaxf(len, kEps));
}

__device__ Vec3 evaluate_force_field(const SsblXpbdForceField& field, Vec3 p, Vec3 velocity) {
    if (!isfinite(field.strength) || field.strength == 0.0f) {
        return {0.0f, 0.0f, 0.0f};
    }
    Vec3 origin = array_vec3(field.origin);
    Vec3 delta = sub(p, origin);
    Vec3 axis = normalize(array_vec3(field.axis));
    Vec3 radial_delta = field.use_2d_force ? sub(delta, mul(axis, dot(delta, axis))) : delta;
    float distance = sqrtf(fmaxf(dot(delta, delta), 0.0f));
    float radial_distance = sqrtf(fmaxf(dot(radial_delta, radial_delta), 0.0f));
    float strength = field.strength * force_field_falloff(field, distance, radial_distance);
    if (strength == 0.0f || !isfinite(strength)) {
        return {0.0f, 0.0f, 0.0f};
    }

    if (field.type == kForceFieldWind) {
        return mul(normalize(array_vec3(field.direction)), strength);
    }
    if (field.type == kForceFieldForce || field.type == kForceFieldCharge) {
        Vec3 source_delta = field.use_2d_force ? radial_delta : delta;
        float source_distance = field.use_2d_force ? radial_distance : distance;
        if (source_distance <= 1.0e-6f) {
            return {0.0f, 0.0f, 0.0f};
        }
        float scale = strength / fmaxf(source_distance, kEps);
        if (field.type == kForceFieldCharge) {
            scale /= fmaxf(source_distance * source_distance, 0.01f);
        }
        return limit_force_field_acceleration(mul(source_delta, scale));
    }
    if (field.type == kForceFieldVortex) {
        Vec3 radial = sub(delta, mul(axis, dot(delta, axis)));
        if (dot(radial, radial) <= 1.0e-10f) {
            return {0.0f, 0.0f, 0.0f};
        }
        Vec3 tangent = normalize(cross(axis, radial));
        return mul(tangent, strength);
    }
    if (field.type == kForceFieldHarmonic) {
        float source_distance = field.use_2d_force ? radial_distance : distance;
        Vec3 source_delta = field.use_2d_force ? radial_delta : delta;
        if (source_distance <= 1.0e-6f) {
            return {0.0f, 0.0f, 0.0f};
        }
        Vec3 direction = mul(source_delta, 1.0f / fmaxf(source_distance, kEps));
        float rest_length = fmaxf(field.rest_length, 0.0f);
        float spring = -strength * (source_distance - rest_length);
        float damping = -dot(velocity, direction) * fmaxf(field.harmonic_damping, 0.0f) * fabsf(strength);
        return limit_force_field_acceleration(mul(direction, spring + damping));
    }
    if (field.type == kForceFieldLennardJ) {
        if (distance <= 1.0e-6f) {
            return {0.0f, 0.0f, 0.0f};
        }
        Vec3 direction = mul(delta, 1.0f / fmaxf(distance, kEps));
        float radius = fmaxf(field.rest_length, fmaxf(field.size, 0.1f));
        float ratio = fminf(radius / fmaxf(distance, 1.0e-3f), 6.0f);
        float ratio2 = ratio * ratio;
        float ratio6 = ratio2 * ratio2 * ratio2;
        float magnitude = strength * (ratio6 * ratio6 - ratio6);
        return limit_force_field_acceleration(mul(direction, magnitude));
    }
    if (field.type == kForceFieldMagnet) {
        Vec3 magnetic_axis = normalize(array_vec3(field.direction));
        return limit_force_field_acceleration(mul(cross(velocity, magnetic_axis), strength));
    }
    if (field.type == kForceFieldDrag) {
        float speed = sqrtf(fmaxf(dot(velocity, velocity), 0.0f));
        if (speed <= 1.0e-6f) {
            return {0.0f, 0.0f, 0.0f};
        }
        float linear = fmaxf(field.linear_drag, 0.0f);
        float quadratic = fmaxf(field.quadratic_drag, 0.0f);
        if (linear <= 0.0f && quadratic <= 0.0f) {
            linear = fabsf(strength);
        }
        return limit_force_field_acceleration(mul(velocity, -(linear + quadratic * speed)));
    }
    if (field.type == kForceFieldTurbulence || field.type == kForceFieldTexture) {
        float frequency = fmaxf(fmaxf(field.noise, field.texture_nabla), 0.25f);
        if (field.size > 1.0e-6f) {
            frequency = fmaxf(frequency, 1.0f / field.size);
        }
        Vec3 q = mul(delta, frequency);
        Vec3 noise_vec{
            force_field_noise(q, field.seed, 0.0f),
            force_field_noise(q, field.seed, 7.0f),
            force_field_noise(q, field.seed, 13.0f),
        };
        return limit_force_field_acceleration(mul(noise_vec, strength));
    }
    return {0.0f, 0.0f, 0.0f};
}

__device__ Vec3 force_field_acceleration(Solver solver, Vec3 p, Vec3 velocity) {
    Vec3 acceleration{0.0f, 0.0f, 0.0f};
    if (!solver.force_fields || solver.force_field_count <= 0) {
        return acceleration;
    }
    int count = min(solver.force_field_count, kMaxForceFields);
    for (int index = 0; index < count; ++index) {
        acceleration = add(acceleration, evaluate_force_field(solver.force_fields[index], p, velocity));
    }
    return limit_force_field_acceleration(acceleration);
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

__device__ Vec3 limited_external_collision_correction(
    Solver solver,
    Vec3 p,
    Vec3 projected,
    int barrier_contact,
    float max_correction_scale,
    float ccd_max_correction_scale
) {
    Vec3 delta = sub(projected, p);
    float delta_sq = dot(delta, delta);
    if (!isfinite(delta_sq) || delta_sq <= 1.0e-12f) {
        return {0.0f, 0.0f, 0.0f};
    }
    float delta_len = sqrtf(delta_sq);
    float contact_distance = external_contact_distance(solver);
    float limit_scale = barrier_contact ? ccd_max_correction_scale : max_correction_scale;
    float relaxation = barrier_contact ? kExternalCollisionCcdRelaxation : kExternalCollisionRelaxation;
    float thickness_limit = fmaxf(contact_distance, fmaxf(solver.cfg.cloth_thickness, 0.0f)) * limit_scale;
    float max_delta = fmaxf(kExternalCollisionMinCorrection, thickness_limit);
    float relaxed_len = fminf(delta_len * relaxation, max_delta);
    if (!isfinite(relaxed_len) || relaxed_len <= 0.0f) {
        return {0.0f, 0.0f, 0.0f};
    }
    return mul(delta, relaxed_len / fmaxf(delta_len, kEps));
}

__device__ void apply_external_collision_response(Vec3* p, Vec3* prev, Vec3 correction) {
    float correction_sq = dot(correction, correction);
    if (!p || !prev || !isfinite(correction_sq) || correction_sq <= 1.0e-12f) {
        return;
    }
    Vec3 next = add(*p, correction);
    float correction_len = sqrtf(correction_sq);
    Vec3 normal = mul(correction, 1.0f / fmaxf(correction_len, kEps));
    Vec3 step = sub(next, *prev);
    float inward = dot(step, normal);
    Vec3 adjusted_prev = *prev;
    if (inward < 0.0f) {
        adjusted_prev = add(adjusted_prev, mul(normal, inward * kExternalCollisionInwardVelocityDamping));
    }
    *p = next;
    *prev = adjusted_prev;
}

__device__ void diag_note_external_contact_cache_hit(Solver solver);
__device__ void diag_note_external_contact_cache_miss(Solver solver);
__device__ void diag_note_external_contact_cache_overflow(Solver solver);
__device__ void diag_note_external_friction_correction(Solver solver);
__device__ unsigned long long external_contact_key(int kind, int vertex, int triangle);
__device__ ExternalContact* find_external_contact(Solver solver, unsigned long long key);
__device__ ExternalContact* reserve_external_contact(Solver solver, unsigned long long key);

__device__ void apply_external_contact_friction(
    Solver solver,
    Vec3 normal,
    float normal_lambda,
    Vec3* p,
    Vec3* prev,
    Vec3* lambda_t
) {
    if (!p || !prev || !finite_vec(normal)) {
        return;
    }
    float damping = clamp01(solver.cfg.contact_tangent_damping);
    float friction = fmaxf(solver.cfg.contact_friction, 0.0f);
    if (damping <= 0.0f || friction <= 0.0f || normal_lambda <= 0.0f) {
        return;
    }
    Vec3 step = sub(*p, *prev);
    Vec3 tangent_step = sub(step, mul(normal, dot(step, normal)));
    float tangent_len = sqrtf(fmaxf(dot(tangent_step, tangent_step), 0.0f));
    if (!isfinite(tangent_len) || tangent_len <= 1.0e-7f) {
        return;
    }
    float max_tangent = friction * fmaxf(normal_lambda, 0.0f);
    float damp_len = fminf(tangent_len * damping, max_tangent);
    if (damp_len <= 0.0f) {
        return;
    }
    Vec3 tangent_dir = mul(tangent_step, 1.0f / fmaxf(tangent_len, kEps));
    Vec3 correction = mul(tangent_dir, damp_len);
    *prev = add(*prev, correction);
    if (lambda_t) {
        *lambda_t = add(*lambda_t, correction);
    }
    diag_note_external_friction_correction(solver);
}

__device__ void solve_external_cached_contact(
    Solver solver,
    int kind,
    int vertex,
    int triangle,
    Vec3 normal,
    Vec3 barycentric,
    Vec3 projected,
    int barrier_contact,
    float max_correction_scale,
    float ccd_max_correction_scale,
    Vec3* p,
    Vec3* prev
) {
    if (!p || !prev || !finite_vec(normal)) {
        return;
    }
    float inv_mass = solver.inv_mass[vertex];
    if (inv_mass <= 0.0f) {
        return;
    }

    unsigned long long key = external_contact_key(kind, vertex, triangle);
    ExternalContact* contact = find_external_contact(solver, key);
    bool cache_hit = contact && contact->key == key && contact->age <= kExternalContactMaxAge;
    if (cache_hit) {
        diag_note_external_contact_cache_hit(solver);
        if (!contact->active) {
            float warm_len = fminf(
                fmaxf(contact->lambda_n, 0.0f) * inv_mass * kExternalContactWarmStartScale,
                fmaxf(external_contact_distance(solver), kExternalCollisionMinCorrection) * 0.5f
            );
            if (warm_len > 0.0f) {
                Vec3 warm = mul(normal, warm_len);
                *p = add(*p, warm);
                *prev = add(*prev, warm);
            }
        }
    } else {
        contact = reserve_external_contact(solver, key);
        diag_note_external_contact_cache_miss(solver);
        if (contact) {
            contact->lambda_n = 0.0f;
            contact->lambda_t = {0.0f, 0.0f, 0.0f};
        }
    }

    Vec3 desired_delta = sub(projected, *p);
    float depth = fmaxf(dot(desired_delta, normal), 0.0f);
    float old_lambda = contact ? fmaxf(contact->lambda_n, 0.0f) : 0.0f;
    float alpha = solver.cfg.contact_compliance / fmaxf(solver.cfg.dt * solver.cfg.dt, kEps);
    float dlambda = 0.0f;
    if (depth > 0.0f) {
        dlambda = (depth - alpha * old_lambda) / fmaxf(inv_mass + alpha, kEps);
    }
    float new_lambda = fmaxf(old_lambda + dlambda, 0.0f);
    float correction_len = fmaxf((new_lambda - old_lambda) * inv_mass, 0.0f);
    Vec3 xpbd_projected = add(*p, mul(normal, correction_len));
    Vec3 correction = limited_external_collision_correction(
        solver,
        *p,
        xpbd_projected,
        barrier_contact,
        max_correction_scale,
        ccd_max_correction_scale
    );
    float applied_normal = fmaxf(dot(correction, normal), 0.0f);
    apply_external_collision_response(p, prev, correction);
    if (contact) {
        contact->normal = normal;
        contact->barycentric = barycentric;
        contact->lambda_n = fmaxf(old_lambda + applied_normal / fmaxf(inv_mass, kEps), 0.0f);
        contact->active = 1;
        contact->age = 0;
        apply_external_contact_friction(solver, normal, contact->lambda_n * inv_mass, p, prev, &contact->lambda_t);
    } else {
        Vec3 unused_lambda_t{0.0f, 0.0f, 0.0f};
        apply_external_contact_friction(solver, normal, applied_normal, p, prev, &unused_lambda_t);
    }
}

__global__ void begin_external_contact_cache_step_kernel(Solver solver) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (!solver.external_contacts || index >= solver.external_contact_capacity) {
        return;
    }
    if (solver.external_contacts[index].key != 0ull) {
        solver.external_contacts[index].active = 0;
    }
}

__global__ void finalize_external_contact_cache_step_kernel(Solver solver) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (!solver.external_contacts || index >= solver.external_contact_capacity) {
        return;
    }
    ExternalContact* contact = &solver.external_contacts[index];
    if (contact->key == 0ull) {
        return;
    }
    if (contact->active) {
        contact->age = 0;
        if (solver.diag_counts) {
            atomicAdd(&solver.diag_counts[kDiagExternalContactCacheCount], 1ull);
        }
        return;
    }
    contact->age += 1;
    contact->lambda_n *= 0.5f;
    contact->lambda_t = mul(contact->lambda_t, 0.5f);
    if (contact->age > kExternalContactMaxAge) {
        contact->key = 0ull;
        contact->lambda_n = 0.0f;
        contact->lambda_t = {0.0f, 0.0f, 0.0f};
        contact->active = 0;
    } else if (solver.diag_counts) {
        atomicAdd(&solver.diag_counts[kDiagExternalContactCacheCount], 1ull);
    }
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

__device__ float dynamic_particle_cell_size(Solver solver) {
    float radius = external_contact_distance(solver) + fmaxf(solver.dynamic_particle_max_radius, 0.0f);
    return fmaxf(radius * 2.0f, 1.25e-2f);
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

__device__ void diag_note_external_contact_cache_hit(Solver solver) {
    if (solver.diag_counts) {
        atomicAdd(&solver.diag_counts[kDiagExternalContactCacheHits], 1ull);
    }
}

__device__ void diag_note_external_contact_cache_miss(Solver solver) {
    if (solver.diag_counts) {
        atomicAdd(&solver.diag_counts[kDiagExternalContactCacheMisses], 1ull);
    }
}

__device__ void diag_note_external_contact_cache_overflow(Solver solver) {
    if (solver.diag_counts) {
        atomicAdd(&solver.diag_counts[kDiagExternalContactCacheOverflow], 1ull);
    }
}

__device__ void diag_note_external_friction_correction(Solver solver) {
    if (solver.diag_counts) {
        atomicAdd(&solver.diag_counts[kDiagExternalFrictionCorrections], 1ull);
    }
}

__device__ void diag_note_dynamic_particle_candidate(Solver solver) {
    if (solver.diag_counts) {
        atomicAdd(&solver.diag_counts[kDiagDynamicParticleCandidates], 1ull);
    }
}

__device__ void diag_note_dynamic_particle_contact(Solver solver) {
    if (solver.diag_counts) {
        atomicAdd(&solver.diag_counts[kDiagDynamicParticleContacts], 1ull);
    }
}

__device__ void diag_note_dynamic_particle_overflow(Solver solver) {
    if (solver.diag_counts) {
        atomicAdd(&solver.diag_counts[kDiagDynamicParticleOverflow], 1ull);
    }
}

__device__ unsigned long long external_contact_key(int kind, int vertex, int triangle) {
    return (static_cast<unsigned long long>(kind & 0xff) << 56)
        ^ (static_cast<unsigned long long>(vertex & 0x0fffffff) << 28)
        ^ static_cast<unsigned long long>(triangle & 0x0fffffff);
}

__device__ int external_contact_slot(Solver solver, unsigned long long key) {
    if (!solver.external_contacts || solver.external_contact_capacity <= 0) {
        return -1;
    }
    unsigned long long mixed = key ^ (key >> 33);
    mixed *= 0xff51afd7ed558ccdULL;
    mixed ^= (mixed >> 33);
    return static_cast<int>(mixed % static_cast<unsigned long long>(solver.external_contact_capacity));
}

__device__ ExternalContact* find_external_contact(Solver solver, unsigned long long key) {
    int base = external_contact_slot(solver, key);
    if (base < 0) {
        return nullptr;
    }
    for (int probe = 0; probe < kExternalContactProbeLimit; ++probe) {
        int slot = (base + probe) % solver.external_contact_capacity;
        ExternalContact* contact = &solver.external_contacts[slot];
        if (contact->key == key) {
            return contact;
        }
        if (contact->key == 0ull) {
            return nullptr;
        }
    }
    return nullptr;
}

__device__ ExternalContact* reserve_external_contact(Solver solver, unsigned long long key) {
    int base = external_contact_slot(solver, key);
    if (base < 0) {
        return nullptr;
    }
    for (int probe = 0; probe < kExternalContactProbeLimit; ++probe) {
        int slot = (base + probe) % solver.external_contact_capacity;
        ExternalContact* contact = &solver.external_contacts[slot];
        unsigned long long previous = atomicCAS(&contact->key, 0ull, key);
        if (previous == 0ull || previous == key) {
            return contact;
        }
    }
    diag_note_external_contact_cache_overflow(solver);
    return nullptr;
}

__device__ void store_external_contact_cache(
    Solver solver,
    int kind,
    int vertex,
    int triangle,
    Vec3 normal,
    Vec3 barycentric,
    float lambda_n
) {
    unsigned long long key = external_contact_key(kind, vertex, triangle);
    ExternalContact* contact = find_external_contact(solver, key);
    if (contact && contact->age <= kExternalContactMaxAge) {
        diag_note_external_contact_cache_hit(solver);
    } else {
        contact = reserve_external_contact(solver, key);
        diag_note_external_contact_cache_miss(solver);
    }
    if (!contact) {
        return;
    }
    contact->normal = normal;
    contact->barycentric = barycentric;
    contact->lambda_n = fmaxf(lambda_n, 0.0f);
    contact->lambda_t = {0.0f, 0.0f, 0.0f};
    contact->active = 1;
    contact->age = 0;
}

__device__ Vec3 barycentric_on_triangle(Vec3 p, Vec3 a, Vec3 b, Vec3 c) {
    Vec3 v0 = sub(b, a);
    Vec3 v1 = sub(c, a);
    Vec3 v2 = sub(p, a);
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    if (fabsf(denom) <= kEps) {
        return {1.0f, 0.0f, 0.0f};
    }
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    v = clamp01(v);
    w = clamp01(w);
    if (v + w > 1.0f) {
        float sum = fmaxf(v + w, kEps);
        v /= sum;
        w /= sum;
    }
    return {1.0f - v - w, v, w};
}

__device__ void diag_note_nonfinite(Solver solver) {
    if (solver.diag_counts) {
        atomicExch(&solver.diag_counts[kDiagFiniteFlag], 0ull);
    }
}

__device__ void diag_note_self_source_skipped(Solver solver) {
    if (solver.diag_counts) {
        atomicAdd(&solver.diag_counts[kDiagSelfSkippedSources], 1ull);
    }
}

__device__ bool self_sleep_available(Solver solver) {
    return solver.cfg.self_sleep_enabled
        && !solver.self_sleep_force_active
        && solver.self_sleep_region_count > 0
        && solver.self_sleep_region_sleeping
        && solver.self_sleep_has_sleeping
        && solver.self_sleep_has_sleeping[0] != 0;
}

__device__ int self_sleep_region_for_vertex(Solver solver, int index) {
    if (!solver.self_sleep_vertex_regions || index < 0 || index >= solver.cfg.vertex_count) {
        return -1;
    }
    int region = solver.self_sleep_vertex_regions[index];
    return (region >= 0 && region < solver.self_sleep_region_count) ? region : -1;
}

__device__ int self_sleep_region_for_triangle(Solver solver, int triangle_index) {
    if (!solver.self_sleep_triangle_regions || triangle_index < 0 || triangle_index >= solver.cfg.triangle_count) {
        return -1;
    }
    int region = solver.self_sleep_triangle_regions[triangle_index];
    return (region >= 0 && region < solver.self_sleep_region_count) ? region : -1;
}

__device__ bool self_sleep_source_vertex_skipped(Solver solver, int index) {
    if (!self_sleep_available(solver)) {
        return false;
    }
    int region = self_sleep_region_for_vertex(solver, index);
    return region >= 0 && solver.self_sleep_region_sleeping[region] != 0;
}

__device__ bool self_sleep_source_triangle_skipped(Solver solver, int triangle_index) {
    if (!self_sleep_available(solver)) {
        return false;
    }
    int region = self_sleep_region_for_triangle(solver, triangle_index);
    return region >= 0 && solver.self_sleep_region_sleeping[region] != 0;
}

__device__ bool self_region_is_active(Solver solver, int region) {
    if (region < 0 || region >= solver.self_sleep_region_count) {
        return false;
    }
    return solver.self_sleep_force_active
        || !solver.self_sleep_region_sleeping
        || solver.self_sleep_region_sleeping[region] == 0;
}

__device__ bool self_region_is_suspect(Solver solver, int region) {
    if (region < 0 || region >= solver.self_sleep_region_count) {
        return false;
    }
    if (self_region_is_active(solver, region)) {
        return true;
    }
    int dim_x = solver.self_sleep_dim_x > 1 ? solver.self_sleep_dim_x : 1;
    int dim_y = solver.self_sleep_dim_y > 1 ? solver.self_sleep_dim_y : 1;
    int dim_z = solver.self_sleep_dim_z > 1 ? solver.self_sleep_dim_z : 1;
    int z = region / (dim_x * dim_y);
    int rem = region - z * dim_x * dim_y;
    int y = rem / dim_x;
    int x = rem - y * dim_x;
    for (int dz = -1; dz <= 1; ++dz) {
        int nz = z + dz;
        if (nz < 0 || nz >= dim_z) {
            continue;
        }
        for (int dy = -1; dy <= 1; ++dy) {
            int ny = y + dy;
            if (ny < 0 || ny >= dim_y) {
                continue;
            }
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = x + dx;
                if (nx < 0 || nx >= dim_x) {
                    continue;
                }
                int neighbor = nx + ny * dim_x + nz * dim_x * dim_y;
                if (self_region_is_active(solver, neighbor)) {
                    return true;
                }
            }
        }
    }
    return false;
}

__device__ float self_compaction_fraction_threshold(Solver solver) {
    float threshold = solver.cfg.self_compaction_active_fraction_threshold;
    if (!(threshold > 0.0f)) {
        threshold = 0.75f;
    }
    return fminf(fmaxf(threshold, 0.01f), 1.0f);
}

__device__ int self_compaction_count(Solver solver, int slot) {
    if (slot == kSelfCompactionActiveVertices) {
        return solver.self_active_vertex_count;
    }
    if (slot == kSelfCompactionSuspectVertices) {
        return solver.self_suspect_vertex_count;
    }
    if (slot == kSelfCompactionActiveSamples) {
        return solver.self_active_sample_count;
    }
    if (slot == kSelfCompactionSuspectSamples) {
        return solver.self_suspect_sample_count;
    }
    if (slot == kSelfCompactionSuspectRegions) {
        return solver.self_suspect_region_count;
    }
    return 0;
}

__device__ bool self_compaction_uses_vertex_list(Solver solver) {
    if (!solver.cfg.self_compaction_enabled || !solver.self_compaction_used) {
        return false;
    }
    int slot = -1;
    if (solver.self_source_mode == kSelfSourceActive && solver.self_active_vertices) {
        slot = kSelfCompactionActiveVertices;
    } else if (solver.self_source_mode == kSelfSourceSuspect && solver.self_suspect_vertices) {
        slot = kSelfCompactionSuspectVertices;
    } else {
        return false;
    }
    int full_count = solver.cfg.vertex_count;
    if (full_count <= 0) {
        return false;
    }
    int compacted_count = self_compaction_count(solver, slot);
    return compacted_count >= 0
        && compacted_count < full_count
        && static_cast<float>(compacted_count) < static_cast<float>(full_count) * self_compaction_fraction_threshold(solver);
}

__device__ bool self_compaction_uses_sample_list(Solver solver) {
    if (!solver.cfg.self_compaction_enabled || !solver.self_compaction_used) {
        return false;
    }
    if (solver.self_compaction_samples_per_triangle != solver.self_samples_per_triangle) {
        return false;
    }
    int slot = -1;
    if (solver.self_source_mode == kSelfSourceActive && solver.self_active_samples) {
        slot = kSelfCompactionActiveSamples;
    } else if (solver.self_source_mode == kSelfSourceSuspect && solver.self_suspect_samples) {
        slot = kSelfCompactionSuspectSamples;
    } else {
        return false;
    }
    int full_count = solver.self_sample_count;
    if (full_count <= 0) {
        return false;
    }
    int compacted_count = self_compaction_count(solver, slot);
    return compacted_count >= 0
        && compacted_count < full_count
        && static_cast<float>(compacted_count) < static_cast<float>(full_count) * self_compaction_fraction_threshold(solver);
}

__device__ int self_vertex_source_count(Solver solver) {
    if (self_compaction_uses_vertex_list(solver)) {
        if (solver.self_source_mode == kSelfSourceActive) {
            return self_compaction_count(solver, kSelfCompactionActiveVertices);
        }
        if (solver.self_source_mode == kSelfSourceSuspect) {
            return self_compaction_count(solver, kSelfCompactionSuspectVertices);
        }
    }
    return solver.cfg.vertex_count;
}

__device__ int self_sample_source_count(Solver solver) {
    if (self_compaction_uses_sample_list(solver)) {
        if (solver.self_source_mode == kSelfSourceActive) {
            return self_compaction_count(solver, kSelfCompactionActiveSamples);
        }
        if (solver.self_source_mode == kSelfSourceSuspect) {
            return self_compaction_count(solver, kSelfCompactionSuspectSamples);
        }
    }
    return solver.self_sample_count;
}

__device__ int self_vertex_source_index(Solver solver, int ordinal) {
    if (self_compaction_uses_vertex_list(solver) && solver.self_source_mode == kSelfSourceActive && solver.self_active_vertices) {
        return solver.self_active_vertices[ordinal];
    }
    if (self_compaction_uses_vertex_list(solver) && solver.self_source_mode == kSelfSourceSuspect && solver.self_suspect_vertices) {
        return solver.self_suspect_vertices[ordinal];
    }
    return ordinal;
}

__device__ bool self_source_vertex_flag(Solver solver, int vertex) {
    if (vertex < 0 || vertex >= solver.cfg.vertex_count) {
        return false;
    }
    if (!solver.cfg.self_compaction_enabled || !solver.self_compaction_used) {
        return true;
    }
    if (solver.self_source_mode == kSelfSourceActive && solver.self_active_vertex_flags) {
        return solver.self_active_vertex_flags[vertex] != 0;
    }
    if (solver.self_source_mode == kSelfSourceSuspect && solver.self_suspect_vertex_flags) {
        return solver.self_suspect_vertex_flags[vertex] != 0;
    }
    return true;
}

__device__ bool self_source_edge_flag(Solver solver, Int2 edge) {
    return self_source_vertex_flag(solver, edge.x) || self_source_vertex_flag(solver, edge.y);
}

__device__ int self_sample_source_index(Solver solver, int ordinal) {
    if (self_compaction_uses_sample_list(solver) && solver.self_source_mode == kSelfSourceActive && solver.self_active_samples) {
        return solver.self_active_samples[ordinal];
    }
    if (self_compaction_uses_sample_list(solver) && solver.self_source_mode == kSelfSourceSuspect && solver.self_suspect_samples) {
        return solver.self_suspect_samples[ordinal];
    }
    return ordinal;
}

__device__ int self_triangle_hash_source_count(Solver solver) {
    if (self_compaction_uses_sample_list(solver)) {
        return self_sample_source_count(solver);
    }
    return solver.cfg.triangle_count;
}

__device__ int self_triangle_hash_source_index(Solver solver, int ordinal) {
    if (self_compaction_uses_sample_list(solver)) {
        int sample = self_sample_source_index(solver, ordinal);
        if (sample < 0 || sample >= solver.self_sample_count) {
            return -1;
        }
        int samples_per_triangle = solver.self_samples_per_triangle > 0 ? solver.self_samples_per_triangle : 1;
        int tri_index = sample / samples_per_triangle;
        if (sample - tri_index * samples_per_triangle != 0) {
            return -1;
        }
        return tri_index;
    }
    return ordinal;
}

__device__ void note_self_region_touch(Solver solver, int index) {
    if (!solver.self_sleep_region_touched) {
        return;
    }
    int region = self_sleep_region_for_vertex(solver, index);
    if (region >= 0) {
        atomicExch(&solver.self_sleep_region_touched[region], 1);
    }
}

__device__ void note_self_collision_correction(Solver solver, int index, Vec3 delta) {
    if (index < 0 || index >= solver.cfg.vertex_count) {
        return;
    }
    note_self_region_touch(solver, index);
    if (solver.self_recovery_touched) {
        atomicExch(&solver.self_recovery_touched[index], 1);
    }
    if (solver.self_recovery_delta) {
        atomic_add(&solver.self_recovery_delta[index], delta);
    }
}

__device__ void apply_self_collision_correction(Solver solver, int index, Vec3 delta) {
    if (index < 0 || index >= solver.cfg.vertex_count) {
        return;
    }
    atomic_add(&solver.pos[index], delta);
    if (!solver.self_recovery_mode) {
        atomic_add(&solver.prev[index], delta);
    }
    note_self_collision_correction(solver, index, delta);
}

__device__ void apply_self_collision_correction_untracked(Solver solver, int index, Vec3 delta) {
    if (index < 0 || index >= solver.cfg.vertex_count) {
        return;
    }
    atomic_add(&solver.pos[index], delta);
    if (!solver.self_recovery_mode) {
        atomic_add(&solver.prev[index], delta);
    }
    note_self_region_touch(solver, index);
}

__device__ void apply_self_collision_correction_without_frontier(Solver solver, int index, Vec3 delta) {
    if (index < 0 || index >= solver.cfg.vertex_count) {
        return;
    }
    atomic_add(&solver.pos[index], delta);
    if (!solver.self_recovery_mode) {
        atomic_add(&solver.prev[index], delta);
    }
    note_self_region_touch(solver, index);
    if (solver.self_recovery_delta) {
        atomic_add(&solver.self_recovery_delta[index], delta);
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

__device__ bool self_fast_mode(Solver solver) {
    return solver.cfg.self_collision_mode == kSelfCollisionModeFast;
}

__device__ bool self_contact_frontier_vertex(Solver solver, int vertex) {
    if (!self_fast_mode(solver) || !solver.self_recovery_touched || vertex < 0 || vertex >= solver.cfg.vertex_count) {
        return false;
    }
    if (solver.self_recovery_touched[vertex] != 0) {
        return true;
    }
    if (!solver.vertex_neighbor_offsets || !solver.vertex_neighbors) {
        return false;
    }
    int start = solver.vertex_neighbor_offsets[vertex];
    int end = solver.vertex_neighbor_offsets[vertex + 1];
    for (int idx = start; idx < end; ++idx) {
        int neighbor = solver.vertex_neighbors[idx];
        if (neighbor >= 0 && neighbor < solver.cfg.vertex_count && solver.self_recovery_touched[neighbor] != 0) {
            return true;
        }
    }
    return false;
}

__device__ bool self_contact_frontier_triangle(Solver solver, Int3 tri) {
    return self_contact_frontier_vertex(solver, tri.x)
        || self_contact_frontier_vertex(solver, tri.y)
        || self_contact_frontier_vertex(solver, tri.z);
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

__device__ bool self_should_project_contact(float gap, Vec3 delta, Vec3 previous_delta, Vec3 normal) {
    return gap < 0.0f || self_is_approaching(delta, previous_delta, normal);
}

__device__ float self_projection_relaxation(Solver solver) {
    if (self_fast_mode(solver)) {
        if (!solver.self_recovery_mode && solver.self_cleanup_mode) {
            return kFastSelfCleanupProjectionRelaxation;
        }
        return solver.self_recovery_mode ? kFastSelfRecoveryProjectionRelaxation : kFastSelfProjectionRelaxation;
    }
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

__global__ void build_self_compaction_regions_kernel(Solver solver) {
    int region = blockIdx.x * blockDim.x + threadIdx.x;
    if (region >= solver.self_sleep_region_count || !solver.self_compaction_counts) {
        return;
    }
    if (self_region_is_active(solver, region)) {
        atomicAdd(&solver.self_compaction_counts[kSelfCompactionActiveRegions], 1);
    }
    if (self_region_is_suspect(solver, region)) {
        atomicAdd(&solver.self_compaction_counts[kSelfCompactionSuspectRegions], 1);
    }
}

__global__ void build_self_compaction_vertices_kernel(Solver solver) {
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= solver.cfg.vertex_count || !solver.self_compaction_counts) {
        return;
    }
    int region = self_sleep_region_for_vertex(solver, vertex);
    bool use_frontier = self_fast_mode(solver) && solver.self_recovery_touched;
    bool frontier = use_frontier && self_contact_frontier_vertex(solver, vertex);
    bool active = use_frontier ? frontier : (region < 0 || self_region_is_active(solver, region));
    bool suspect = use_frontier ? frontier : (region < 0 || self_region_is_suspect(solver, region));
    if (active && solver.self_active_vertices) {
        int slot = atomicAdd(&solver.self_compaction_counts[kSelfCompactionActiveVertices], 1);
        if (slot < solver.cfg.vertex_count) {
            solver.self_active_vertices[slot] = vertex;
        }
        if (solver.self_active_vertex_flags) {
            solver.self_active_vertex_flags[vertex] = 1;
        }
    }
    if (suspect && solver.self_suspect_vertices) {
        int slot = atomicAdd(&solver.self_compaction_counts[kSelfCompactionSuspectVertices], 1);
        if (slot < solver.cfg.vertex_count) {
            solver.self_suspect_vertices[slot] = vertex;
        }
        if (solver.self_suspect_vertex_flags) {
            solver.self_suspect_vertex_flags[vertex] = 1;
        }
    }
}

__device__ void atomic_max_float(float* address, float value) {
    if (!address || !isfinite(value) || value < 0.0f) {
        return;
    }
    int* address_as_int = reinterpret_cast<int*>(address);
    int old_bits = *address_as_int;
    while (__int_as_float(old_bits) < value) {
        int assumed = old_bits;
        old_bits = atomicCAS(address_as_int, assumed, __float_as_int(value));
        if (old_bits == assumed) {
            break;
        }
    }
}

__global__ void build_self_compaction_samples_kernel(Solver solver) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= solver.self_sample_count || !solver.self_compaction_counts) {
        return;
    }
    int tri_index = self_sample_triangle_index(solver, sample);
    int region = self_sleep_region_for_triangle(solver, tri_index);
    Int3 tri = solver.triangles[tri_index];
    bool use_frontier = self_fast_mode(solver) && solver.self_recovery_touched;
    bool frontier = use_frontier && self_contact_frontier_triangle(solver, tri);
    bool active = use_frontier ? frontier : (region < 0 || self_region_is_active(solver, region));
    bool suspect = use_frontier ? frontier : (region < 0 || self_region_is_suspect(solver, region));
    if (active && solver.self_active_samples) {
        int slot = atomicAdd(&solver.self_compaction_counts[kSelfCompactionActiveSamples], 1);
        if (slot < solver.self_sample_count) {
            solver.self_active_samples[slot] = sample;
        }
    }
    if (suspect && solver.self_suspect_samples) {
        int slot = atomicAdd(&solver.self_compaction_counts[kSelfCompactionSuspectSamples], 1);
        if (slot < solver.self_sample_count) {
            solver.self_suspect_samples[slot] = sample;
        }
    }
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

__device__ bool edges_share_vertex(Int2 a, Int2 b) {
    return a.x == b.x || a.x == b.y || a.y == b.x || a.y == b.y;
}

__device__ bool rest_edges_neighbor(Solver solver, Int2 a, Int2 b) {
    if (edges_share_vertex(a, b)) {
        return true;
    }
    if (same_or_one_ring_neighbor(solver, a.x, b.x)
        || same_or_one_ring_neighbor(solver, a.x, b.y)
        || same_or_one_ring_neighbor(solver, a.y, b.x)
        || same_or_one_ring_neighbor(solver, a.y, b.y)) {
        return true;
    }
    float thickness = self_contact_distance(solver);
    float close_threshold = fmaxf(thickness * 0.5f, 1.0e-5f);
    Vec3 mid_a = mul(add(solver.rest[a.x], solver.rest[a.y]), 0.5f);
    Vec3 mid_b = mul(add(solver.rest[b.x], solver.rest[b.y]), 0.5f);
    return norm(sub(mid_a, mid_b)) <= close_threshold;
}

__device__ void closest_segment_parameters(
    Vec3 p1,
    Vec3 q1,
    Vec3 p2,
    Vec3 q2,
    float* s_out,
    float* t_out
) {
    Vec3 d1 = sub(q1, p1);
    Vec3 d2 = sub(q2, p2);
    Vec3 r = sub(p1, p2);
    float a = dot(d1, d1);
    float e = dot(d2, d2);
    float f = dot(d2, r);
    float s = 0.0f;
    float t = 0.0f;

    if (a <= kEps && e <= kEps) {
        *s_out = 0.0f;
        *t_out = 0.0f;
        return;
    }
    if (a <= kEps) {
        s = 0.0f;
        t = f / fmaxf(e, kEps);
        t = fminf(fmaxf(t, 0.0f), 1.0f);
    } else {
        float c = dot(d1, r);
        if (e <= kEps) {
            t = 0.0f;
            s = fminf(fmaxf(-c / fmaxf(a, kEps), 0.0f), 1.0f);
        } else {
            float b = dot(d1, d2);
            float denom = a * e - b * b;
            if (denom != 0.0f) {
                s = fminf(fmaxf((b * f - c * e) / denom, 0.0f), 1.0f);
            } else {
                s = 0.0f;
            }
            t = (b * s + f) / e;
            if (t < 0.0f) {
                t = 0.0f;
                s = fminf(fmaxf(-c / fmaxf(a, kEps), 0.0f), 1.0f);
            } else if (t > 1.0f) {
                t = 1.0f;
                s = fminf(fmaxf((b - c) / fmaxf(a, kEps), 0.0f), 1.0f);
            }
        }
    }
    *s_out = s;
    *t_out = t;
}

__device__ bool segment_triangle_intersection(
    Vec3 p0,
    Vec3 p1,
    Vec3 a,
    Vec3 b,
    Vec3 c,
    float* edge_t,
    float* wa,
    float* wb,
    float* wc
) {
    Vec3 direction = sub(p1, p0);
    Vec3 edge1 = sub(b, a);
    Vec3 edge2 = sub(c, a);
    Vec3 h = cross(direction, edge2);
    float det = dot(edge1, h);
    if (fabsf(det) <= 1.0e-7f) {
        return false;
    }
    float inv_det = 1.0f / det;
    Vec3 s = sub(p0, a);
    float u = inv_det * dot(s, h);
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    Vec3 q = cross(s, edge1);
    float v = inv_det * dot(direction, q);
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }
    float t = inv_det * dot(edge2, q);
    if (t < 0.0f || t > 1.0f) {
        return false;
    }
    *edge_t = t;
    *wa = 1.0f - u - v;
    *wb = u;
    *wc = v;
    return true;
}

__device__ Vec3 closest_point_on_triangle(
    Vec3 p,
    Vec3 a,
    Vec3 b,
    Vec3 c,
    float* wa,
    float* wb,
    float* wc
) {
    Vec3 ab = sub(b, a);
    Vec3 ac = sub(c, a);
    Vec3 ap = sub(p, a);
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        *wa = 1.0f;
        *wb = 0.0f;
        *wc = 0.0f;
        return a;
    }

    Vec3 bp = sub(p, b);
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        *wa = 0.0f;
        *wb = 1.0f;
        *wc = 0.0f;
        return b;
    }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / fmaxf(d1 - d3, kEps);
        *wa = 1.0f - v;
        *wb = v;
        *wc = 0.0f;
        return add(a, mul(ab, v));
    }

    Vec3 cp = sub(p, c);
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        *wa = 0.0f;
        *wb = 0.0f;
        *wc = 1.0f;
        return c;
    }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / fmaxf(d2 - d6, kEps);
        *wa = 1.0f - w;
        *wb = 0.0f;
        *wc = w;
        return add(a, mul(ac, w));
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / fmaxf((d4 - d3) + (d5 - d6), kEps);
        *wa = 0.0f;
        *wb = 1.0f - w;
        *wc = w;
        return add(b, mul(sub(c, b), w));
    }

    float denom = 1.0f / fmaxf(va + vb + vc, kEps);
    float v = vb * denom;
    float w = vc * denom;
    *wa = 1.0f - v - w;
    *wb = v;
    *wc = w;
    return add(a, add(mul(ab, v), mul(ac, w)));
}

__device__ void closest_segment_triangle_contact(
    Vec3 p0,
    Vec3 p1,
    Vec3 a,
    Vec3 b,
    Vec3 c,
    float* edge_t,
    float* wa,
    float* wb,
    float* wc,
    float* distance,
    Vec3* edge_point,
    Vec3* triangle_point
) {
    if (segment_triangle_intersection(p0, p1, a, b, c, edge_t, wa, wb, wc)) {
        *edge_point = add(p0, mul(sub(p1, p0), *edge_t));
        *triangle_point = weighted_triangle_point(a, b, c, *wa, *wb, *wc);
        *distance = 0.0f;
        return;
    }

    float best_d2 = FLT_MAX;
    float best_t = 0.0f;
    float best_wa = 1.0f;
    float best_wb = 0.0f;
    float best_wc = 0.0f;
    Vec3 best_edge_point = p0;
    Vec3 best_tri_point = a;

    for (int endpoint = 0; endpoint < 2; ++endpoint) {
        Vec3 p = endpoint == 0 ? p0 : p1;
        float twa;
        float twb;
        float twc;
        Vec3 q = closest_point_on_triangle(p, a, b, c, &twa, &twb, &twc);
        Vec3 delta = sub(p, q);
        float d2 = dot(delta, delta);
        if (d2 < best_d2) {
            best_d2 = d2;
            best_t = endpoint == 0 ? 0.0f : 1.0f;
            best_wa = twa;
            best_wb = twb;
            best_wc = twc;
            best_edge_point = p;
            best_tri_point = q;
        }
    }

    for (int tri_edge = 0; tri_edge < 3; ++tri_edge) {
        Vec3 q0 = tri_edge == 0 ? a : (tri_edge == 1 ? b : c);
        Vec3 q1 = tri_edge == 0 ? b : (tri_edge == 1 ? c : a);
        float s = 0.0f;
        float t = 0.0f;
        closest_segment_parameters(p0, p1, q0, q1, &s, &t);
        Vec3 p = add(p0, mul(sub(p1, p0), s));
        Vec3 q = add(q0, mul(sub(q1, q0), t));
        Vec3 delta = sub(p, q);
        float d2 = dot(delta, delta);
        if (d2 < best_d2) {
            best_d2 = d2;
            best_t = s;
            if (tri_edge == 0) {
                best_wa = 1.0f - t;
                best_wb = t;
                best_wc = 0.0f;
            } else if (tri_edge == 1) {
                best_wa = 0.0f;
                best_wb = 1.0f - t;
                best_wc = t;
            } else {
                best_wa = t;
                best_wb = 0.0f;
                best_wc = 1.0f - t;
            }
            best_edge_point = p;
            best_tri_point = q;
        }
    }

    *edge_t = best_t;
    *wa = best_wa;
    *wb = best_wb;
    *wc = best_wc;
    *edge_point = best_edge_point;
    *triangle_point = best_tri_point;
    *distance = sqrtf(fmaxf(best_d2, 0.0f));
}

__device__ bool rest_edge_surface_neighbor(Solver solver, Int2 edge, Int3 tri) {
    if (edge.x == tri.x || edge.x == tri.y || edge.x == tri.z
        || edge.y == tri.x || edge.y == tri.y || edge.y == tri.z) {
        return true;
    }
    if (same_or_one_ring_neighbor(solver, edge.x, tri.x)
        || same_or_one_ring_neighbor(solver, edge.x, tri.y)
        || same_or_one_ring_neighbor(solver, edge.x, tri.z)
        || same_or_one_ring_neighbor(solver, edge.y, tri.x)
        || same_or_one_ring_neighbor(solver, edge.y, tri.y)
        || same_or_one_ring_neighbor(solver, edge.y, tri.z)) {
        return true;
    }

    float edge_t;
    float wa;
    float wb;
    float wc;
    float distance;
    Vec3 edge_point;
    Vec3 tri_point;
    closest_segment_triangle_contact(
        solver.rest[edge.x],
        solver.rest[edge.y],
        solver.rest[tri.x],
        solver.rest[tri.y],
        solver.rest[tri.z],
        &edge_t,
        &wa,
        &wb,
        &wc,
        &distance,
        &edge_point,
        &tri_point
    );
    float close_threshold = fmaxf(self_contact_distance(solver) * 0.35f, 1.0e-5f);
    return distance <= close_threshold;
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
    Vec3 acceleration{solver.cfg.gravity[0], solver.cfg.gravity[1], solver.cfg.gravity[2]};
    acceleration = add(acceleration, force_field_acceleration(solver, solver.pos[i], solver.vel[i]));
    Vec3 v = add(solver.vel[i], mul(acceleration, dt));
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
    if (wi > 0.0f) {
        atomic_add(&solver.pos[i], mul(corr, -wi));
    }
    if (wj > 0.0f) {
        atomic_add(&solver.pos[j], mul(corr, wj));
    }
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

__global__ void volume_value_partial_kernel(Solver solver, float* partials) {
    extern __shared__ float shared[];
    int global = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.0f;
    if (global < solver.cfg.triangle_count) {
        Int3 tri = solver.triangles[global];
        Vec3 a = solver.pos[tri.x];
        Vec3 b = solver.pos[tri.y];
        Vec3 c = solver.pos[tri.z];
        value = dot(a, cross(b, c)) / 6.0f;
    }
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = shared[0];
    }
}

__global__ void volume_gradient_incident_kernel(Solver solver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count) {
        return;
    }
    Vec3 grad{0.0f, 0.0f, 0.0f};
    int start = solver.volume_vertex_offsets[i];
    int end = solver.volume_vertex_offsets[i + 1];
    for (int cursor = start; cursor < end; ++cursor) {
        int t = solver.volume_vertex_triangles[cursor];
        if (t < 0 || t >= solver.cfg.triangle_count) {
            continue;
        }
        Int3 tri = solver.triangles[t];
        Vec3 a = solver.pos[tri.x];
        Vec3 b = solver.pos[tri.y];
        Vec3 c = solver.pos[tri.z];
        if (tri.x == i) {
            grad = add(grad, mul(cross(b, c), 1.0f / 6.0f));
        } else if (tri.y == i) {
            grad = add(grad, mul(cross(c, a), 1.0f / 6.0f));
        } else if (tri.z == i) {
            grad = add(grad, mul(cross(a, b), 1.0f / 6.0f));
        }
    }
    solver.volume_gradient[i] = grad;
}

__global__ void volume_denominator_partial_kernel(Solver solver, float* partials) {
    extern __shared__ float shared[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.0f;
    if (i < solver.cfg.vertex_count && solver.inv_mass[i] > 0.0f) {
        Vec3 grad = solver.volume_gradient[i];
        if (finite_vec(grad)) {
            value = solver.inv_mass[i] * dot(grad, grad);
        }
    }
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = shared[0];
    }
}

__global__ void volume_reduce_partials_kernel(
    const float* volume_partials,
    int volume_count,
    const float* denominator_partials,
    int denominator_count,
    float* accum
) {
    extern __shared__ float shared[];
    float* volume_shared = shared;
    float* denominator_shared = shared + blockDim.x;
    float volume = 0.0f;
    float denominator = 0.0f;
    for (int i = threadIdx.x; i < volume_count; i += blockDim.x) {
        volume += volume_partials[i];
    }
    for (int i = threadIdx.x; i < denominator_count; i += blockDim.x) {
        denominator += denominator_partials[i];
    }
    volume_shared[threadIdx.x] = volume;
    denominator_shared[threadIdx.x] = denominator;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            volume_shared[threadIdx.x] += volume_shared[threadIdx.x + stride];
            denominator_shared[threadIdx.x] += denominator_shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        accum[0] = volume_shared[0];
        accum[1] = denominator_shared[0];
    }
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
        Vec3 correction{0.0f, 0.0f, target_z - p.z};
        apply_external_collision_response(&p, &prev, correction);
    }
    if (solver.cfg.use_wall) {
        Vec3 o{solver.cfg.wall_origin[0], solver.cfg.wall_origin[1], solver.cfg.wall_origin[2]};
        Vec3 n = normalize({solver.cfg.wall_normal[0], solver.cfg.wall_normal[1], solver.cfg.wall_normal[2]});
        float d = dot(sub(p, o), n);
        if (d < margin) {
            diag_note_candidate(solver, d - margin);
            diag_note_resolved(solver);
            Vec3 correction = mul(n, margin - d);
            apply_external_collision_response(&p, &prev, correction);
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
            apply_external_collision_response(&p, &prev, correction);
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
    int* used_ccd_out,
    int prefer_outward_normal,
    Vec3* normal_out,
    Vec3* barycentric_out
) {
    Vec3 normal = normalize(cross(sub(b, a), sub(c, a)));
    if (!finite_vec(normal)) {
        return false;
    }

    Vec3 closest = closest_point_on_triangle(p, a, b, c);
    Vec3 barycentric = barycentric_on_triangle(closest, a, b, c);
    Vec3 delta = sub(p, closest);
    float d = norm(delta);
    float signed_prev = dot(sub(prev, a), normal);
    float signed_now = dot(sub(p, a), normal);

    if (d < contact_distance) {
        float delta_sq = dot(delta, delta);
        bool crossed_plane = signed_prev * signed_now < 0.0f;
        bool normal_contact = crossed_plane && fabsf(signed_now) < contact_distance;
        if (delta_sq <= 1.0e-12f) {
            normal_contact = true;
        }
        if (prefer_outward_normal) {
            *projected_out = add(closest, mul(normal, contact_distance));
            *used_ccd_out = signed_now < 0.0f ? 2 : (crossed_plane ? 1 : 0);
            if (normal_out) {
                *normal_out = normal;
            }
        } else if (normal_contact) {
            float side = signed_prev >= 0.0f ? 1.0f : -1.0f;
            if (fabsf(signed_prev) <= kEps && fabsf(signed_now) > kEps) {
                side = signed_now >= 0.0f ? 1.0f : -1.0f;
            }
            *projected_out = add(closest, mul(normal, side * contact_distance));
            *used_ccd_out = crossed_plane ? 1 : 0;
            if (normal_out) {
                *normal_out = mul(normal, side);
            }
        } else {
            Vec3 radial_normal = mul(delta, 1.0f / sqrtf(delta_sq));
            *projected_out = add(closest, mul(radial_normal, contact_distance));
            *used_ccd_out = crossed_plane ? 1 : 0;
            if (normal_out) {
                *normal_out = radial_normal;
            }
        }
        if (barycentric_out) {
            *barycentric_out = barycentric;
        }
        *score_out = d;
        *gap_out = d - contact_distance;
        return true;
    }

    float denom = signed_prev - signed_now;
    if (signed_prev * signed_now < 0.0f && fabsf(denom) > kEps) {
        float t = signed_prev / denom;
        if (t >= 0.0f && t <= 1.0f) {
            Vec3 hit = add(prev, mul(sub(p, prev), t));
            Vec3 closest_hit = closest_point_on_triangle(hit, a, b, c);
            if (norm(sub(hit, closest_hit)) <= fmaxf(contact_distance * 2.0f, 1.0e-4f)) {
                float side = prefer_outward_normal ? 1.0f : (signed_prev >= 0.0f ? 1.0f : -1.0f);
                *projected_out = add(closest_hit, mul(normal, side * contact_distance));
                if (normal_out) {
                    *normal_out = mul(normal, side);
                }
                if (barycentric_out) {
                    *barycentric_out = barycentric_on_triangle(closest_hit, a, b, c);
                }
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
    Vec3 best_normal{0.0f, 0.0f, 1.0f};
    Vec3 best_barycentric{1.0f, 0.0f, 0.0f};
    int best_triangle = -1;
    int best_hard_contact = 0;
    for (int t = 0; t < solver.cfg.static_triangle_count; ++t) {
        Vec3 a = solver.static_triangles[t * 3 + 0];
        Vec3 b = solver.static_triangles[t * 3 + 1];
        Vec3 c = solver.static_triangles[t * 3 + 2];
        Vec3 projected = p;
        float score = 1.0e30f;
        float gap = FLT_MAX;
        int used_ccd = 0;
        Vec3 contact_normal{0.0f, 0.0f, 1.0f};
        Vec3 barycentric{1.0f, 0.0f, 0.0f};
        if (static_triangle_contact_candidate(
                a,
                b,
                c,
                contact_distance,
                p,
                prev,
                &projected,
                &score,
                &gap,
                &used_ccd,
                solver.static_collider_complex ? 1 : 0,
                &contact_normal,
                &barycentric
            )) {
            diag_note_candidate(solver, gap);
            if (used_ccd == 1) {
                diag_note_ccd(solver);
            }
        }
        if (score < best_score && score < 1.0e30f) {
            found = true;
            best_score = score;
            best_projected = projected;
            best_normal = contact_normal;
            best_barycentric = barycentric;
            best_triangle = t;
            best_hard_contact = solver.static_collider_complex ? 1 : (used_ccd == 2 ? 1 : 0);
        }
    }
    if (found) {
        diag_note_resolved(solver);
        if (best_hard_contact) {
            p = best_projected;
            prev = p;
            store_external_contact_cache(
                solver,
                kExternalContactKindStatic,
                i,
                best_triangle,
                best_normal,
                best_barycentric,
                external_contact_distance(solver)
            );
        } else {
            solve_external_cached_contact(
                solver,
                kExternalContactKindStatic,
                i,
                best_triangle,
                best_normal,
                best_barycentric,
                best_projected,
                0,
                kStaticCollisionMaxCorrectionScale,
                kStaticCollisionCcdMaxCorrectionScale,
                &p,
                &prev
            );
        }
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
    Vec3 best_normal{0.0f, 0.0f, 1.0f};
    Vec3 best_barycentric{1.0f, 0.0f, 0.0f};
    int best_triangle = -1;
    int best_hard_contact = 0;
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
                        Vec3 contact_normal{0.0f, 0.0f, 1.0f};
                        Vec3 barycentric{1.0f, 0.0f, 0.0f};
                        if (static_triangle_contact_candidate(
                                a,
                                b,
                                c,
                                contact_distance,
                                p,
                                prev,
                                &projected,
                                &score,
                                &gap,
                                &used_ccd,
                                solver.static_collider_complex ? 1 : 0,
                                &contact_normal,
                                &barycentric
                            )) {
                            diag_note_candidate(solver, gap);
                            if (used_ccd == 1) {
                                diag_note_ccd(solver);
                            }
                        }
                        if (score < best_score && score < 1.0e30f) {
                            found = true;
                            best_score = score;
                            best_projected = projected;
                            best_normal = contact_normal;
                            best_barycentric = barycentric;
                            best_triangle = t;
                            best_hard_contact = solver.static_collider_complex ? 1 : (used_ccd == 2 ? 1 : 0);
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
        if (best_hard_contact) {
            p = best_projected;
            prev = p;
            store_external_contact_cache(
                solver,
                kExternalContactKindStatic,
                i,
                best_triangle,
                best_normal,
                best_barycentric,
                external_contact_distance(solver)
            );
        } else {
            solve_external_cached_contact(
                solver,
                kExternalContactKindStatic,
                i,
                best_triangle,
                best_normal,
                best_barycentric,
                best_projected,
                0,
                kStaticCollisionMaxCorrectionScale,
                kStaticCollisionCcdMaxCorrectionScale,
                &p,
                &prev
            );
        }
    }
    solver.pos[i] = p;
    solver.prev[i] = prev;
}

__global__ void build_dynamic_particle_hash_kernel(Solver solver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.dynamic_particle_count || !solver.dynamic_particles) {
        return;
    }
    DynamicParticle particle = solver.dynamic_particles[i];
    Vec3 p = particle.position;
    if (!finite_vec(p)) {
        return;
    }
    float cell_size = dynamic_particle_cell_size(solver);
    int cx = cell_coord(p.x, cell_size);
    int cy = cell_coord(p.y, cell_size);
    int cz = cell_coord(p.z, cell_size);
    int hash = hash_cell(cx, cy, cz, solver.dynamic_particle_hash_table_size);
    int entry = atomicAdd(solver.dynamic_particle_count_buffer, 1);
    if (entry >= solver.dynamic_particle_count) {
        diag_note_dynamic_particle_overflow(solver);
        return;
    }
    solver.dynamic_particle_index[entry] = i;
    solver.dynamic_particle_next[entry] = atomicExch(&solver.dynamic_particle_heads[hash], entry);
}

__global__ void dynamic_particle_collision_kernel(Solver solver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count || solver.inv_mass[i] <= 0.0f || solver.dynamic_particle_count <= 0) {
        return;
    }
    if (!solver.dynamic_particles
        || !solver.dynamic_particle_heads
        || !solver.dynamic_particle_next
        || !solver.dynamic_particle_index) {
        return;
    }
    Vec3 p = solver.pos[i];
    Vec3 prev = solver.prev[i];
    if (!finite_vec(p) || !finite_vec(prev)) {
        diag_note_nonfinite(solver);
        return;
    }
    float target_radius = external_contact_distance(solver);
    float cell_size = dynamic_particle_cell_size(solver);
    float query_radius = target_radius + fmaxf(solver.dynamic_particle_max_radius, 0.0f);
    float expand = fmaxf(query_radius, 1.0e-4f);
    int min_x = cell_coord(p.x - expand, cell_size);
    int min_y = cell_coord(p.y - expand, cell_size);
    int min_z = cell_coord(p.z - expand, cell_size);
    int max_x = cell_coord(p.x + expand, cell_size);
    int max_y = cell_coord(p.y + expand, cell_size);
    int max_z = cell_coord(p.z + expand, cell_size);
    int queried = 0;
    int contacts = 0;
    float wi = fmaxf(solver.inv_mass[i], 0.0f);

    for (int z = min_z; z <= max_z && queried < kMaxDynamicParticleQueryCells; ++z) {
        for (int y = min_y; y <= max_y && queried < kMaxDynamicParticleQueryCells; ++y) {
            for (int x = min_x; x <= max_x && queried < kMaxDynamicParticleQueryCells; ++x) {
                int hash = hash_cell(x, y, z, solver.dynamic_particle_hash_table_size);
                int entry = solver.dynamic_particle_heads[hash];
                while (entry >= 0) {
                    int particle_index = solver.dynamic_particle_index[entry];
                    if (particle_index >= 0 && particle_index < solver.dynamic_particle_count) {
                        DynamicParticle particle = solver.dynamic_particles[particle_index];
                        if (particle.phase >= 0 && finite_vec(particle.position)) {
                            float source_radius = fmaxf(particle.radius, 0.0f);
                            float min_dist = target_radius + source_radius;
                            if (min_dist > 0.0f) {
                                Vec3 delta = sub(p, particle.position);
                                float dist_sq = dot(delta, delta);
                                if (isfinite(dist_sq) && dist_sq < min_dist * min_dist) {
                                    float dist = sqrtf(fmaxf(dist_sq, 0.0f));
                                    float gap = dist - min_dist;
                                    diag_note_candidate(solver, gap);
                                    diag_note_dynamic_particle_candidate(solver);
                                    if (contacts < kMaxDynamicParticleNeighbors) {
                                        Vec3 normal = {0.0f, 0.0f, 1.0f};
                                        if (dist > 1.0e-7f) {
                                            normal = mul(delta, 1.0f / dist);
                                        } else {
                                            Vec3 motion = sub(p, prev);
                                            float motion_len = norm(motion);
                                            if (motion_len > 1.0e-7f && finite_vec(motion)) {
                                                normal = mul(motion, -1.0f / motion_len);
                                            }
                                        }
                                        float wj = fmaxf(particle.inv_mass, 0.0f);
                                        float total = wi + wj;
                                        float target_fraction = total > kEps ? wi / total : 0.0f;
                                        float depth = fmaxf(-gap, 0.0f);
                                        float correction_len = depth * target_fraction * kDynamicParticleCollisionRelaxation;
                                        float max_correction = fmaxf(
                                            kExternalCollisionMinCorrection,
                                            min_dist * kDynamicParticleMaxCorrectionScale
                                        );
                                        correction_len = fminf(correction_len, max_correction);
                                        if (correction_len > 0.0f && finite_vec(normal)) {
                                            Vec3 correction = mul(normal, correction_len);
                                            apply_external_collision_response(&p, &prev, correction);
                                            diag_note_resolved(solver);
                                            diag_note_dynamic_particle_contact(solver);
                                            ++contacts;
                                        }
                                    } else {
                                        diag_note_dynamic_particle_overflow(solver);
                                    }
                                }
                            }
                        }
                    }
                    entry = solver.dynamic_particle_next[entry];
                }
                ++queried;
            }
        }
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
    Vec3 best_normal{0.0f, 0.0f, 1.0f};
    Vec3 best_barycentric{1.0f, 0.0f, 0.0f};
    int best_triangle = -1;
    int best_barrier_contact = 0;
    for (int t = 0; t < solver.dynamic_triangle_count; ++t) {
        Vec3 a = solver.dynamic_triangles[t * 3 + 0];
        Vec3 b = solver.dynamic_triangles[t * 3 + 1];
        Vec3 c = solver.dynamic_triangles[t * 3 + 2];
        Vec3 projected = p;
        float score = 1.0e30f;
        float gap = FLT_MAX;
        int used_ccd = 0;
        Vec3 contact_normal{0.0f, 0.0f, 1.0f};
        Vec3 barycentric{1.0f, 0.0f, 0.0f};
        if (static_triangle_contact_candidate(
                a, b, c, contact_distance, p, prev, &projected, &score, &gap, &used_ccd, 0, &contact_normal, &barycentric
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
            best_normal = contact_normal;
            best_barycentric = barycentric;
            best_triangle = t;
            best_barrier_contact = used_ccd;
        }
    }
    if (found) {
        diag_note_resolved(solver);
        solve_external_cached_contact(
            solver,
            kExternalContactKindDynamic,
            i,
            best_triangle,
            best_normal,
            best_barycentric,
            best_projected,
            best_barrier_contact,
            kDynamicCollisionMaxCorrectionScale,
            kDynamicCollisionCcdMaxCorrectionScale,
            &p,
            &prev
        );
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
    Vec3 best_normal{0.0f, 0.0f, 1.0f};
    Vec3 best_barycentric{1.0f, 0.0f, 0.0f};
    int best_triangle = -1;
    int best_barrier_contact = 0;
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
                        Vec3 contact_normal{0.0f, 0.0f, 1.0f};
                        Vec3 barycentric{1.0f, 0.0f, 0.0f};
                        if (static_triangle_contact_candidate(
                                a, b, c, contact_distance, p, prev, &projected, &score, &gap, &used_ccd, 0, &contact_normal, &barycentric
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
                            best_normal = contact_normal;
                            best_barycentric = barycentric;
                            best_triangle = t;
                            best_barrier_contact = used_ccd;
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
        solve_external_cached_contact(
            solver,
            kExternalContactKindDynamic,
            i,
            best_triangle,
            best_normal,
            best_barycentric,
            best_projected,
            best_barrier_contact,
            kDynamicCollisionMaxCorrectionScale,
            kDynamicCollisionCcdMaxCorrectionScale,
            &p,
            &prev
        );
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

__global__ void build_self_edge_hash_kernel(Solver solver) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= solver.cfg.edge_count || !solver.edges || !solver.self_edge_heads || !solver.self_edge_next) {
        return;
    }
    Int2 edge = solver.edges[e];
    if (edge.x < 0 || edge.x >= solver.cfg.vertex_count || edge.y < 0 || edge.y >= solver.cfg.vertex_count) {
        return;
    }
    Vec3 mid = mul(add(solver.pos[edge.x], solver.pos[edge.y]), 0.5f);
    if (!finite_vec(mid)) {
        return;
    }
    float cell_size = self_cell_size(solver);
    int cx = cell_coord(mid.x, cell_size);
    int cy = cell_coord(mid.y, cell_size);
    int cz = cell_coord(mid.z, cell_size);
    int hash = hash_cell(cx, cy, cz, solver.self_edge_hash_table_size);
    solver.self_edge_next[e] = atomicExch(&solver.self_edge_heads[hash], e);
}

__global__ void self_edge_edge_collision_kernel(Solver solver) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= solver.cfg.edge_count || !solver.edges || !solver.self_edge_heads || !solver.self_edge_next) {
        return;
    }
    Int2 edge_a = solver.edges[e];
    if (edge_a.x < 0 || edge_a.x >= solver.cfg.vertex_count
        || edge_a.y < 0 || edge_a.y >= solver.cfg.vertex_count) {
        return;
    }
    const bool compacted_fast_source = self_fast_mode(solver)
        && solver.self_compaction_used
        && (solver.self_source_mode == kSelfSourceActive || solver.self_source_mode == kSelfSourceSuspect);
    const bool edge_a_in_source = self_source_edge_flag(solver, edge_a);
    if (compacted_fast_source && !edge_a_in_source) {
        return;
    }
    float wx0 = solver.inv_mass[edge_a.x];
    float wx1 = solver.inv_mass[edge_a.y];
    if (wx0 <= 0.0f && wx1 <= 0.0f) {
        return;
    }

    Vec3 a0 = solver.pos[edge_a.x];
    Vec3 a1 = solver.pos[edge_a.y];
    if (!finite_vec(a0) || !finite_vec(a1)) {
        diag_note_nonfinite(solver);
        return;
    }
    Vec3 mid = mul(add(a0, a1), 0.5f);
    float thickness = self_contact_distance(solver);
    float cell_size = self_cell_size(solver);
    int max_neighbors = solver.cfg.max_self_collision_neighbors > 1 ? solver.cfg.max_self_collision_neighbors : 1;
    if (solver.self_recovery_mode) {
        max_neighbors = min(max_neighbors * 2, 256);
    }
    int candidates = 0;
    int scanned = 0;
    int scan_limit = max_neighbors * (self_fast_mode(solver) ? 8 : 32);
    int base_x = cell_coord(mid.x, cell_size);
    int base_y = cell_coord(mid.y, cell_size);
    int base_z = cell_coord(mid.z, cell_size);

    for (int dz = -2; dz <= 2; ++dz) {
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                int hash = hash_cell(base_x + dx, base_y + dy, base_z + dz, solver.self_edge_hash_table_size);
                int other = solver.self_edge_heads[hash];
                while (other >= 0 && candidates < max_neighbors && scanned < scan_limit) {
                    ++scanned;
                    if (other != e) {
                        Int2 edge_b = solver.edges[other];
                        const bool edge_b_in_source = self_source_edge_flag(solver, edge_b);
                        const bool should_process_pair = other > e || (compacted_fast_source && edge_a_in_source && !edge_b_in_source);
                        if (!should_process_pair) {
                            other = solver.self_edge_next[other];
                            continue;
                        }
                        if (edge_b.x >= 0 && edge_b.x < solver.cfg.vertex_count
                            && edge_b.y >= 0 && edge_b.y < solver.cfg.vertex_count
                            && !rest_edges_neighbor(solver, edge_a, edge_b)) {
                            Vec3 b0 = solver.pos[edge_b.x];
                            Vec3 b1 = solver.pos[edge_b.y];
                            if (finite_vec(b0) && finite_vec(b1)) {
                                float s = 0.0f;
                                float t = 0.0f;
                                closest_segment_parameters(a0, a1, b0, b1, &s, &t);
                                Vec3 pa = add(a0, mul(sub(a1, a0), s));
                                Vec3 pb = add(b0, mul(sub(b1, b0), t));
                                Vec3 delta = sub(pa, pb);
                                float d2 = dot(delta, delta);
                                float d = sqrtf(fmaxf(d2, 0.0f));
                                if (self_coarse_distance_ok(d, thickness)) {
                                    float gap = d - thickness;
                                    diag_note_gap(solver, gap);
                                    Vec3 prev_a0 = solver.prev[edge_a.x];
                                    Vec3 prev_a1 = solver.prev[edge_a.y];
                                    Vec3 prev_b0 = solver.prev[edge_b.x];
                                    Vec3 prev_b1 = solver.prev[edge_b.y];
                                    Vec3 prev_pa = add(prev_a0, mul(sub(prev_a1, prev_a0), s));
                                    Vec3 prev_pb = add(prev_b0, mul(sub(prev_b1, prev_b0), t));
                                    Vec3 previous_delta = sub(prev_pa, prev_pb);
                                    Vec3 normal;
                                    float contact_distance = d;
                                    if (d2 > kEps) {
                                        normal = mul(delta, 1.0f / fmaxf(d, kEps));
                                    } else {
                                        Vec3 rest_a = sub(solver.rest[edge_a.y], solver.rest[edge_a.x]);
                                        Vec3 rest_b = sub(solver.rest[edge_b.y], solver.rest[edge_b.x]);
                                        normal = cross(rest_a, rest_b);
                                        if (dot(normal, normal) <= kEps) {
                                            normal = sub(
                                                mul(add(solver.rest[edge_a.x], solver.rest[edge_a.y]), 0.5f),
                                                mul(add(solver.rest[edge_b.x], solver.rest[edge_b.y]), 0.5f)
                                            );
                                        }
                                        if (dot(normal, normal) <= kEps) {
                                            normal = {0.0f, 0.0f, 1.0f};
                                        } else {
                                            normal = normalize(normal);
                                        }
                                        contact_distance = 0.0f;
                                    }
                                    if (self_should_project_contact(gap, delta, previous_delta, normal)) {
                                        diag_note_effective_candidate(solver);
                                        ++candidates;
                                        if (gap < 0.0f) {
                                            float ca0 = 1.0f - s;
                                            float ca1 = s;
                                            float cb0 = 1.0f - t;
                                            float cb1 = t;
                                            float wb0_mass = solver.inv_mass[edge_b.x];
                                            float wb1_mass = solver.inv_mass[edge_b.y];
                                            float total = ca0 * ca0 * wx0
                                                + ca1 * ca1 * wx1
                                                + cb0 * cb0 * wb0_mass
                                                + cb1 * cb1 * wb1_mass;
                                            if (total > 0.0f) {
                                                Vec3 correction = mul(normal, self_projection_relaxation(solver) * (thickness - contact_distance) / total);
                                                apply_self_collision_correction(solver, edge_a.x, mul(correction, wx0 * ca0));
                                                apply_self_collision_correction(solver, edge_a.y, mul(correction, wx1 * ca1));
                                                apply_self_collision_correction(solver, edge_b.x, mul(correction, -wb0_mass * cb0));
                                                apply_self_collision_correction(solver, edge_b.y, mul(correction, -wb1_mass * cb1));
                                                diag_note_resolved(solver);
                                            }
                                        }
                                    }
                                }
                            } else {
                                diag_note_nonfinite(solver);
                            }
                        }
                    }
                    other = solver.self_edge_next[other];
                }
            }
        }
    }
}

__global__ void self_particle_collision_kernel(Solver solver) {
    int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= self_vertex_source_count(solver)) {
        return;
    }
    int i = self_vertex_source_index(solver, ordinal);
    if (i < 0 || i >= solver.cfg.vertex_count || solver.inv_mass[i] <= 0.0f) {
        return;
    }
    if (!self_compaction_uses_vertex_list(solver) && self_sleep_source_vertex_skipped(solver, i)) {
        diag_note_self_source_skipped(solver);
        return;
    }
    Vec3 p = solver.pos[i];
    float wi = solver.inv_mass[i];
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
                        if (!self_should_project_contact(gap, delta, previous_delta, normal)) {
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
                                Vec3 correction_i = mul(correction, wi);
                                Vec3 correction_j = mul(correction, -wj);
                                apply_self_collision_correction(solver, i, correction_i);
                                apply_self_collision_correction(solver, j, correction_j);
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

__device__ void build_self_surface_sample_hash_entry(Solver solver, int sample) {
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

__global__ void build_self_surface_sample_hash_kernel(Solver solver) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= solver.self_sample_count) {
        return;
    }
    build_self_surface_sample_hash_entry(solver, sample);
}

__global__ void clear_self_surface_sample_hash_if_dirty_kernel(Solver solver) {
    if (!solver.self_sample_hash_dirty || solver.self_sample_hash_dirty[0] == 0) {
        return;
    }
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < solver.self_sample_hash_table_size) {
        solver.self_sample_heads[index] = -1;
    }
}

__global__ void build_self_surface_sample_hash_if_dirty_kernel(Solver solver) {
    if (!solver.self_sample_hash_dirty || solver.self_sample_hash_dirty[0] == 0) {
        return;
    }
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= solver.self_sample_count) {
        return;
    }
    build_self_surface_sample_hash_entry(solver, sample);
}

__global__ void build_self_triangle_hash_kernel(Solver solver) {
    int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= self_triangle_hash_source_count(solver)
        || !solver.self_tri_heads
        || !solver.self_tri_entry_next
        || !solver.self_tri_entry_index
        || !solver.self_tri_entry_count) {
        return;
    }
    int t = self_triangle_hash_source_index(solver, ordinal);
    if (t < 0 || t >= solver.cfg.triangle_count) {
        return;
    }
    Int3 tri = solver.triangles[t];
    if (tri.x < 0 || tri.x >= solver.cfg.vertex_count
        || tri.y < 0 || tri.y >= solver.cfg.vertex_count
        || tri.z < 0 || tri.z >= solver.cfg.vertex_count) {
        return;
    }
    Vec3 a = solver.pos[tri.x];
    Vec3 b = solver.pos[tri.y];
    Vec3 c = solver.pos[tri.z];
    if (!finite_vec(a) || !finite_vec(b) || !finite_vec(c)) {
        return;
    }
    float cell_size = self_cell_size(solver);
    float contact_distance = self_contact_distance(solver);
    float expand = fmaxf(contact_distance * 2.0f, cell_size * 0.5f);
    int min_x = cell_coord(fminf(fminf(a.x, b.x), c.x) - expand, cell_size);
    int min_y = cell_coord(fminf(fminf(a.y, b.y), c.y) - expand, cell_size);
    int min_z = cell_coord(fminf(fminf(a.z, b.z), c.z) - expand, cell_size);
    int max_x = cell_coord(fmaxf(fmaxf(a.x, b.x), c.x) + expand, cell_size);
    int max_y = cell_coord(fmaxf(fmaxf(a.y, b.y), c.y) + expand, cell_size);
    int max_z = cell_coord(fmaxf(fmaxf(a.z, b.z), c.z) + expand, cell_size);
    int inserted = 0;
    for (int z = min_z; z <= max_z && inserted < kMaxSelfTriangleHashCells; ++z) {
        for (int y = min_y; y <= max_y && inserted < kMaxSelfTriangleHashCells; ++y) {
            for (int x = min_x; x <= max_x && inserted < kMaxSelfTriangleHashCells; ++x) {
                int entry = atomicAdd(solver.self_tri_entry_count, 1);
                if (entry >= solver.self_tri_entry_capacity) {
                    return;
                }
                int hash = hash_cell(x, y, z, solver.self_tri_hash_table_size);
                solver.self_tri_entry_index[entry] = t;
                solver.self_tri_entry_next[entry] = atomicExch(&solver.self_tri_heads[hash], entry);
                ++inserted;
            }
        }
    }
}

__global__ void self_edge_surface_collision_kernel(Solver solver) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= solver.cfg.edge_count || !solver.edges || !solver.self_sample_heads || !solver.self_sample_next) {
        return;
    }
    Int2 edge = solver.edges[e];
    if (edge.x < 0 || edge.x >= solver.cfg.vertex_count
        || edge.y < 0 || edge.y >= solver.cfg.vertex_count) {
        return;
    }
    if (self_fast_mode(solver)
        && solver.self_compaction_used
        && (solver.self_source_mode == kSelfSourceActive || solver.self_source_mode == kSelfSourceSuspect)
        && !self_source_edge_flag(solver, edge)) {
        return;
    }
    float wx0 = solver.inv_mass[edge.x];
    float wx1 = solver.inv_mass[edge.y];
    if (wx0 <= 0.0f && wx1 <= 0.0f) {
        return;
    }
    Vec3 p0 = solver.pos[edge.x];
    Vec3 p1 = solver.pos[edge.y];
    if (!finite_vec(p0) || !finite_vec(p1)) {
        diag_note_nonfinite(solver);
        return;
    }

    float thickness = self_contact_distance(solver);
    float cell_size = self_cell_size(solver);
    int max_neighbors = solver.cfg.max_self_collision_neighbors > 1 ? solver.cfg.max_self_collision_neighbors : 1;
    if (solver.self_recovery_mode) {
        max_neighbors = min(max_neighbors * 2, 256);
    }
    int candidates = 0;
    int scanned = 0;
    int scan_limit = max_neighbors * 16;

    for (int anchor_index = 0; anchor_index < 1 && candidates < max_neighbors; ++anchor_index) {
        float anchor_t = 0.5f;
        Vec3 query = add(p0, mul(sub(p1, p0), anchor_t));
        int base_x = cell_coord(query.x, cell_size);
        int base_y = cell_coord(query.y, cell_size);
        int base_z = cell_coord(query.z, cell_size);
        for (int dz = -1; dz <= 1 && candidates < max_neighbors && scanned < scan_limit; ++dz) {
            for (int dy = -1; dy <= 1 && candidates < max_neighbors && scanned < scan_limit; ++dy) {
                for (int dx = -1; dx <= 1 && candidates < max_neighbors && scanned < scan_limit; ++dx) {
                    int hash = hash_cell(base_x + dx, base_y + dy, base_z + dz, solver.self_sample_hash_table_size);
                    int sample = solver.self_sample_heads[hash];
                    while (sample >= 0 && candidates < max_neighbors && scanned < scan_limit) {
                        ++scanned;
                        int tri_index = self_sample_triangle_index(solver, sample);
                        if (tri_index >= 0 && tri_index < solver.cfg.triangle_count) {
                            Int3 tri = solver.triangles[tri_index];
                            if (!rest_edge_surface_neighbor(solver, edge, tri)) {
                                Vec3 a = solver.pos[tri.x];
                                Vec3 b = solver.pos[tri.y];
                                Vec3 c = solver.pos[tri.z];
                                if (finite_vec(a) && finite_vec(b) && finite_vec(c)) {
                                    float edge_t;
                                    float wa;
                                    float wb;
                                    float wc;
                                    float distance;
                                    Vec3 edge_point;
                                    Vec3 tri_point;
                                    closest_segment_triangle_contact(
                                        p0,
                                        p1,
                                        a,
                                        b,
                                        c,
                                        &edge_t,
                                        &wa,
                                        &wb,
                                        &wc,
                                        &distance,
                                        &edge_point,
                                        &tri_point
                                    );
                                    if (self_coarse_distance_ok(distance, thickness)) {
                                        float gap = distance - thickness;
                                        diag_note_gap(solver, gap);
                                        Vec3 surface_normal = stable_triangle_normal(
                                            a,
                                            b,
                                            c,
                                            solver.rest[tri.x],
                                            solver.rest[tri.y],
                                            solver.rest[tri.z]
                                        );
                                        Vec3 prev_edge_point = add(
                                            solver.prev[edge.x],
                                            mul(sub(solver.prev[edge.y], solver.prev[edge.x]), edge_t)
                                        );
                                        Vec3 prev_tri_point = weighted_triangle_point(
                                            solver.prev[tri.x],
                                            solver.prev[tri.y],
                                            solver.prev[tri.z],
                                            wa,
                                            wb,
                                            wc
                                        );
                                        Vec3 delta = sub(edge_point, tri_point);
                                        Vec3 previous_delta = sub(prev_edge_point, prev_tri_point);
                                        float contact_distance = 0.0f;
                                        Vec3 normal = self_collision_normal(
                                            delta,
                                            surface_normal,
                                            previous_delta,
                                            thickness,
                                            &contact_distance
                                        );
                                        if (self_should_project_contact(gap, delta, previous_delta, normal)) {
                                            diag_note_effective_candidate(solver);
                                            ++candidates;
                                            if (gap < 0.0f) {
                                                float edge_w0 = 1.0f - edge_t;
                                                float edge_w1 = edge_t;
                                                float tri_weight = weighted_inv_mass(solver, tri, wa, wb, wc);
                                                float total = edge_w0 * edge_w0 * wx0
                                                    + edge_w1 * edge_w1 * wx1
                                                    + tri_weight;
                                                if (total > 0.0f) {
                                                    Vec3 correction = mul(
                                                        normal,
                                                        self_projection_relaxation(solver) * (thickness - contact_distance) / total
                                                    );
                                                    apply_self_collision_correction(solver, edge.x, mul(correction, wx0 * edge_w0));
                                                    apply_self_collision_correction(solver, edge.y, mul(correction, wx1 * edge_w1));
                                                    apply_self_collision_correction(solver, tri.x, mul(correction, -solver.inv_mass[tri.x] * wa));
                                                    apply_self_collision_correction(solver, tri.y, mul(correction, -solver.inv_mass[tri.y] * wb));
                                                    apply_self_collision_correction(solver, tri.z, mul(correction, -solver.inv_mass[tri.z] * wc));
                                                    if (solver.self_sample_hash_dirty) {
                                                        atomicExch(solver.self_sample_hash_dirty, 1);
                                                    }
                                                    diag_note_resolved(solver);
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    diag_note_nonfinite(solver);
                                }
                            }
                        }
                        sample = solver.self_sample_next[sample];
                    }
                }
            }
        }
    }
}

int dynamic_collision_pass_count(const Solver* solver) {
    if (!solver || solver->dynamic_triangle_count <= 0 || solver->cfg.vertex_count <= 0) {
        return 0;
    }
    const long long estimated_work =
        static_cast<long long>(solver->cfg.vertex_count) * static_cast<long long>(solver->dynamic_triangle_count);
    if (solver->dynamic_triangle_count <= kDynamicCollisionTwoPassTriangleLimit
        || estimated_work <= kDynamicCollisionTwoPassWorkLimit) {
        return kMaxDynamicCollisionPasses;
    }
    return 1;
}

__device__ int self_vs_pair_device_count(Solver solver) {
    if (!solver.self_vs_pair_counts) {
        return 0;
    }
    int count = solver.self_vs_pair_counts[kSelfVsPairCount];
    if (count < 0) {
        return 0;
    }
    return count < solver.self_vs_pair_capacity ? count : solver.self_vs_pair_capacity;
}

__device__ void append_self_vs_pair(Solver solver, int vertex, int sample) {
    if (!solver.self_vs_pairs || !solver.self_vs_pair_counts || solver.self_vs_pair_capacity <= 0) {
        return;
    }
    int slot = atomicAdd(&solver.self_vs_pair_counts[kSelfVsPairCount], 1);
    if (slot < solver.self_vs_pair_capacity) {
        solver.self_vs_pairs[slot] = {vertex, sample};
    } else {
        atomicExch(&solver.self_vs_pair_counts[kSelfVsPairOverflow], 1);
    }
}

__global__ void build_self_vertex_surface_pairs_kernel(Solver solver, int store_all_coarse) {
    int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= self_vertex_source_count(solver)) {
        return;
    }
    int i = self_vertex_source_index(solver, ordinal);
    if (i < 0 || i >= solver.cfg.vertex_count || solver.inv_mass[i] <= 0.0f) {
        return;
    }
    Vec3 p = solver.pos[i];
    if (!finite_vec(p)) {
        return;
    }
    float thickness = self_contact_distance(solver);
    float cell_size = self_cell_size(solver);
    int max_neighbors = solver.cfg.max_self_collision_neighbors > 1 ? solver.cfg.max_self_collision_neighbors : 1;
    if (solver.self_recovery_mode) {
        max_neighbors = min(max_neighbors * 2, 256);
    }
    int candidates = 0;
    int scanned = 0;
    int scan_limit = max_neighbors * (self_fast_mode(solver) ? 8 : 32);
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
                        bool append_pair = store_all_coarse != 0;
                        if (!append_pair) {
                            float gap = d_linear - thickness;
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
                            Vec3 normal = self_collision_normal(delta, surface_normal, previous_delta, thickness, &d);
                            append_pair = self_should_project_contact(gap, delta, previous_delta, normal);
                        }
                        if (append_pair) {
                            append_self_vs_pair(solver, i, sample);
                            ++candidates;
                        }
                    }
                    sample = solver.self_sample_next[sample];
                }
            }
        }
    }
}

__global__ void self_vertex_surface_pair_project_kernel(Solver solver) {
    int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_index >= self_vs_pair_device_count(solver)) {
        return;
    }
    Int2 pair = solver.self_vs_pairs[pair_index];
    int i = pair.x;
    int sample = pair.y;
    if (i < 0 || i >= solver.cfg.vertex_count || sample < 0 || sample >= solver.self_sample_count) {
        return;
    }
    if (solver.inv_mass[i] <= 0.0f) {
        return;
    }
    int tri_index = self_sample_triangle_index(solver, sample);
    int kind = self_sample_kind(solver, sample, tri_index);
    Int3 tri = solver.triangles[tri_index];
    float wa;
    float wb;
    float wc;
    self_surface_sample_weights(kind, &wa, &wb, &wc);
    if (rest_surface_neighbor(solver, i, tri, wa, wb, wc)) {
        return;
    }
    Vec3 p = solver.pos[i];
    Vec3 a = solver.pos[tri.x];
    Vec3 b = solver.pos[tri.y];
    Vec3 c = solver.pos[tri.z];
    Vec3 sample_pos = weighted_triangle_point(a, b, c, wa, wb, wc);
    Vec3 delta = sub(p, sample_pos);
    float d2 = dot(delta, delta);
    float d_linear = sqrtf(fmaxf(d2, 0.0f));
    float thickness = self_contact_distance(solver);
    if (!self_coarse_distance_ok(d_linear, thickness)) {
        return;
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
    Vec3 normal = self_collision_normal(delta, surface_normal, previous_delta, thickness, &d);
    if (!self_should_project_contact(gap, delta, previous_delta, normal)) {
        return;
    }
    diag_note_effective_candidate(solver);
    if (gap < 0.0f && dot(previous_delta, surface_normal) * dot(delta, surface_normal) < 0.0f) {
        diag_note_ccd(solver);
    }
    if (gap < 0.0f) {
        float wi = solver.inv_mass[i];
        float wx = solver.inv_mass[tri.x];
        float wy = solver.inv_mass[tri.y];
        float wz = solver.inv_mass[tri.z];
        float sample_weight = weighted_inv_mass(solver, tri, wa, wb, wc);
        float total = wi + sample_weight;
        if (total > 0.0f) {
            Vec3 correction = mul(normal, self_projection_relaxation(solver) * (thickness - d) / total);
            if (solver.self_sample_hash_dirty) {
                atomicExch(solver.self_sample_hash_dirty, 1);
            }
            Vec3 correction_i = mul(correction, wi);
            Vec3 correction_x = mul(correction, -wx * wa);
            Vec3 correction_y = mul(correction, -wy * wb);
            Vec3 correction_z = mul(correction, -wz * wc);
            apply_self_collision_correction(solver, i, correction_i);
            apply_self_collision_correction(solver, tri.x, correction_x);
            apply_self_collision_correction(solver, tri.y, correction_y);
            apply_self_collision_correction(solver, tri.z, correction_z);
            diag_note_resolved(solver);
        }
    }
}

__global__ void probe_self_vertex_surface_pairs_kernel(Solver solver) {
    int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_index >= self_vs_pair_device_count(solver)) {
        return;
    }
    Int2 pair = solver.self_vs_pairs[pair_index];
    int i = pair.x;
    int sample = pair.y;
    if (i < 0 || i >= solver.cfg.vertex_count || sample < 0 || sample >= solver.self_sample_count) {
        return;
    }
    if (solver.inv_mass[i] <= 0.0f) {
        return;
    }
    int tri_index = self_sample_triangle_index(solver, sample);
    int kind = self_sample_kind(solver, sample, tri_index);
    Int3 tri = solver.triangles[tri_index];
    float wa;
    float wb;
    float wc;
    self_surface_sample_weights(kind, &wa, &wb, &wc);
    if (rest_surface_neighbor(solver, i, tri, wa, wb, wc)) {
        return;
    }
    Vec3 p = solver.pos[i];
    Vec3 a = solver.pos[tri.x];
    Vec3 b = solver.pos[tri.y];
    Vec3 c = solver.pos[tri.z];
    Vec3 sample_pos = weighted_triangle_point(a, b, c, wa, wb, wc);
    Vec3 delta = sub(p, sample_pos);
    float d_linear = sqrtf(fmaxf(dot(delta, delta), 0.0f));
    float thickness = self_contact_distance(solver);
    if (!self_coarse_distance_ok(d_linear, thickness)) {
        return;
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
    if (!self_should_project_contact(gap, delta, previous_delta, normal)) {
        return;
    }
    diag_note_effective_candidate(solver);
    if (gap < 0.0f && dot(previous_delta, surface_normal) * dot(delta, surface_normal) < 0.0f) {
        diag_note_ccd(solver);
    }
}

__global__ void self_vertex_triangle_collision_kernel(Solver solver, int project_contacts) {
    int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= self_vertex_source_count(solver)
        || !solver.self_tri_heads
        || !solver.self_tri_entry_next
        || !solver.self_tri_entry_index) {
        return;
    }
    int i = self_vertex_source_index(solver, ordinal);
    if (i < 0 || i >= solver.cfg.vertex_count || solver.inv_mass[i] <= 0.0f) {
        return;
    }
    if (!self_compaction_uses_vertex_list(solver) && self_sleep_source_vertex_skipped(solver, i)) {
        diag_note_self_source_skipped(solver);
        return;
    }
    Vec3 p = solver.pos[i];
    Vec3 prev = solver.prev[i];
    if (!finite_vec(p) || !finite_vec(prev)) {
        diag_note_nonfinite(solver);
        return;
    }
    float thickness = self_contact_distance(solver);
    float cell_size = self_cell_size(solver);
    float expand = fmaxf(thickness * 2.0f, 1.0e-4f);
    int min_x = cell_coord(fminf(p.x, prev.x) - expand, cell_size);
    int min_y = cell_coord(fminf(p.y, prev.y) - expand, cell_size);
    int min_z = cell_coord(fminf(p.z, prev.z) - expand, cell_size);
    int max_x = cell_coord(fmaxf(p.x, prev.x) + expand, cell_size);
    int max_y = cell_coord(fmaxf(p.y, prev.y) + expand, cell_size);
    int max_z = cell_coord(fmaxf(p.z, prev.z) + expand, cell_size);
    int queried = 0;
    int candidates = 0;
    int max_neighbors = solver.cfg.max_self_collision_neighbors > 1 ? solver.cfg.max_self_collision_neighbors : 1;
    if (solver.self_recovery_mode) {
        max_neighbors = min(max_neighbors * 2, 256);
    }
    int visited[kMaxSelfVertexTriangleCandidates];
    int visited_count = 0;

    for (int z = min_z; z <= max_z && queried < kMaxSelfVertexTriangleQueryCells; ++z) {
        for (int y = min_y; y <= max_y && queried < kMaxSelfVertexTriangleQueryCells; ++y) {
            for (int x = min_x; x <= max_x && queried < kMaxSelfVertexTriangleQueryCells; ++x) {
                int hash = hash_cell(x, y, z, solver.self_tri_hash_table_size);
                int entry = solver.self_tri_heads[hash];
                while (entry >= 0 && candidates < max_neighbors && visited_count < kMaxSelfVertexTriangleCandidates) {
                    int t = solver.self_tri_entry_index[entry];
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
                        if (t >= 0 && t < solver.cfg.triangle_count) {
                            Int3 tri = solver.triangles[t];
                            if (tri.x >= 0 && tri.x < solver.cfg.vertex_count
                                && tri.y >= 0 && tri.y < solver.cfg.vertex_count
                                && tri.z >= 0 && tri.z < solver.cfg.vertex_count) {
                                Vec3 a = solver.pos[tri.x];
                                Vec3 b = solver.pos[tri.y];
                                Vec3 c = solver.pos[tri.z];
                                if (finite_vec(a) && finite_vec(b) && finite_vec(c)) {
                                    Vec3 closest = closest_point_on_triangle(p, a, b, c);
                                    Vec3 barycentric = barycentric_on_triangle(closest, a, b, c);
                                    float wa = barycentric.x;
                                    float wb = barycentric.y;
                                    float wc = barycentric.z;
                                    if (!rest_surface_neighbor(solver, i, tri, wa, wb, wc)) {
                                        Vec3 delta = sub(p, closest);
                                        float d_linear = sqrtf(fmaxf(dot(delta, delta), 0.0f));
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
                                        Vec3 previous_delta = sub(prev, prev_sample);
                                        float contact_distance = 0.0f;
                                        Vec3 normal = self_collision_normal(
                                            delta,
                                            surface_normal,
                                            previous_delta,
                                            thickness,
                                            &contact_distance
                                        );
                                        float coarse_distance = contact_distance;
                                        if (self_coarse_distance_ok(coarse_distance, thickness)) {
                                            float gap = coarse_distance - thickness;
                                            diag_note_gap(solver, gap);
                                            if (self_should_project_contact(gap, delta, previous_delta, normal)) {
                                                diag_note_effective_candidate(solver);
                                                ++candidates;
                                                if (gap < 0.0f && dot(previous_delta, surface_normal) * dot(delta, surface_normal) < 0.0f) {
                                                    diag_note_ccd(solver);
                                                }
                                                if (project_contacts != 0 && gap < 0.0f) {
                                                    float wi = solver.inv_mass[i];
                                                    float wx = solver.inv_mass[tri.x];
                                                    float wy = solver.inv_mass[tri.y];
                                                    float wz = solver.inv_mass[tri.z];
                                                    float tri_weight = weighted_inv_mass(solver, tri, wa, wb, wc);
                                                    float total = wi + tri_weight;
                                                    if (total > 0.0f) {
                                                        Vec3 correction = mul(
                                                            normal,
                                                            self_projection_relaxation(solver) * (thickness - contact_distance) / total
                                                        );
                                                        Vec3 correction_i = mul(correction, wi);
                                                        Vec3 correction_x = mul(correction, -wx * wa);
                                                        Vec3 correction_y = mul(correction, -wy * wb);
                                                        Vec3 correction_z = mul(correction, -wz * wc);
                                                        if (self_fast_mode(solver) && !solver.self_recovery_mode) {
                                                            apply_self_collision_correction_without_frontier(solver, i, correction_i);
                                                        } else {
                                                            apply_self_collision_correction(solver, i, correction_i);
                                                        }
                                                        apply_self_collision_correction_untracked(solver, tri.x, correction_x);
                                                        apply_self_collision_correction_untracked(solver, tri.y, correction_y);
                                                        apply_self_collision_correction_untracked(solver, tri.z, correction_z);
                                                        diag_note_resolved(solver);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                } else {
                                    diag_note_nonfinite(solver);
                                }
                            }
                        }
                    }
                    entry = solver.self_tri_entry_next[entry];
                }
                ++queried;
            }
        }
    }
}

__global__ void self_vertex_surface_collision_kernel(Solver solver) {
    int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= self_vertex_source_count(solver)) {
        return;
    }
    int i = self_vertex_source_index(solver, ordinal);
    if (i < 0 || i >= solver.cfg.vertex_count || solver.inv_mass[i] <= 0.0f) {
        return;
    }
    if (!self_compaction_uses_vertex_list(solver) && self_sleep_source_vertex_skipped(solver, i)) {
        diag_note_self_source_skipped(solver);
        return;
    }
    Vec3 p = solver.pos[i];
    if (!finite_vec(p)) {
        return;
    }
    float wi = solver.inv_mass[i];
    float thickness = self_contact_distance(solver);
    float cell_size = self_cell_size(solver);
    int max_neighbors = solver.cfg.max_self_collision_neighbors > 1 ? solver.cfg.max_self_collision_neighbors : 1;
    if (solver.self_recovery_mode) {
        max_neighbors = min(max_neighbors * 2, 256);
    }
    int candidates = 0;
    int scanned = 0;
    int scan_limit = max_neighbors * (self_fast_mode(solver) ? 8 : 32);
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
                        if (!self_should_project_contact(gap, delta, previous_delta, normal)) {
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
                                if (solver.self_sample_hash_dirty) {
                                    atomicExch(solver.self_sample_hash_dirty, 1);
                                }
                                Vec3 correction_i = mul(correction, wi);
                                Vec3 correction_x = mul(correction, -wx * wa);
                                Vec3 correction_y = mul(correction, -wy * wb);
                                Vec3 correction_z = mul(correction, -wz * wc);
                                apply_self_collision_correction(solver, i, correction_i);
                                apply_self_collision_correction(solver, tri.x, correction_x);
                                apply_self_collision_correction(solver, tri.y, correction_y);
                                apply_self_collision_correction(solver, tri.z, correction_z);
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
    int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= self_sample_source_count(solver)) {
        return;
    }
    int sample_a = self_sample_source_index(solver, ordinal);
    if (sample_a < 0 || sample_a >= solver.self_sample_count) {
        return;
    }
    int tri_index_a = self_sample_triangle_index(solver, sample_a);
    if (!self_compaction_uses_sample_list(solver) && self_sleep_source_triangle_skipped(solver, tri_index_a)) {
        diag_note_self_source_skipped(solver);
        return;
    }
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
    float thickness = self_contact_distance(solver);
    float cell_size = self_cell_size(solver);
    int max_neighbors = solver.cfg.max_self_collision_neighbors > 1 ? solver.cfg.max_self_collision_neighbors : 1;
    if (solver.self_recovery_mode) {
        max_neighbors = min(max_neighbors * 2, 256);
    }
    int candidates = 0;
    int scanned = 0;
    int scan_limit = max_neighbors * (self_fast_mode(solver) ? 8 : 32);
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
                            if (!self_should_project_contact(gap, delta, previous_delta, normal)) {
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
                                    Vec3 correction_ax = mul(correction, solver.inv_mass[tri_a.x] * aa);
                                    Vec3 correction_ay = mul(correction, solver.inv_mass[tri_a.y] * ab);
                                    Vec3 correction_az = mul(correction, solver.inv_mass[tri_a.z] * ac);
                                    Vec3 correction_bx = mul(correction, -solver.inv_mass[tri_b.x] * ba);
                                    Vec3 correction_by = mul(correction, -solver.inv_mass[tri_b.y] * bb);
                                    Vec3 correction_bz = mul(correction, -solver.inv_mass[tri_b.z] * bc);
                                    apply_self_collision_correction(solver, tri_a.x, correction_ax);
                                    apply_self_collision_correction(solver, tri_a.y, correction_ay);
                                    apply_self_collision_correction(solver, tri_a.z, correction_az);
                                    apply_self_collision_correction(solver, tri_b.x, correction_bx);
                                    apply_self_collision_correction(solver, tri_b.y, correction_by);
                                    apply_self_collision_correction(solver, tri_b.z, correction_bz);
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
    int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= self_vertex_source_count(solver)) {
        return;
    }
    int i = self_vertex_source_index(solver, ordinal);
    if (i < 0 || i >= solver.cfg.vertex_count || solver.inv_mass[i] <= 0.0f) {
        return;
    }
    if (!self_compaction_uses_vertex_list(solver) && self_sleep_source_vertex_skipped(solver, i)) {
        diag_note_self_source_skipped(solver);
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
                        if (!self_should_project_contact(gap, delta, previous_delta, normal)) {
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
    int ordinal = blockIdx.x * blockDim.x + threadIdx.x;
    if (ordinal >= self_vertex_source_count(solver)) {
        return;
    }
    int i = self_vertex_source_index(solver, ordinal);
    if (i < 0 || i >= solver.cfg.vertex_count || solver.inv_mass[i] <= 0.0f) {
        return;
    }
    if (!self_compaction_uses_vertex_list(solver) && self_sleep_source_vertex_skipped(solver, i)) {
        diag_note_self_source_skipped(solver);
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
    int scan_limit = max_neighbors * (self_fast_mode(solver) ? 8 : 32);
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
                        if (!self_should_project_contact(gap, delta, previous_delta, normal)) {
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

__global__ void damp_self_recovery_velocity_kernel(Solver solver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count || !solver.self_recovery_touched || solver.self_recovery_touched[i] == 0) {
        return;
    }
    if (solver.inv_mass[i] <= 0.0f) {
        return;
    }
    Vec3 recovery_delta = solver.self_recovery_delta
        ? solver.self_recovery_delta[i]
        : sub(solver.pos[i], solver.prev[i]);
    Vec3 adjusted_prev = add(solver.prev[i], mul(recovery_delta, kSelfRecoveryVelocityDamping));
    float recovery_len = norm(recovery_delta);
    if (recovery_len > kEps) {
        Vec3 dir = mul(recovery_delta, 1.0f / recovery_len);
        Vec3 step = sub(solver.pos[i], adjusted_prev);
        float returning = dot(step, dir);
        if (returning < 0.0f) {
            step = sub(step, mul(dir, returning));
            adjusted_prev = sub(solver.pos[i], step);
        }
    }
    solver.prev[i] = adjusted_prev;
}

__device__ float one_ring_stretch_residual(Solver solver, int vertex, Vec3 candidate) {
    if (!solver.vertex_neighbor_offsets || !solver.vertex_neighbors || !solver.rest) {
        return -1.0f;
    }
    int start = solver.vertex_neighbor_offsets[vertex];
    int end = solver.vertex_neighbor_offsets[vertex + 1];
    if (end <= start) {
        return -1.0f;
    }
    float residual = 0.0f;
    for (int idx = start; idx < end; ++idx) {
        int neighbor = solver.vertex_neighbors[idx];
        if (neighbor < 0 || neighbor >= solver.cfg.vertex_count) {
            continue;
        }
        float rest_len = norm(sub(solver.rest[vertex], solver.rest[neighbor]));
        float len = norm(sub(candidate, solver.pos[neighbor]));
        residual += fabsf(len - rest_len);
    }
    return residual;
}

__device__ float triangle_area_value(Vec3 a, Vec3 b, Vec3 c) {
    Vec3 n = cross(sub(b, a), sub(c, a));
    return 0.5f * sqrtf(fmaxf(dot(n, n), 0.0f));
}

__device__ float one_ring_area_residual(Solver solver, int vertex, Vec3 candidate) {
    if (!solver.volume_vertex_offsets || !solver.volume_vertex_triangles || !solver.triangles || !solver.rest) {
        return -1.0f;
    }
    int start = solver.volume_vertex_offsets[vertex];
    int end = solver.volume_vertex_offsets[vertex + 1];
    if (end <= start) {
        return -1.0f;
    }
    float residual = 0.0f;
    for (int cursor = start; cursor < end; ++cursor) {
        int t = solver.volume_vertex_triangles[cursor];
        if (t < 0 || t >= solver.cfg.triangle_count) {
            continue;
        }
        Int3 tri = solver.triangles[t];
        if (tri.x < 0 || tri.x >= solver.cfg.vertex_count
            || tri.y < 0 || tri.y >= solver.cfg.vertex_count
            || tri.z < 0 || tri.z >= solver.cfg.vertex_count) {
            continue;
        }
        Vec3 a = (tri.x == vertex) ? candidate : solver.pos[tri.x];
        Vec3 b = (tri.y == vertex) ? candidate : solver.pos[tri.y];
        Vec3 c = (tri.z == vertex) ? candidate : solver.pos[tri.z];
        Vec3 ra = solver.rest[tri.x];
        Vec3 rb = solver.rest[tri.y];
        Vec3 rc = solver.rest[tri.z];
        float rest_area = triangle_area_value(ra, rb, rc);
        float area = triangle_area_value(a, b, c);
        residual += fabsf(area - rest_area) / fmaxf(rest_area, 1.0e-6f);
    }
    return residual;
}

__device__ float one_ring_area_peak_ratio(Solver solver, int vertex, Vec3 candidate) {
    if (!solver.volume_vertex_offsets || !solver.volume_vertex_triangles || !solver.triangles || !solver.rest) {
        return -1.0f;
    }
    int start = solver.volume_vertex_offsets[vertex];
    int end = solver.volume_vertex_offsets[vertex + 1];
    if (end <= start) {
        return -1.0f;
    }
    float peak = 0.0f;
    for (int cursor = start; cursor < end; ++cursor) {
        int t = solver.volume_vertex_triangles[cursor];
        if (t < 0 || t >= solver.cfg.triangle_count) {
            continue;
        }
        Int3 tri = solver.triangles[t];
        if (tri.x < 0 || tri.x >= solver.cfg.vertex_count
            || tri.y < 0 || tri.y >= solver.cfg.vertex_count
            || tri.z < 0 || tri.z >= solver.cfg.vertex_count) {
            continue;
        }
        Vec3 a = (tri.x == vertex) ? candidate : solver.pos[tri.x];
        Vec3 b = (tri.y == vertex) ? candidate : solver.pos[tri.y];
        Vec3 c = (tri.z == vertex) ? candidate : solver.pos[tri.z];
        float rest_area = triangle_area_value(solver.rest[tri.x], solver.rest[tri.y], solver.rest[tri.z]);
        if (rest_area <= 1.0e-10f) {
            continue;
        }
        float area = triangle_area_value(a, b, c);
        peak = fmaxf(peak, area / rest_area);
    }
    return peak;
}

__global__ void jitter_stabilizer_kernel(Solver solver, float threshold, float correction_limit) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count
        || !solver.jitter_frame_start_pos
        || !solver.jitter_prev_delta
        || !solver.jitter_score
        || solver.inv_mass[i] <= 0.0f) {
        return;
    }
    Vec3 p = solver.pos[i];
    Vec3 start = solver.jitter_frame_start_pos[i];
    if (!finite_vec(p) || !finite_vec(start)) {
        solver.jitter_prev_delta[i] = {0.0f, 0.0f, 0.0f};
        solver.jitter_score[i] = 0;
        return;
    }
    Vec3 delta = sub(p, start);
    Vec3 previous_delta = solver.jitter_prev_delta[i];
    Vec3 high_freq = sub(delta, previous_delta);
    bool oscillating = dot(delta, previous_delta) < 0.0f && norm(high_freq) > threshold;
    int score = solver.jitter_score[i];
    score = oscillating ? min(score + 1, 4) : max(score - 1, 0);
    solver.jitter_score[i] = score;
    solver.jitter_prev_delta[i] = delta;
    if (score < kJitterScoreThreshold) {
        return;
    }

    Vec3 correction = mul(high_freq, -kJitterCorrectionScale);
    float correction_len = norm(correction);
    if (correction_len <= kEps) {
        return;
    }
    if (correction_len > correction_limit) {
        correction = mul(correction, correction_limit / correction_len);
        correction_len = correction_limit;
    }

    if (solver.self_recovery_touched
        && solver.self_recovery_delta
        && solver.self_recovery_touched[i] != 0) {
        Vec3 recovery_delta = solver.self_recovery_delta[i];
        float recovery_len = norm(recovery_delta);
        if (recovery_len > kEps
            && dot(correction, recovery_delta) < kJitterRecoveryOpposeLimit * correction_len * recovery_len) {
            if (solver.jitter_counts) {
                atomicAdd(&solver.jitter_counts[kJitterRejectedCount], 1ull);
            }
            return;
        }
    }

    Vec3 candidate = add(p, correction);
    float current_residual = one_ring_stretch_residual(solver, i, p);
    float candidate_residual = one_ring_stretch_residual(solver, i, candidate);
    if (current_residual < 0.0f
        || candidate_residual < 0.0f
        || candidate_residual > current_residual * kJitterStretchResidualTolerance + 1.0e-6f) {
        if (solver.jitter_counts) {
            atomicAdd(&solver.jitter_counts[kJitterRejectedCount], 1ull);
        }
        return;
    }

    float current_area_residual = one_ring_area_residual(solver, i, p);
    float candidate_area_residual = one_ring_area_residual(solver, i, candidate);
    if (current_area_residual >= 0.0f
        && candidate_area_residual >= 0.0f
        && candidate_area_residual > current_area_residual * kJitterAreaResidualTolerance + 1.0e-6f) {
        if (solver.jitter_counts) {
            atomicAdd(&solver.jitter_counts[kJitterRejectedCount], 1ull);
        }
        return;
    }
    float current_area_peak = one_ring_area_peak_ratio(solver, i, p);
    float candidate_area_peak = one_ring_area_peak_ratio(solver, i, candidate);
    if (current_area_peak >= 0.0f
        && candidate_area_peak >= 0.0f
        && candidate_area_peak > current_area_peak * kJitterAreaPeakTolerance + 1.0e-6f) {
        if (solver.jitter_counts) {
            atomicAdd(&solver.jitter_counts[kJitterRejectedCount], 1ull);
        }
        return;
    }

    solver.pos[i] = candidate;
    solver.jitter_prev_delta[i] = sub(candidate, start);
    if (solver.self_sample_hash_dirty) {
        atomicExch(solver.self_sample_hash_dirty, 1);
    }
    if (solver.jitter_counts) {
        atomicAdd(&solver.jitter_counts[kJitterStabilizedCount], 1ull);
    }
    if (solver.jitter_max_correction) {
        atomic_max_float(solver.jitter_max_correction, correction_len);
    }

    Vec3 step = sub(candidate, solver.prev[i]);
    Vec3 dir = mul(correction, 1.0f / fmaxf(correction_len, kEps));
    float returning = dot(step, dir);
    if (returning < 0.0f) {
        step = sub(step, mul(dir, returning));
        solver.prev[i] = sub(candidate, step);
    }
}

__global__ void clamp_self_recovery_displacement_kernel(Solver solver, float max_delta, int preserve_velocity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count
        || !solver.self_recovery_touched
        || !solver.self_recovery_delta
        || solver.self_recovery_touched[i] == 0
        || solver.inv_mass[i] <= 0.0f) {
        return;
    }
    Vec3 delta = solver.self_recovery_delta[i];
    if (!finite_vec(delta)) {
        diag_note_nonfinite(solver);
        solver.self_recovery_delta[i] = {0.0f, 0.0f, 0.0f};
        return;
    }
    float len = norm(delta);
    if (len <= max_delta || len <= kEps) {
        return;
    }
    Vec3 allowed = mul(delta, max_delta / len);
    Vec3 excess = sub(delta, allowed);
    solver.pos[i] = sub(solver.pos[i], excess);
    if (preserve_velocity) {
        solver.prev[i] = sub(solver.prev[i], excess);
    }
    solver.self_recovery_delta[i] = allowed;
    if (solver.self_sample_hash_dirty) {
        atomicExch(solver.self_sample_hash_dirty, 1);
    }
}

__global__ void clear_self_sleep_frame_flags_kernel(Solver solver) {
    int region = blockIdx.x * blockDim.x + threadIdx.x;
    if (region >= solver.self_sleep_region_count) {
        return;
    }
    if (solver.self_sleep_region_motion) {
        solver.self_sleep_region_motion[region] = 0;
    }
    if (solver.self_sleep_region_touched) {
        solver.self_sleep_region_touched[region] = 0;
    }
}

__global__ void wake_self_sleep_regions_kernel(Solver solver) {
    int region = blockIdx.x * blockDim.x + threadIdx.x;
    if (region >= solver.self_sleep_region_count) {
        return;
    }
    if (solver.self_sleep_region_still_frames) {
        solver.self_sleep_region_still_frames[region] = 0;
    }
    if (solver.self_sleep_region_sleeping) {
        solver.self_sleep_region_sleeping[region] = 0;
    }
    if (region == 0 && solver.self_sleep_has_sleeping) {
        solver.self_sleep_has_sleeping[0] = 0;
    }
}

__global__ void clear_self_sleep_summary_kernel(Solver solver) {
    if (solver.self_sleep_has_sleeping) {
        solver.self_sleep_has_sleeping[0] = 0;
    }
}

__global__ void update_self_sleep_motion_kernel(Solver solver, float motion_threshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= solver.cfg.vertex_count || !solver.self_sleep_prev_pos || !solver.self_sleep_region_motion) {
        return;
    }
    int region = self_sleep_region_for_vertex(solver, i);
    if (region < 0) {
        return;
    }
    Vec3 current = solver.pos[i];
    Vec3 previous = solver.self_sleep_prev_pos[i];
    solver.self_sleep_prev_pos[i] = current;
    if (!finite_vec(current) || norm(sub(current, previous)) > motion_threshold) {
        atomicExch(&solver.self_sleep_region_motion[region], 1);
    }
}

__global__ void finalize_self_sleep_regions_kernel(Solver solver) {
    int region = blockIdx.x * blockDim.x + threadIdx.x;
    if (region >= solver.self_sleep_region_count
        || !solver.self_sleep_region_still_frames
        || !solver.self_sleep_region_sleeping) {
        return;
    }
    if (solver.self_sleep_region_vertex_counts && solver.self_sleep_region_vertex_counts[region] <= 0) {
        return;
    }
    bool moved = solver.self_sleep_region_motion && solver.self_sleep_region_motion[region] != 0;
    bool touched = solver.self_sleep_region_touched && solver.self_sleep_region_touched[region] != 0;
    if (moved || touched) {
        solver.self_sleep_region_still_frames[region] = 0;
        solver.self_sleep_region_sleeping[region] = 0;
    } else {
        int still_frames = solver.self_sleep_region_still_frames[region] + 1;
        solver.self_sleep_region_still_frames[region] = still_frames;
        int required_frames = solver.cfg.self_sleep_still_frames > 1 ? solver.cfg.self_sleep_still_frames : 1;
        if (still_frames >= required_frames) {
            solver.self_sleep_region_sleeping[region] = 1;
        }
    }
    if (solver.diag_counts) {
        if (solver.self_sleep_region_sleeping[region] != 0) {
            atomicAdd(&solver.diag_counts[kDiagSelfSleepingRegions], 1ull);
            if (solver.self_sleep_has_sleeping) {
                atomicExch(solver.self_sleep_has_sleeping, 1);
            }
        } else {
            atomicAdd(&solver.diag_counts[kDiagSelfActiveRegions], 1ull);
        }
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

bool allocate_dynamic_particle_collision(Solver* solver, int particle_count) {
    if (particle_count <= 0) {
        solver->dynamic_particle_count = 0;
        solver->dynamic_particle_max_radius = 0.0f;
        return true;
    }
    if (particle_count <= solver->dynamic_particle_capacity
        && solver->dynamic_particles
        && solver->dynamic_particle_heads
        && solver->dynamic_particle_next
        && solver->dynamic_particle_index
        && solver->dynamic_particle_count_buffer) {
        solver->dynamic_particle_count = particle_count;
        return true;
    }

    cudaFree(solver->dynamic_particles);
    cudaFree(solver->dynamic_particle_heads);
    cudaFree(solver->dynamic_particle_next);
    cudaFree(solver->dynamic_particle_index);
    cudaFree(solver->dynamic_particle_count_buffer);
    solver->dynamic_particles = nullptr;
    solver->dynamic_particle_heads = nullptr;
    solver->dynamic_particle_next = nullptr;
    solver->dynamic_particle_index = nullptr;
    solver->dynamic_particle_count_buffer = nullptr;

    solver->dynamic_particle_capacity = particle_count;
    solver->dynamic_particle_count = particle_count;
    solver->dynamic_particle_hash_table_size = next_power_of_two(std::max(1024, particle_count * 2));

    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&solver->dynamic_particles), sizeof(DynamicParticle) * particle_count);
    if (!set_cuda_error(err, "dynamic particle allocation")) {
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&solver->dynamic_particle_heads), sizeof(int) * solver->dynamic_particle_hash_table_size);
    if (!set_cuda_error(err, "dynamic particle hash allocation")) {
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&solver->dynamic_particle_next), sizeof(int) * particle_count);
    if (!set_cuda_error(err, "dynamic particle link allocation")) {
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&solver->dynamic_particle_index), sizeof(int) * particle_count);
    if (!set_cuda_error(err, "dynamic particle entry allocation")) {
        return false;
    }
    err = cudaMalloc(reinterpret_cast<void**>(&solver->dynamic_particle_count_buffer), sizeof(int));
    return set_cuda_error(err, "dynamic particle counter allocation");
}

bool rebuild_dynamic_particle_hash(Solver* solver) {
    if (!solver || solver->dynamic_particle_count <= 0) {
        return true;
    }
    if (!solver->dynamic_particle_heads
        || !solver->dynamic_particle_next
        || !solver->dynamic_particle_index
        || !solver->dynamic_particle_count_buffer) {
        return set_error("dynamic particle hash is not allocated");
    }
    const auto hash_start = std::chrono::high_resolution_clock::now();
    cudaMemset(solver->dynamic_particle_heads, 0xff, sizeof(int) * solver->dynamic_particle_hash_table_size);
    cudaMemset(solver->dynamic_particle_count_buffer, 0, sizeof(int));
    build_dynamic_particle_hash_kernel<<<block_count(solver->dynamic_particle_count), 256>>>(*solver);
    if (!set_cuda_error(cudaDeviceSynchronize(), "rebuild dynamic particle hash")) {
        return false;
    }
    solver->pending_hash_build_ms += elapsed_ms_since(hash_start);
    return true;
}

bool clear_external_contact_cache(Solver* solver, const char* label) {
    if (!solver || !solver->external_contacts || solver->external_contact_capacity <= 0) {
        return true;
    }
    return set_cuda_error(
        cudaMemset(solver->external_contacts, 0, sizeof(ExternalContact) * solver->external_contact_capacity),
        label
    );
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
    unsigned long long counts[kDiagCountSlots] = {};
    counts[kDiagFiniteFlag] = 1ull;
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
    unsigned long long counts[kDiagCountSlots] = {};
    counts[kDiagFiniteFlag] = 1ull;
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
    host_diag->self_skipped_sources = static_cast<long long>(counts[kDiagSelfSkippedSources]);
    host_diag->self_active_regions = static_cast<long long>(counts[kDiagSelfActiveRegions]);
    host_diag->self_sleeping_regions = static_cast<long long>(counts[kDiagSelfSleepingRegions]);
    host_diag->external_contact_cache_hits = static_cast<long long>(counts[kDiagExternalContactCacheHits]);
    host_diag->external_contact_cache_misses = static_cast<long long>(counts[kDiagExternalContactCacheMisses]);
    host_diag->external_contact_cache_overflow = static_cast<long long>(counts[kDiagExternalContactCacheOverflow]);
    host_diag->external_friction_corrections = static_cast<long long>(counts[kDiagExternalFrictionCorrections]);
    host_diag->external_contact_cache_count = static_cast<long long>(counts[kDiagExternalContactCacheCount]);
    host_diag->dynamic_particle_candidate_count = static_cast<long long>(counts[kDiagDynamicParticleCandidates]);
    host_diag->dynamic_particle_contacts = static_cast<long long>(counts[kDiagDynamicParticleContacts]);
    host_diag->dynamic_particle_overflow = static_cast<long long>(counts[kDiagDynamicParticleOverflow]);
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
    if (!fetch_diagnostics_buffers(
        solver->diag_counts,
        solver->diag_min_gap,
        &solver->diag,
        "download diagnostic counts",
        "download diagnostic min gap"
    )) {
        return false;
    }
    if (solver->jitter_counts) {
        unsigned long long counts[kJitterCountSlots] = {};
        if (!set_cuda_error(
                cudaMemcpy(counts, solver->jitter_counts, sizeof(counts), cudaMemcpyDeviceToHost),
                "download jitter stabilizer counts")) {
            return false;
        }
        solver->diag.jitter_stabilized_vertices = static_cast<long long>(counts[kJitterStabilizedCount]);
        solver->diag.jitter_rejected_vertices = static_cast<long long>(counts[kJitterRejectedCount]);
    }
    if (solver->jitter_max_correction) {
        float max_correction = 0.0f;
        if (!set_cuda_error(
                cudaMemcpy(&max_correction, solver->jitter_max_correction, sizeof(float), cudaMemcpyDeviceToHost),
                "download jitter stabilizer max correction")) {
            return false;
        }
        solver->diag.jitter_max_correction = std::isfinite(max_correction) ? max_correction : 0.0f;
    }
    solver->diag.force_field_count = static_cast<long long>(solver->force_field_count);
    solver->diag.unsupported_force_field_count = static_cast<long long>(solver->unsupported_force_field_count);
    solver->diag.dynamic_particle_count = static_cast<long long>(solver->dynamic_particle_count);
    solver->diag.dynamic_triangle_count = static_cast<long long>(solver->dynamic_triangle_count);
    solver->diag.static_triangle_count = static_cast<long long>(solver->cfg.static_triangle_count);
    return true;
}

bool reset_jitter_diagnostics(Solver* solver) {
    if (!solver || !solver->cfg.jitter_stabilizer_enabled) {
        return true;
    }
    if (solver->jitter_counts
        && !set_cuda_error(
            cudaMemset(solver->jitter_counts, 0, sizeof(unsigned long long) * kJitterCountSlots),
            "reset jitter stabilizer counts")) {
        return false;
    }
    if (solver->jitter_max_correction
        && !set_cuda_error(
            cudaMemset(solver->jitter_max_correction, 0, sizeof(float)),
            "reset jitter stabilizer max correction")) {
        return false;
    }
    return true;
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

void reset_self_compaction_state(Solver* solver) {
    if (!solver) {
        return;
    }
    solver->self_active_vertex_count = 0;
    solver->self_suspect_vertex_count = 0;
    solver->self_active_sample_count = 0;
    solver->self_suspect_sample_count = 0;
    solver->self_suspect_region_count = 0;
    solver->self_compaction_used = 0;
    solver->self_compaction_samples_per_triangle = 0;
    solver->self_source_mode = kSelfSourceFull;
    solver->self_full_recovery_fallbacks = 0;
    solver->self_vs_pair_count = 0;
    solver->self_vs_pair_overflow = 0;
    solver->self_vs_pair_current_overflow = 0;
    solver->self_vs_pair_compaction_used = 0;
    solver->self_vs_pair_valid = 0;
    solver->self_sample_hash_valid = 0;
    if (solver->self_compaction_counts) {
        cudaMemset(solver->self_compaction_counts, 0, sizeof(int) * kSelfCompactionCountSlots);
    }
    if (solver->self_vs_pair_counts) {
        cudaMemset(solver->self_vs_pair_counts, 0, sizeof(int) * kSelfVsPairCountSlots);
    }
}

bool self_compaction_ready(const Solver* solver) {
    if (!solver) {
        return false;
    }
    if (!solver->cfg.self_compaction_enabled
        || !solver->cfg.self_collision
        || solver->self_sleep_force_active
        || !solver->self_compaction_counts
        || !solver->self_active_vertices
        || !solver->self_suspect_vertices
        || !solver->self_active_vertex_flags
        || !solver->self_suspect_vertex_flags) {
        return false;
    }
    const bool sleep_ready = solver->cfg.self_sleep_enabled
        && solver->self_sleep_region_count > 0
        && solver->self_sleep_frame_count >= std::max(solver->cfg.self_sleep_still_frames, 1);
    const bool fast_frontier_ready = solver->cfg.self_collision_mode == kSelfCollisionModeFast
        && solver->self_recovery_touched != nullptr;
    return sleep_ready || fast_frontier_ready;
}

bool prepare_self_compaction_lists(Solver* solver, int v_blocks) {
    if (!solver || !self_compaction_ready(solver)) {
        return true;
    }
    if (!set_cuda_error(
            cudaMemset(solver->self_compaction_counts, 0, sizeof(int) * kSelfCompactionCountSlots),
            "reset self compaction counts")) {
        return false;
    }
    if (!set_cuda_error(
            cudaMemset(solver->self_active_vertex_flags, 0, sizeof(int) * solver->cfg.vertex_count),
            "reset self active vertex flags")) {
        return false;
    }
    if (!set_cuda_error(
            cudaMemset(solver->self_suspect_vertex_flags, 0, sizeof(int) * solver->cfg.vertex_count),
            "reset self suspect vertex flags")) {
        return false;
    }
    Solver compact_solver = *solver;
    compact_solver.self_source_mode = kSelfSourceFull;
    compact_solver.self_samples_per_triangle = solver->self_samples_per_triangle;
    compact_solver.self_sample_count = solver->self_sample_count;
    compact_solver.self_compaction_samples_per_triangle = compact_solver.self_samples_per_triangle;
    if (compact_solver.self_sleep_region_count > 0) {
        build_self_compaction_regions_kernel<<<block_count(compact_solver.self_sleep_region_count), 256>>>(compact_solver);
    }
    build_self_compaction_vertices_kernel<<<v_blocks, 256>>>(compact_solver);
    if (!set_cuda_error(cudaGetLastError(), "launch self compaction")) {
        return false;
    }
    int counts[kSelfCompactionCountSlots] = {};
    if (!set_cuda_error(
            cudaMemcpy(counts, solver->self_compaction_counts, sizeof(counts), cudaMemcpyDeviceToHost),
            "download self compaction counts")) {
        return false;
    }
    if (counts[kSelfCompactionActiveVertices] > 0
        && compact_solver.self_sample_count > 0
        && compact_solver.self_active_samples
        && compact_solver.self_suspect_samples) {
        build_self_compaction_samples_kernel<<<block_count(compact_solver.self_sample_count), 256>>>(compact_solver);
        if (!set_cuda_error(cudaGetLastError(), "launch self sample compaction")) {
            return false;
        }
        if (!set_cuda_error(
                cudaMemcpy(counts, solver->self_compaction_counts, sizeof(counts), cudaMemcpyDeviceToHost),
                "download self sample compaction counts")) {
            return false;
        }
    }
    solver->self_active_vertex_count = counts[kSelfCompactionActiveVertices];
    solver->self_suspect_vertex_count = counts[kSelfCompactionSuspectVertices];
    solver->self_active_sample_count = counts[kSelfCompactionActiveSamples];
    solver->self_suspect_sample_count = counts[kSelfCompactionSuspectSamples];
    solver->self_suspect_region_count = counts[kSelfCompactionSuspectRegions];
    solver->self_compaction_samples_per_triangle = compact_solver.self_samples_per_triangle;
    float threshold = solver->cfg.self_compaction_active_fraction_threshold;
    if (!std::isfinite(threshold) || threshold <= 0.0f) {
        threshold = 0.75f;
    }
    threshold = std::clamp(threshold, 0.01f, 1.0f);
    bool active_list_used = solver->cfg.vertex_count > 0
        && solver->self_active_vertex_count > 0
        && solver->self_active_vertex_count < solver->cfg.vertex_count
        && static_cast<float>(solver->self_active_vertex_count) < static_cast<float>(solver->cfg.vertex_count) * threshold;
    solver->self_compaction_used = active_list_used ? 1 : 0;
    return true;
}

bool fetch_self_compaction_diagnostics(Solver* solver) {
    if (!solver) {
        return set_error("invalid solver");
    }
    int counts[kSelfCompactionCountSlots] = {};
    if (solver->self_compaction_counts) {
        if (!set_cuda_error(
                cudaMemcpy(counts, solver->self_compaction_counts, sizeof(counts), cudaMemcpyDeviceToHost),
                "download self compaction counts")) {
            return false;
        }
    }
    solver->self_active_vertex_count = counts[kSelfCompactionActiveVertices];
    solver->self_suspect_vertex_count = counts[kSelfCompactionSuspectVertices];
    solver->self_active_sample_count = counts[kSelfCompactionActiveSamples];
    solver->self_suspect_sample_count = counts[kSelfCompactionSuspectSamples];
    solver->self_suspect_region_count = counts[kSelfCompactionSuspectRegions];
    solver->diag.self_active_vertices = static_cast<long long>(solver->self_active_vertex_count);
    solver->diag.self_active_samples = static_cast<long long>(solver->self_active_sample_count);
    solver->diag.self_suspect_regions = static_cast<long long>(solver->self_suspect_region_count);
    solver->diag.self_full_recovery_fallbacks = solver->self_full_recovery_fallbacks;
    solver->diag.self_vs_pair_count = static_cast<long long>(solver->self_vs_pair_count);
    solver->diag.self_vs_pair_capacity = static_cast<long long>(solver->self_vs_pair_capacity);
    solver->diag.self_vs_pair_overflow = static_cast<long long>(solver->self_vs_pair_overflow);
    solver->diag.self_vs_pair_compaction_used = static_cast<long long>(solver->self_vs_pair_compaction_used);
    float threshold = solver->cfg.self_compaction_active_fraction_threshold;
    if (!std::isfinite(threshold) || threshold <= 0.0f) {
        threshold = 0.75f;
    }
    threshold = std::clamp(threshold, 0.01f, 1.0f);
    bool active_list_used = solver->self_compaction_used
        && solver->cfg.vertex_count > 0
        && solver->self_active_vertex_count > 0
        && solver->self_active_vertex_count < solver->cfg.vertex_count
        && static_cast<float>(solver->self_active_vertex_count) < static_cast<float>(solver->cfg.vertex_count) * threshold;
    bool suspect_list_used = solver->self_compaction_used
        && solver->cfg.vertex_count > 0
        && solver->self_suspect_vertex_count > 0
        && solver->self_suspect_vertex_count < solver->cfg.vertex_count
        && static_cast<float>(solver->self_suspect_vertex_count) < static_cast<float>(solver->cfg.vertex_count) * threshold;
    solver->diag.self_compaction_used = (active_list_used || suspect_list_used) ? 1ll : 0ll;
    return true;
}

float host_self_compaction_threshold(const Solver* solver) {
    if (!solver) {
        return 0.75f;
    }
    float threshold = solver->cfg.self_compaction_active_fraction_threshold;
    if (!std::isfinite(threshold) || threshold <= 0.0f) {
        threshold = 0.75f;
    }
    return std::clamp(threshold, 0.01f, 1.0f);
}

bool host_self_uses_vertex_list(const Solver* solver, int source_mode) {
    if (!solver || !solver->cfg.self_compaction_enabled || !solver->self_compaction_used || solver->cfg.vertex_count <= 0) {
        return false;
    }
    int count = 0;
    if (source_mode == kSelfSourceActive && solver->self_active_vertices) {
        count = solver->self_active_vertex_count;
    } else if (source_mode == kSelfSourceSuspect && solver->self_suspect_vertices) {
        count = solver->self_suspect_vertex_count;
    } else {
        return false;
    }
    return count > 0
        && count < solver->cfg.vertex_count
        && static_cast<float>(count) < static_cast<float>(solver->cfg.vertex_count) * host_self_compaction_threshold(solver);
}

bool host_self_uses_sample_list(const Solver* solver, int source_mode, int samples_per_triangle, int sample_count) {
    if (!solver
        || !solver->cfg.self_compaction_enabled
        || !solver->self_compaction_used
        || sample_count <= 0
        || solver->self_compaction_samples_per_triangle != samples_per_triangle) {
        return false;
    }
    int count = 0;
    if (source_mode == kSelfSourceActive && solver->self_active_samples) {
        count = solver->self_active_sample_count;
    } else if (source_mode == kSelfSourceSuspect && solver->self_suspect_samples) {
        count = solver->self_suspect_sample_count;
    } else {
        return false;
    }
    return count > 0
        && count < sample_count
        && static_cast<float>(count) < static_cast<float>(sample_count) * host_self_compaction_threshold(solver);
}

int host_self_vertex_source_count(const Solver* solver, int source_mode) {
    if (host_self_uses_vertex_list(solver, source_mode)) {
        return source_mode == kSelfSourceActive ? solver->self_active_vertex_count : solver->self_suspect_vertex_count;
    }
    return solver ? solver->cfg.vertex_count : 0;
}

int host_self_sample_source_count(const Solver* solver, int source_mode, int samples_per_triangle, int sample_count) {
    if (host_self_uses_sample_list(solver, source_mode, samples_per_triangle, sample_count)) {
        return source_mode == kSelfSourceActive ? solver->self_active_sample_count : solver->self_suspect_sample_count;
    }
    return sample_count;
}

bool host_self_pair_compaction_ready(const Solver* solver, int source_mode) {
    if (!solver
        || !solver->cfg.self_pair_compaction_enabled
        || !solver->cfg.self_collision
        || !solver->self_vs_pairs
        || !solver->self_vs_pair_counts
        || solver->self_vs_pair_capacity <= 0
        || !solver->self_sample_heads
        || !solver->self_sample_next
        || solver->self_sample_count <= 0) {
        return false;
    }
    if (!host_self_uses_vertex_list(solver, source_mode)) {
        return false;
    }
    int source_count = host_self_vertex_source_count(solver, source_mode);
    if (source_count <= 0 || solver->cfg.vertex_count <= 0) {
        return false;
    }
    float threshold = std::min(host_self_compaction_threshold(solver), 0.40f);
    return static_cast<float>(source_count) < static_cast<float>(solver->cfg.vertex_count) * threshold;
}

bool fetch_self_vs_pair_counts(Solver* solver, const char* label) {
    if (!solver || !solver->self_vs_pair_counts) {
        return set_error("self vertex-surface pair counters are not allocated");
    }
    int counts[kSelfVsPairCountSlots] = {};
    if (!set_cuda_error(
            cudaMemcpy(counts, solver->self_vs_pair_counts, sizeof(counts), cudaMemcpyDeviceToHost),
            label)) {
        return false;
    }
    int raw_count = std::max(counts[kSelfVsPairCount], 0);
    bool current_overflow = counts[kSelfVsPairOverflow] != 0 || raw_count > solver->self_vs_pair_capacity;
    solver->self_vs_pair_count = std::min(raw_count, solver->self_vs_pair_capacity);
    solver->self_vs_pair_current_overflow = current_overflow ? 1 : 0;
    solver->self_vs_pair_overflow = solver->self_vs_pair_overflow || current_overflow;
    solver->self_vs_pair_valid = (raw_count > 0 && !current_overflow) ? 1 : 0;
    return true;
}

bool reset_self_vs_pair_counts(Solver* solver) {
    if (!solver || !solver->self_vs_pair_counts) {
        return set_error("self vertex-surface pair counters are not allocated");
    }
    solver->self_vs_pair_count = 0;
    solver->self_vs_pair_current_overflow = 0;
    solver->self_vs_pair_valid = 0;
    return set_cuda_error(
        cudaMemset(solver->self_vs_pair_counts, 0, sizeof(int) * kSelfVsPairCountSlots),
        "reset self vertex-surface pair counters"
    );
}

bool build_self_triangle_hash(
    Solver* solver,
    const Solver& triangle_solver,
    int triangle_source_blocks,
    std::vector<TimedSegment>* timings,
    const char* label
) {
    if (!solver
        || !solver->self_tri_heads
        || !solver->self_tri_entry_next
        || !solver->self_tri_entry_index
        || !solver->self_tri_entry_count
        || solver->self_tri_hash_table_size <= 0
        || solver->self_tri_entry_capacity <= 0) {
        return set_error("self triangle hash is not allocated");
    }
    TimedSegment segment;
    if (!begin_timed_segment(timings, kTimingSelfHash, &segment, label)) {
        return false;
    }
    cudaMemset(solver->self_tri_heads, 0xff, sizeof(int) * solver->self_tri_hash_table_size);
    cudaMemset(solver->self_tri_entry_count, 0, sizeof(int));
    build_self_triangle_hash_kernel<<<triangle_source_blocks, 256>>>(triangle_solver);
    if (!end_timed_segment(timings, &segment, label)) {
        return false;
    }
    return set_cuda_error(cudaGetLastError(), label);
}

bool build_self_vs_pairs(
    Solver* solver,
    const Solver& pair_solver,
    int vertex_source_blocks,
    int store_all_coarse,
    std::vector<TimedSegment>* timings,
    const char* label
) {
    if (!solver) {
        return set_error("invalid self vertex-surface pair build");
    }
    if (!reset_self_vs_pair_counts(solver)) {
        return false;
    }
    TimedSegment segment;
    if (!begin_timed_segment(timings, kTimingSelfVsPairBuild, &segment, label)) {
        return false;
    }
    build_self_vertex_surface_pairs_kernel<<<vertex_source_blocks, 256>>>(pair_solver, store_all_coarse);
    if (!end_timed_segment(timings, &segment, label)) {
        return false;
    }
    if (!set_cuda_error(cudaGetLastError(), label)) {
        return false;
    }
    return fetch_self_vs_pair_counts(solver, "download self vertex-surface pair counts");
}

int current_self_vs_pair_count(const Solver* solver) {
    if (!solver || solver->self_vs_pair_capacity <= 0) {
        return 0;
    }
    return std::max(0, std::min(solver->self_vs_pair_count, solver->self_vs_pair_capacity));
}

bool project_self_vs_pairs(
    Solver* solver,
    Solver pair_solver,
    int timing_slot,
    std::vector<TimedSegment>* timings,
    const char* label
) {
    int pair_count = current_self_vs_pair_count(solver);
    if (pair_count <= 0) {
        return true;
    }
    TimedSegment segment;
    if (!begin_timed_segment(timings, timing_slot, &segment, label)) {
        return false;
    }
    self_vertex_surface_pair_project_kernel<<<block_count(pair_count), 256>>>(pair_solver);
    if (!end_timed_segment(timings, &segment, label)) {
        return false;
    }
    solver->self_vs_pair_compaction_used = 1;
    solver->self_vs_pair_valid = 0;
    return set_cuda_error(cudaGetLastError(), label);
}

bool probe_self_vs_pairs(
    Solver* solver,
    Solver pair_solver,
    std::vector<TimedSegment>* timings,
    const char* label
) {
    int pair_count = current_self_vs_pair_count(solver);
    if (pair_count <= 0) {
        return false;
    }
    TimedSegment segment;
    if (!begin_timed_segment(timings, kTimingSelfVsPairProjectProbe, &segment, label)) {
        return false;
    }
    probe_self_vertex_surface_pairs_kernel<<<block_count(pair_count), 256>>>(pair_solver);
    if (!end_timed_segment(timings, &segment, label)) {
        return false;
    }
    solver->self_vs_pair_compaction_used = 1;
    return set_cuda_error(cudaGetLastError(), label);
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

bool clear_self_recovery_tracking(Solver* solver, const char* label) {
    if (!solver) {
        return set_error("invalid self recovery tracking reset");
    }
    if (solver->self_recovery_touched) {
        if (!set_cuda_error(
                cudaMemset(solver->self_recovery_touched, 0, sizeof(int) * solver->cfg.vertex_count),
                label)) {
            return false;
        }
    }
    if (solver->self_recovery_delta) {
        if (!set_cuda_error(
                cudaMemset(solver->self_recovery_delta, 0, sizeof(Vec3) * solver->cfg.vertex_count),
                label)) {
            return false;
        }
    }
    return true;
}

bool clamp_self_recovery_displacement(
    Solver* solver,
    int v_blocks,
    float cloth_thickness,
    float max_scale,
    bool preserve_velocity,
    std::vector<TimedSegment>* timings
) {
    if (!solver || !solver->self_recovery_touched || !solver->self_recovery_delta) {
        return true;
    }
    TimedSegment segment;
    if (!begin_timed_segment(timings, kTimingSelfRecovery, &segment, "start recovery displacement clamp timing")) {
        return false;
    }
    float max_delta = std::max(1.0e-4f, cloth_thickness * max_scale);
    clamp_self_recovery_displacement_kernel<<<v_blocks, 256>>>(*solver, max_delta, preserve_velocity ? 1 : 0);
    if (!end_timed_segment(timings, &segment, "end recovery displacement clamp timing")) {
        return false;
    }
    return set_cuda_error(cudaGetLastError(), "launch self recovery displacement clamp");
}

bool run_self_collision_pass(
    Solver* solver,
    int v_blocks,
    bool recovery_mode,
    bool run_surface_sample_pairs,
    bool force_full_fast_triangle_source,
    std::vector<TimedSegment>* timings
) {
    if (!solver || !solver->self_vert_heads || !solver->self_vert_next) {
        return true;
    }
    Solver collision_solver = *solver;
    collision_solver.self_recovery_mode = recovery_mode ? 1 : 0;
    collision_solver.self_cleanup_mode = force_full_fast_triangle_source ? 1 : 0;
    if (force_full_fast_triangle_source && collision_solver.cfg.self_collision_mode == kSelfCollisionModeFast) {
        collision_solver.self_source_mode = kSelfSourceFull;
    }
    if (!recovery_mode) {
        collision_solver.self_samples_per_triangle = self_fast_surface_sample_count_per_triangle(solver);
        collision_solver.self_sample_count = solver->cfg.triangle_count * collision_solver.self_samples_per_triangle;
    }
    const bool track_corrections = solver->self_recovery_touched && solver->self_recovery_delta;
    if (track_corrections && !clear_self_recovery_tracking(solver, "clear self collision correction tracking")) {
        return false;
    }
    const int vertex_source_count = host_self_vertex_source_count(&collision_solver, collision_solver.self_source_mode);
    const int vertex_source_blocks = block_count(std::max(vertex_source_count, 1));
    const int edge_blocks = block_count(std::max(solver->cfg.edge_count, 1));
    const int sample_source_count = host_self_sample_source_count(
        &collision_solver,
        collision_solver.self_source_mode,
        collision_solver.self_samples_per_triangle,
        collision_solver.self_sample_count
    );
    const int sample_source_blocks = block_count(std::max(sample_source_count, 1));
    TimedSegment hash_segment;
    if (!begin_timed_segment(timings, kTimingSelfHash, &hash_segment, "start self hash timing")) {
        return false;
    }
    cudaMemset(solver->self_vert_heads, 0xff, sizeof(int) * solver->self_vert_hash_table_size);
    build_self_vertex_hash_kernel<<<v_blocks, 256>>>(collision_solver);
    if (solver->self_edge_heads && solver->self_edge_next && solver->cfg.edge_count > 0) {
        cudaMemset(solver->self_edge_heads, 0xff, sizeof(int) * solver->self_edge_hash_table_size);
        build_self_edge_hash_kernel<<<edge_blocks, 256>>>(collision_solver);
    }
    if (!end_timed_segment(timings, &hash_segment, "end self hash timing")) {
        return false;
    }

    const int solve_slot = recovery_mode ? kTimingSelfRecovery : kTimingSelfSolve;
    TimedSegment solve_segment;
    if (!begin_timed_segment(timings, solve_slot, &solve_segment, "start self solve timing")) {
        return false;
    }
    self_particle_collision_kernel<<<vertex_source_blocks, 256>>>(collision_solver);
    if (solver->self_edge_heads
        && solver->self_edge_next
        && solver->cfg.edge_count > 0
        && collision_solver.cfg.self_collision_mode != kSelfCollisionModeFast) {
        self_edge_edge_collision_kernel<<<edge_blocks, 256>>>(collision_solver);
    }
    if (!end_timed_segment(timings, &solve_segment, "end self solve timing")) {
        return false;
    }
    const bool use_fast_triangle_path = false;
    if (use_fast_triangle_path) {
        const int triangle_source_count = solver->cfg.triangle_count;
        const int triangle_source_blocks = block_count(std::max(triangle_source_count, 1));
        Solver triangle_solver = collision_solver;
        triangle_solver.self_source_mode = kSelfSourceFull;
        if (!build_self_triangle_hash(
                solver,
                triangle_solver,
                triangle_source_blocks,
                timings,
                "build fast self triangle hash")) {
            return false;
        }
        if (!begin_timed_segment(timings, solve_slot, &solve_segment, "start fast self vertex-triangle timing")) {
            return false;
        }
        self_vertex_triangle_collision_kernel<<<vertex_source_blocks, 256>>>(collision_solver, 1);
        if (!end_timed_segment(timings, &solve_segment, "end fast self vertex-triangle timing")) {
            return false;
        }
    } else if (solver->self_sample_heads
        && solver->self_sample_next
        && collision_solver.self_sample_count > 0) {
        if (solver->self_sample_hash_dirty) {
            cudaMemset(solver->self_sample_hash_dirty, 0, sizeof(int));
        }
        if (!begin_timed_segment(timings, kTimingSelfHash, &hash_segment, "start self sample hash timing")) {
            return false;
        }
        cudaMemset(solver->self_sample_heads, 0xff, sizeof(int) * solver->self_sample_hash_table_size);
        build_self_surface_sample_hash_kernel<<<block_count(collision_solver.self_sample_count), 256>>>(collision_solver);
        solver->self_sample_hash_valid = 1;
        if (!end_timed_segment(timings, &hash_segment, "end self sample hash timing")) {
            return false;
        }
        bool used_pair_path = false;
        bool use_vs_pair_path = collision_solver.cfg.self_collision_mode != kSelfCollisionModeFast
            && solver->self_compaction_used
            && host_self_pair_compaction_ready(solver, collision_solver.self_source_mode);
        if (use_vs_pair_path) {
            Solver pair_solver = collision_solver;
            pair_solver.diag_counts = collision_solver.diag_counts;
            pair_solver.diag_min_gap = collision_solver.diag_min_gap;
            int store_all_coarse = recovery_mode ? 1 : 0;
            if (!build_self_vs_pairs(
                    solver,
                    pair_solver,
                    vertex_source_blocks,
                    store_all_coarse,
                    timings,
                    "build self vertex-surface pairs")) {
                return false;
            }
            if (!solver->self_vs_pair_current_overflow && current_self_vs_pair_count(solver) > 0) {
                int project_slot = recovery_mode ? kTimingSelfVsPairProjectRecovery : kTimingSelfVsPairProjectSolve;
                if (!project_self_vs_pairs(
                        solver,
                        pair_solver,
                        project_slot,
                        timings,
                        "project self vertex-surface pairs")) {
                    return false;
                }
                used_pair_path = true;
            } else if (!solver->self_vs_pair_current_overflow && current_self_vs_pair_count(solver) == 0 && !recovery_mode) {
                used_pair_path = true;
            }
        }
        if (!used_pair_path) {
            if (!begin_timed_segment(timings, solve_slot, &solve_segment, "start self vertex-surface timing")) {
                return false;
            }
            self_vertex_surface_collision_kernel<<<vertex_source_blocks, 256>>>(collision_solver);
            if (!end_timed_segment(timings, &solve_segment, "end self vertex-surface timing")) {
                return false;
            }
        }
        if (run_surface_sample_pairs) {
            if (!begin_timed_segment(timings, kTimingSelfHash, &hash_segment, "start self surface-pair hash timing")) {
                return false;
            }
            clear_self_surface_sample_hash_if_dirty_kernel<<<block_count(solver->self_sample_hash_table_size), 256>>>(collision_solver);
            build_self_surface_sample_hash_if_dirty_kernel<<<block_count(collision_solver.self_sample_count), 256>>>(collision_solver);
            solver->self_sample_hash_valid = 1;
            if (solver->self_sample_hash_dirty) {
                cudaMemset(solver->self_sample_hash_dirty, 0, sizeof(int));
            }
            if (!end_timed_segment(timings, &hash_segment, "end self surface-pair hash timing")) {
                return false;
            }
            if (!begin_timed_segment(timings, solve_slot, &solve_segment, "start self surface-pair timing")) {
                return false;
            }
            self_surface_sample_collision_kernel<<<sample_source_blocks, 256>>>(collision_solver);
            if (!end_timed_segment(timings, &solve_segment, "end self surface-pair timing")) {
                return false;
            }
        }
        if (recovery_mode && solver->cfg.edge_count > 0) {
            if (!begin_timed_segment(timings, solve_slot, &solve_segment, "start self edge-surface timing")) {
                return false;
            }
            self_edge_surface_collision_kernel<<<edge_blocks, 256>>>(collision_solver);
            if (!end_timed_segment(timings, &solve_segment, "end self edge-surface timing")) {
                return false;
            }
        }
    }
    if (track_corrections) {
        float cloth_thickness = std::max(solver->cfg.cloth_thickness, 1.0e-4f);
        float max_scale = recovery_mode ? kSelfRecoveryMaxDisplacementScale : kSelfCorrectionMaxDisplacementScale;
        if (!clamp_self_recovery_displacement(
                solver,
                v_blocks,
                cloth_thickness,
                max_scale,
                !recovery_mode,
                timings)) {
            return false;
        }
    }
    if (!begin_timed_segment(timings, solve_slot, &solve_segment, "start self sanitize timing")) {
        return false;
    }
    sanitize_positions_kernel<<<v_blocks, 256>>>(collision_solver);
    if (!end_timed_segment(timings, &solve_segment, "end self sanitize timing")) {
        return false;
    }
    return set_cuda_error(cudaGetLastError(), "launch self collision pass");
}

bool probe_self_collision(
    Solver* solver,
    int v_blocks,
    SsblXpbdDiagnostics* out_diag,
    std::vector<TimedSegment>* timings
) {
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
    const int vertex_source_count = host_self_vertex_source_count(&probe_solver, probe_solver.self_source_mode);
    const int vertex_source_blocks = block_count(std::max(vertex_source_count, 1));
    const int sample_source_count = host_self_sample_source_count(
        &probe_solver,
        probe_solver.self_source_mode,
        probe_solver.self_samples_per_triangle,
        probe_solver.self_sample_count
    );
    if (!reset_probe_diagnostics(solver)) {
        return false;
    }
    TimedSegment probe_segment;
    if (!begin_timed_segment(timings, kTimingSelfProbe, &probe_segment, "start self probe timing")) {
        return false;
    }
    cudaMemset(solver->self_vert_heads, 0xff, sizeof(int) * solver->self_vert_hash_table_size);
    build_self_vertex_hash_kernel<<<v_blocks, 256>>>(probe_solver);
    probe_self_particle_collision_kernel<<<vertex_source_blocks, 256>>>(probe_solver);
    const bool use_fast_triangle_path = false
        && probe_solver.cfg.self_collision_mode == kSelfCollisionModeFast
        && solver->self_tri_heads
        && solver->self_tri_entry_next
        && solver->self_tri_entry_index
        && solver->self_tri_entry_count
        && solver->cfg.triangle_count > 0;
    if (use_fast_triangle_path) {
        const int triangle_source_count = solver->cfg.triangle_count;
        const int triangle_source_blocks = block_count(std::max(triangle_source_count, 1));
        Solver triangle_solver = probe_solver;
        triangle_solver.self_source_mode = kSelfSourceFull;
        if (!build_self_triangle_hash(
                solver,
                triangle_solver,
                triangle_source_blocks,
                timings,
                "build fast probe self triangle hash")) {
            return false;
        }
        self_vertex_triangle_collision_kernel<<<vertex_source_blocks, 256>>>(probe_solver, 0);
    } else if (solver->self_sample_heads
        && solver->self_sample_next
        && probe_solver.self_sample_count > 0) {
        if (solver->self_sample_hash_valid && solver->self_sample_hash_dirty) {
            clear_self_surface_sample_hash_if_dirty_kernel<<<block_count(solver->self_sample_hash_table_size), 256>>>(probe_solver);
            build_self_surface_sample_hash_if_dirty_kernel<<<block_count(probe_solver.self_sample_count), 256>>>(probe_solver);
            solver->self_sample_hash_valid = 1;
            cudaMemset(solver->self_sample_hash_dirty, 0, sizeof(int));
        } else if (!solver->self_sample_hash_valid) {
            cudaMemset(solver->self_sample_heads, 0xff, sizeof(int) * solver->self_sample_hash_table_size);
            build_self_surface_sample_hash_kernel<<<block_count(probe_solver.self_sample_count), 256>>>(probe_solver);
            solver->self_sample_hash_valid = 1;
        }
        bool used_pair_probe = false;
        bool use_vs_pair_probe = probe_solver.cfg.self_collision_mode != kSelfCollisionModeFast
            && solver->self_compaction_used
            && host_self_pair_compaction_ready(solver, probe_solver.self_source_mode);
        if (use_vs_pair_probe) {
            if (!build_self_vs_pairs(
                    solver,
                    probe_solver,
                    vertex_source_blocks,
                    1,
                    timings,
                    "build probe vertex-surface pairs")) {
                return false;
            }
            if (!solver->self_vs_pair_current_overflow && current_self_vs_pair_count(solver) > 0) {
                if (!probe_self_vs_pairs(solver, probe_solver, timings, "probe vertex-surface pairs")) {
                    return false;
                }
                used_pair_probe = true;
            }
        }
        if (!used_pair_probe) {
            probe_self_vertex_surface_collision_kernel<<<vertex_source_blocks, 256>>>(probe_solver);
        }
    }
    if (!end_timed_segment(timings, &probe_segment, "end self probe timing")) {
        return false;
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
    const float trigger_gap = -0.05f * cloth_thickness;
    return valid_min_gap(diag.min_gap) && diag.min_gap < trigger_gap;
}

bool self_probe_triggers_retry(const SsblXpbdDiagnostics& diag, float cloth_thickness) {
    const float retry_gap = -0.50f * cloth_thickness;
    return valid_min_gap(diag.min_gap) && diag.min_gap < retry_gap;
}

bool run_jitter_stabilizer(Solver* solver, int v_blocks) {
    if (!solver || !solver->cfg.jitter_stabilizer_enabled) {
        return true;
    }
    if (!solver->jitter_frame_start_pos
        || !solver->jitter_prev_delta
        || !solver->jitter_score) {
        return set_error("jitter stabilizer buffers are not allocated");
    }
    const float cloth_thickness = std::max(solver->cfg.cloth_thickness, 1.0e-4f);
    const float threshold = std::max(1.0e-4f, cloth_thickness * kJitterDeltaThresholdScale);
    const float correction_limit = std::max(1.0e-4f, cloth_thickness * kJitterCorrectionLimitScale);
    jitter_stabilizer_kernel<<<v_blocks, 256>>>(*solver, threshold, correction_limit);
    return set_cuda_error(cudaGetLastError(), "launch jitter stabilizer");
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
    bool allow_retry,
    bool fast_cleanup_substep,
    bool run_jitter_filter,
    std::vector<TimedSegment>* timings
) {
    if (!solver) {
        return set_error("invalid solver");
    }
    const bool strict_self_collision = run_self_collision
        && solver->cfg.self_collision_mode >= kSelfCollisionModeStrict;
    const bool fast_self_collision = run_self_collision
        && solver->cfg.self_collision_mode == kSelfCollisionModeFast;
    if (allow_retry && strict_self_collision) {
        if (!backup_solver_state(solver) || !backup_step_diagnostics_state(solver)) {
            return false;
        }
    }

    TimedSegment segment;
    if (!begin_timed_segment(timings, kTimingConstraints, &segment, "start integrate timing")) {
        return false;
    }
    integrate_kernel<<<v_blocks, 256>>>(*solver, sub_dt);
    if (solver->pin_count > 0) {
        pin_project_kernel<<<p_blocks, 256>>>(*solver);
    }
    if (!end_timed_segment(timings, &segment, "end integrate timing")) {
        return false;
    }
    for (int it = 0; it < iterations; ++it) {
        if (!begin_timed_segment(timings, kTimingConstraints, &segment, "start constraint timing")) {
            return false;
        }
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
        if (!end_timed_segment(timings, &segment, "end constraint timing")) {
            return false;
        }
        if (run_volume_pressure && solver->volume_gradient && solver->volume_accum && solver->cfg.triangle_count > 0) {
            if (!begin_timed_segment(timings, kTimingVolume, &segment, "start volume timing")) {
                return false;
            }
            if (solver->volume_vertex_offsets
                && solver->volume_vertex_triangles
                && solver->volume_partial_values
                && solver->volume_partial_denominators
                && solver->volume_partial_capacity >= std::max(t_blocks, v_blocks)) {
                const size_t shared_floats = sizeof(float) * 256;
                volume_value_partial_kernel<<<t_blocks, 256, shared_floats>>>(*solver, solver->volume_partial_values);
                volume_gradient_incident_kernel<<<v_blocks, 256>>>(*solver);
                volume_denominator_partial_kernel<<<v_blocks, 256, shared_floats>>>(*solver, solver->volume_partial_denominators);
                volume_reduce_partials_kernel<<<1, 256, sizeof(float) * 512>>>(
                    solver->volume_partial_values,
                    t_blocks,
                    solver->volume_partial_denominators,
                    v_blocks,
                    solver->volume_accum
                );
            } else {
                cudaMemset(solver->volume_gradient, 0, sizeof(Vec3) * solver->cfg.vertex_count);
                cudaMemset(solver->volume_accum, 0, sizeof(float) * 2);
                volume_accumulate_kernel<<<t_blocks, 256>>>(*solver);
                volume_denominator_kernel<<<v_blocks, 256>>>(*solver);
            }
            volume_project_kernel<<<v_blocks, 256>>>(*solver, sub_dt);
            if (!end_timed_segment(timings, &segment, "end volume timing")) {
                return false;
            }
        }
        if (!begin_timed_segment(timings, kTimingConstraints, &segment, "start post-constraint timing")) {
            return false;
        }
        if (solver->pin_count > 0) {
            pin_project_kernel<<<p_blocks, 256>>>(*solver);
        }
        if (!end_timed_segment(timings, &segment, "end post-constraint timing")) {
            return false;
        }
        if (!begin_timed_segment(timings, kTimingAnalyticCollision, &segment, "start analytic collision timing")) {
            return false;
        }
        analytic_collision_kernel<<<v_blocks, 256>>>(*solver);
        if (!end_timed_segment(timings, &segment, "end analytic collision timing")) {
            return false;
        }
        if (solver->cfg.static_triangle_count > 0) {
            if (!begin_timed_segment(timings, kTimingStaticCollision, &segment, "start static collision timing")) {
                return false;
            }
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
            if (!end_timed_segment(timings, &segment, "end static collision timing")) {
                return false;
            }
        }
        if (solver->dynamic_particle_count > 0) {
            if (!begin_timed_segment(timings, kTimingDynamicParticleCollision, &segment, "start dynamic particle collision timing")) {
                return false;
            }
            dynamic_particle_collision_kernel<<<v_blocks, 256>>>(*solver);
            if (!end_timed_segment(timings, &segment, "end dynamic particle collision timing")) {
                return false;
            }
        }
        if (solver->dynamic_triangle_count > 0) {
            if (!begin_timed_segment(timings, kTimingDynamicCollision, &segment, "start dynamic collision timing")) {
                return false;
            }
            const int dynamic_passes = dynamic_collision_pass_count(solver);
            for (int dynamic_pass = 0; dynamic_pass < dynamic_passes; ++dynamic_pass) {
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
            if (!end_timed_segment(timings, &segment, "end dynamic collision timing")) {
                return false;
            }
        }
        if (run_self_collision && it == iterations - 1) {
            ++solver->self_collision_run_count;
            const bool use_source_compaction = solver->self_compaction_used != 0;
            const int solve_source_mode = use_source_compaction ? kSelfSourceActive : kSelfSourceFull;
            const int probe_source_mode = (fast_self_collision && fast_cleanup_substep)
                ? kSelfSourceFull
                : (use_source_compaction ? kSelfSourceSuspect : kSelfSourceFull);
            int surface_pair_interval = std::max(solver->cfg.self_surface_pair_interval, 1);
            bool run_surface_pairs = (solver->self_collision_run_count % surface_pair_interval) == 0;
            const int self_collision_passes = fast_self_collision ? kFastSelfCollisionPasses : kSelfCollisionPasses;
            solver->self_source_mode = solve_source_mode;
            for (int self_pass = 0; self_pass < self_collision_passes; ++self_pass) {
                const bool last_self_pass = self_pass == self_collision_passes - 1;
                bool run_fast_triangle_slot = fast_self_collision
                    && last_self_pass
                    && fast_cleanup_substep;
                bool run_surface_sample_pairs = (run_surface_pairs || run_fast_triangle_slot) && last_self_pass;
                if (!run_self_collision_pass(solver, v_blocks, false, run_surface_sample_pairs, run_fast_triangle_slot, timings)) {
                    solver->self_source_mode = kSelfSourceFull;
                    return false;
                }
            }
            solver->self_source_mode = kSelfSourceFull;

            int probe_interval = std::max(solver->cfg.self_probe_interval, 1);
            bool run_probe = (solver->self_collision_run_count % probe_interval) == 0;
            if (!run_probe) {
                if (!begin_timed_segment(timings, kTimingConstraints, &segment, "start sanitize timing")) {
                    return false;
                }
                sanitize_positions_kernel<<<v_blocks, 256>>>(*solver);
                if (!end_timed_segment(timings, &segment, "end sanitize timing")) {
                    return false;
                }
                solver->self_source_mode = kSelfSourceFull;
                continue;
            }

            SsblXpbdDiagnostics probe_diag{};
            solver->self_source_mode = probe_source_mode;
            if (!probe_self_collision(solver, v_blocks, &probe_diag, timings)) {
                solver->self_source_mode = kSelfSourceFull;
                return false;
            }
            float cloth_thickness = std::max(solver->cfg.cloth_thickness, 1.0e-4f);
            int extra_recovery_passes = 0;
            const int recovery_pass_limit = fast_self_collision ? 1 : kSelfRecoveryPassLimit;
            while (extra_recovery_passes < recovery_pass_limit
                && self_probe_triggers_recovery(probe_diag, cloth_thickness)) {
                if (!clear_self_recovery_tracking(solver, "clear self recovery tracking")) {
                    return false;
                }
                solver->self_source_mode = probe_source_mode;
                if (!run_self_collision_pass(solver, v_blocks, true, true, false, timings)) {
                    solver->self_source_mode = kSelfSourceFull;
                    return false;
                }
                if (!clamp_self_recovery_displacement(
                        solver,
                        v_blocks,
                        cloth_thickness,
                        kSelfRecoveryMaxDisplacementScale,
                        false,
                        timings)) {
                    solver->self_source_mode = kSelfSourceFull;
                    return false;
                }
                ++extra_recovery_passes;
                if (recovery_passes_total) {
                    ++(*recovery_passes_total);
                }
                solver->self_source_mode = probe_source_mode;
                if (fast_self_collision) {
                    break;
                }
                if (!probe_self_collision(solver, v_blocks, &probe_diag, timings)) {
                    solver->self_source_mode = kSelfSourceFull;
                    return false;
                }
            }
            if (!fast_self_collision && use_source_compaction && self_probe_triggers_retry(probe_diag, cloth_thickness)) {
                if (!clear_self_recovery_tracking(solver, "clear full self recovery tracking")) {
                    return false;
                }
                solver->self_source_mode = kSelfSourceFull;
                if (!run_self_collision_pass(solver, v_blocks, true, true, false, timings)) {
                    return false;
                }
                if (!clamp_self_recovery_displacement(
                        solver,
                        v_blocks,
                        cloth_thickness,
                        kSelfRecoveryMaxDisplacementScale,
                        false,
                        timings)) {
                    return false;
                }
                ++extra_recovery_passes;
                ++solver->self_full_recovery_fallbacks;
                if (recovery_passes_total) {
                    ++(*recovery_passes_total);
                }
                if (!probe_self_collision(solver, v_blocks, &probe_diag, timings)) {
                    return false;
                }
            }
            solver->self_source_mode = kSelfSourceFull;
            if (extra_recovery_passes > 0 && solver->self_recovery_touched) {
                if (!begin_timed_segment(timings, kTimingSelfRecovery, &segment, "start recovery velocity damping timing")) {
                    return false;
                }
                damp_self_recovery_velocity_kernel<<<v_blocks, 256>>>(*solver);
                if (!end_timed_segment(timings, &segment, "end recovery velocity damping timing")) {
                    return false;
                }
            }
            if (allow_retry && strict_self_collision && self_probe_triggers_retry(probe_diag, cloth_thickness)) {
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
                        false,
                        false,
                        false,
                        timings)) {
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
                    false,
                    false,
                    false,
                    timings
                );
            }
        }
        if (!begin_timed_segment(timings, kTimingConstraints, &segment, "start sanitize timing")) {
            return false;
        }
        sanitize_positions_kernel<<<v_blocks, 256>>>(*solver);
        if (!end_timed_segment(timings, &segment, "end sanitize timing")) {
            return false;
        }
    }
    if (run_jitter_filter) {
        if (!run_jitter_stabilizer(solver, v_blocks)) {
            return false;
        }
    }
    if (!begin_timed_segment(timings, kTimingConstraints, &segment, "start velocity timing")) {
        return false;
    }
    update_velocity_kernel<<<v_blocks, 256>>>(*solver, sub_dt);
    if (!end_timed_segment(timings, &segment, "end velocity timing")) {
        return false;
    }
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
    cudaFree(solver->volume_vertex_offsets);
    cudaFree(solver->volume_vertex_triangles);
    cudaFree(solver->volume_partial_values);
    cudaFree(solver->volume_partial_denominators);
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
    cudaFree(solver->dynamic_particles);
    cudaFree(solver->dynamic_particle_heads);
    cudaFree(solver->dynamic_particle_next);
    cudaFree(solver->dynamic_particle_index);
    cudaFree(solver->dynamic_particle_count_buffer);
    cudaFree(solver->external_contacts);
    cudaFree(solver->force_fields);
    cudaFree(solver->self_vert_heads);
    cudaFree(solver->self_vert_next);
    cudaFree(solver->self_edge_heads);
    cudaFree(solver->self_edge_next);
    cudaFree(solver->self_sample_heads);
    cudaFree(solver->self_sample_next);
    cudaFree(solver->self_sample_hash_dirty);
    cudaFree(solver->self_tri_heads);
    cudaFree(solver->self_tri_entry_next);
    cudaFree(solver->self_tri_entry_index);
    cudaFree(solver->self_tri_entry_count);
    cudaFree(solver->self_recovery_touched);
    cudaFree(solver->self_recovery_delta);
    cudaFree(solver->self_sleep_vertex_regions);
    cudaFree(solver->self_sleep_triangle_regions);
    cudaFree(solver->self_sleep_prev_pos);
    cudaFree(solver->self_sleep_region_still_frames);
    cudaFree(solver->self_sleep_region_sleeping);
    cudaFree(solver->self_sleep_region_motion);
    cudaFree(solver->self_sleep_region_touched);
    cudaFree(solver->self_sleep_region_vertex_counts);
    cudaFree(solver->self_sleep_has_sleeping);
    cudaFree(solver->self_active_vertices);
    cudaFree(solver->self_suspect_vertices);
    cudaFree(solver->self_active_vertex_flags);
    cudaFree(solver->self_suspect_vertex_flags);
    cudaFree(solver->self_active_samples);
    cudaFree(solver->self_suspect_samples);
    cudaFree(solver->self_compaction_counts);
    cudaFree(solver->self_vs_pairs);
    cudaFree(solver->self_vs_pair_counts);
    cudaFree(solver->jitter_frame_start_pos);
    cudaFree(solver->jitter_prev_delta);
    cudaFree(solver->jitter_score);
    cudaFree(solver->jitter_counts);
    cudaFree(solver->jitter_max_correction);
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

bool update_pin_targets_internal(Solver* solver, const int* indices, const float* positions, int count) {
    if (!solver) {
        return set_error("invalid solver handle");
    }
    if (count <= 0) {
        solver->pin_count = 0;
        return true;
    }
    if (!indices || !positions) {
        return set_error("missing pin targets");
    }
    if (count > solver->pin_capacity) {
        cudaFree(solver->pin_indices);
        cudaFree(solver->pin_targets);
        solver->pin_indices = nullptr;
        solver->pin_targets = nullptr;
        solver->pin_capacity = count;
        if (!set_cuda_error(cudaMalloc(reinterpret_cast<void**>(&solver->pin_indices), sizeof(int) * count), "pin index allocation")) {
            return false;
        }
        if (!set_cuda_error(cudaMalloc(reinterpret_cast<void**>(&solver->pin_targets), sizeof(Vec3) * count), "pin target allocation")) {
            return false;
        }
    }
    solver->pin_count = count;
    if (!set_cuda_error(cudaMemcpy(solver->pin_indices, indices, sizeof(int) * count, cudaMemcpyHostToDevice), "pin index upload")) {
        return false;
    }
    return set_cuda_error(cudaMemcpy(solver->pin_targets, positions, sizeof(float) * count * 3, cudaMemcpyHostToDevice), "pin target upload");
}

bool update_runtime_colliders_internal(Solver* solver, const SsblXpbdRuntimeColliders* inputs) {
    if (!solver || !inputs) {
        return set_error("invalid runtime collider update");
    }
    solver->cfg.use_ground = inputs->use_ground;
    solver->cfg.ground_height = inputs->ground_height;
    solver->cfg.use_wall = inputs->use_wall;
    std::memcpy(solver->cfg.wall_origin, inputs->wall_origin, sizeof(float) * 3);
    std::memcpy(solver->cfg.wall_normal, inputs->wall_normal, sizeof(float) * 3);
    solver->cfg.use_sphere = inputs->use_sphere;
    std::memcpy(solver->cfg.sphere_center, inputs->sphere_center, sizeof(float) * 3);
    solver->cfg.sphere_radius = inputs->sphere_radius;
    return true;
}

bool update_positions_internal(Solver* solver, const float* positions, int max_floats) {
    if (!solver || !positions) {
        return set_error("invalid position update");
    }
    const int needed = solver->cfg.vertex_count * 3;
    if (max_floats < needed) {
        return set_error("position update buffer is too small");
    }
    const size_t bytes = sizeof(float) * static_cast<size_t>(needed);
    if (!set_cuda_error(cudaMemcpy(solver->pos, positions, bytes, cudaMemcpyHostToDevice), "position upload")) {
        return false;
    }
    if (!set_cuda_error(cudaMemcpy(solver->prev, positions, bytes, cudaMemcpyHostToDevice), "previous-position upload")) {
        return false;
    }
    if (!set_cuda_error(cudaMemset(solver->vel, 0, sizeof(Vec3) * solver->cfg.vertex_count), "velocity reset")) {
        return false;
    }
    if (solver->self_sleep_prev_pos
        && !set_cuda_error(cudaMemcpy(solver->self_sleep_prev_pos, positions, bytes, cudaMemcpyHostToDevice), "self sleep previous-position upload")) {
        return false;
    }
    if (solver->self_sleep_region_sleeping
        && !set_cuda_error(cudaMemset(solver->self_sleep_region_sleeping, 0, sizeof(int) * solver->self_sleep_region_count), "self sleep state reset")) {
        return false;
    }
    if (solver->self_sleep_has_sleeping
        && !set_cuda_error(cudaMemset(solver->self_sleep_has_sleeping, 0, sizeof(int)), "self sleep summary reset")) {
        return false;
    }
    if (solver->jitter_frame_start_pos
        && !set_cuda_error(cudaMemcpy(solver->jitter_frame_start_pos, positions, bytes, cudaMemcpyHostToDevice), "jitter frame-start upload")) {
        return false;
    }
    if (solver->jitter_prev_delta
        && !set_cuda_error(cudaMemset(solver->jitter_prev_delta, 0, sizeof(Vec3) * solver->cfg.vertex_count), "jitter previous-delta reset")) {
        return false;
    }
    if (!clear_external_contact_cache(solver, "clear external contact cache after position upload")) {
        return false;
    }
    return true;
}

bool update_static_triangles_internal(Solver* solver, const float* triangles, int triangle_count) {
    if (!solver) {
        return set_error("invalid static collider update");
    }
    if (triangle_count != solver->cfg.static_triangle_count) {
        return set_error("static collider triangle count changed; fixed topology is required");
    }
    if (triangle_count <= 0) {
        return true;
    }
    if (!triangles || !solver->static_triangles) {
        return set_error("missing static collider triangles");
    }
    if (!set_cuda_error(
        cudaMemcpy(solver->static_triangles, triangles, sizeof(float) * triangle_count * 9, cudaMemcpyHostToDevice),
        "upload static triangles"
    )) {
        return false;
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
            return false;
        }
        solver->pending_hash_build_ms += elapsed_ms_since(hash_start);
    }
    return true;
}

bool update_dynamic_triangles_internal(Solver* solver, const float* triangles, int triangle_count) {
    if (!solver) {
        return set_error("invalid dynamic collider update");
    }
    int previous_triangle_count = solver->dynamic_triangle_count;
    if (triangle_count <= 0) {
        solver->dynamic_triangle_count = 0;
        if (previous_triangle_count > 0
            && !clear_external_contact_cache(solver, "clear external contact cache after dynamic collider removal")) {
            return false;
        }
        return true;
    }
    if (!triangles) {
        return set_error("missing dynamic collider triangles");
    }
    if (!allocate_dynamic_triangle_collision(solver, triangle_count)) {
        return false;
    }
    if (previous_triangle_count != triangle_count
        && !clear_external_contact_cache(solver, "clear external contact cache after dynamic topology change")) {
        return false;
    }
    if (!set_cuda_error(
        cudaMemcpy(solver->dynamic_triangles, triangles, sizeof(float) * triangle_count * 9, cudaMemcpyHostToDevice),
        "upload dynamic triangles"
    )) {
        return false;
    }
    return rebuild_dynamic_triangle_hash(solver);
}

bool update_dynamic_particles_internal(
    Solver* solver,
    const float* positions,
    const float* radii,
    const float* inv_mass,
    const int* slot_ids,
    const int* phases,
    int particle_count
) {
    if (!solver) {
        return set_error("invalid dynamic particle update");
    }
    if (particle_count <= 0) {
        solver->dynamic_particle_count = 0;
        solver->dynamic_particle_max_radius = 0.0f;
        return true;
    }
    if (!positions || !radii || !inv_mass || !slot_ids || !phases) {
        return set_error("missing dynamic particle data");
    }
    if (!allocate_dynamic_particle_collision(solver, particle_count)) {
        return false;
    }

    std::vector<DynamicParticle> particles(static_cast<size_t>(particle_count));
    float max_radius = 0.0f;
    for (int i = 0; i < particle_count; ++i) {
        DynamicParticle& particle = particles[static_cast<size_t>(i)];
        particle.position = {
            positions[i * 3 + 0],
            positions[i * 3 + 1],
            positions[i * 3 + 2],
        };
        particle.radius = std::isfinite(radii[i]) ? std::max(radii[i], 0.0f) : 0.0f;
        particle.inv_mass = std::isfinite(inv_mass[i]) ? std::max(inv_mass[i], 0.0f) : 0.0f;
        particle.slot_id = slot_ids[i];
        particle.phase = phases[i];
        max_radius = std::max(max_radius, particle.radius);
    }
    solver->dynamic_particle_count = particle_count;
    solver->dynamic_particle_max_radius = max_radius;
    if (!set_cuda_error(
            cudaMemcpy(
                solver->dynamic_particles,
                particles.data(),
                sizeof(DynamicParticle) * static_cast<size_t>(particle_count),
                cudaMemcpyHostToDevice),
            "upload dynamic particles")) {
        return false;
    }
    return rebuild_dynamic_particle_hash(solver);
}

bool update_force_fields_internal(
    Solver* solver,
    const SsblXpbdForceField* force_fields,
    int force_field_count,
    int unsupported_force_field_count
) {
    if (!solver) {
        return set_error("invalid force field update");
    }
    if (force_field_count < 0 || force_field_count > kMaxForceFields) {
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
        solver->force_field_capacity = force_field_count;
        cudaError_t err = cudaMalloc(
            reinterpret_cast<void**>(&solver->force_fields),
            sizeof(SsblXpbdForceField) * solver->force_field_capacity
        );
        if (!set_cuda_error(err, "force field allocation")) {
            solver->force_field_capacity = 0;
            return false;
        }
    }
    if (!set_cuda_error(
            cudaMemcpy(
                solver->force_fields,
                force_fields,
                sizeof(SsblXpbdForceField) * force_field_count,
                cudaMemcpyHostToDevice
            ),
            "upload force fields")) {
        return false;
    }
    solver->force_field_count = force_field_count;
    return true;
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
    // Off is controlled by self_collision=false. Enabled self-collision has
    // two solver modes: fast preview and strict no-intersection.
    solver->cfg.self_collision_mode = solver->cfg.self_collision
        ? std::min(std::max(solver->cfg.self_collision_mode, 0), kSelfCollisionModeStrict)
        : 0;
    solver->cfg.self_sleep_enabled = (solver->cfg.self_collision && solver->cfg.self_collision_mode > 0 && solver->cfg.self_sleep_enabled)
        ? 1
        : 0;
    solver->cfg.self_sleep_still_frames = std::clamp(solver->cfg.self_sleep_still_frames, 1, 60);
    solver->cfg.self_sleep_full_scan_interval = std::clamp(solver->cfg.self_sleep_full_scan_interval, 1, 240);
    solver->cfg.self_compaction_enabled = (solver->cfg.self_sleep_enabled && solver->cfg.self_compaction_enabled) ? 1 : 0;
    if (!std::isfinite(solver->cfg.self_sleep_motion_scale) || solver->cfg.self_sleep_motion_scale <= 0.0f) {
        solver->cfg.self_sleep_motion_scale = 1.0f;
    }
    solver->cfg.self_sleep_motion_scale = std::clamp(solver->cfg.self_sleep_motion_scale, 0.05f, 4.0f);
    if (!std::isfinite(solver->cfg.self_compaction_active_fraction_threshold)
        || solver->cfg.self_compaction_active_fraction_threshold <= 0.0f) {
        solver->cfg.self_compaction_active_fraction_threshold = 0.75f;
    }
    solver->cfg.self_compaction_active_fraction_threshold = std::clamp(
        solver->cfg.self_compaction_active_fraction_threshold,
        0.01f,
        1.0f
    );
    solver->cfg.self_pair_compaction_enabled = (solver->cfg.self_collision && solver->cfg.self_pair_compaction_enabled) ? 1 : 0;
    solver->cfg.jitter_stabilizer_enabled = (
        solver->cfg.self_collision
        && solver->cfg.self_collision_mode > 0
        && solver->cfg.jitter_stabilizer_enabled
    ) ? 1 : 0;
    if (!std::isfinite(solver->cfg.contact_friction) || solver->cfg.contact_friction < 0.0f) {
        solver->cfg.contact_friction = 0.35f;
    }
    solver->cfg.contact_friction = std::clamp(solver->cfg.contact_friction, 0.0f, 4.0f);
    if (!std::isfinite(solver->cfg.contact_tangent_damping) || solver->cfg.contact_tangent_damping < 0.0f) {
        solver->cfg.contact_tangent_damping = 0.2f;
    }
    solver->cfg.contact_tangent_damping = std::clamp(solver->cfg.contact_tangent_damping, 0.0f, 1.0f);
    if (!std::isfinite(solver->cfg.contact_compliance) || solver->cfg.contact_compliance < 0.0f) {
        solver->cfg.contact_compliance = 0.0f;
    }

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
    if (ok && solver->cfg.jitter_stabilizer_enabled) {
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&solver->jitter_frame_start_pos), sizeof(Vec3) * vertex_count);
        ok = ok && set_cuda_error(err, "jitter frame-start allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->jitter_prev_delta), sizeof(Vec3) * vertex_count);
        ok = ok && set_cuda_error(err, "jitter previous-delta allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->jitter_score), sizeof(int) * vertex_count);
        ok = ok && set_cuda_error(err, "jitter score allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->jitter_counts), sizeof(unsigned long long) * kJitterCountSlots);
        ok = ok && set_cuda_error(err, "jitter count allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->jitter_max_correction), sizeof(float));
        ok = ok && set_cuda_error(err, "jitter max-correction allocation");
        if (ok) {
            ok = ok && set_cuda_error(
                cudaMemcpy(solver->jitter_frame_start_pos, solver->pos, sizeof(Vec3) * vertex_count, cudaMemcpyDeviceToDevice),
                "jitter frame-start initialization"
            );
            ok = ok && set_cuda_error(
                cudaMemset(solver->jitter_prev_delta, 0, sizeof(Vec3) * vertex_count),
                "jitter previous-delta reset"
            );
            ok = ok && set_cuda_error(
                cudaMemset(solver->jitter_score, 0, sizeof(int) * vertex_count),
                "jitter score reset"
            );
            ok = ok && set_cuda_error(
                cudaMemset(solver->jitter_counts, 0, sizeof(unsigned long long) * kJitterCountSlots),
                "jitter count reset"
            );
            ok = ok && set_cuda_error(
                cudaMemset(solver->jitter_max_correction, 0, sizeof(float)),
                "jitter max-correction reset"
            );
        }
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
        solver->volume_partial_capacity = std::max(1, std::max(block_count(vertex_count), block_count(config->triangle_count)));
        err = cudaMalloc(reinterpret_cast<void**>(&solver->volume_partial_values), sizeof(float) * solver->volume_partial_capacity);
        ok = ok && set_cuda_error(err, "volume partial value allocation");
        err = cudaMalloc(reinterpret_cast<void**>(&solver->volume_partial_denominators), sizeof(float) * solver->volume_partial_capacity);
        ok = ok && set_cuda_error(err, "volume partial denominator allocation");
    }
    if (ok) {
        solver->pinned_download_floats = vertex_count * 3;
        cudaError_t err = cudaMallocHost(
            reinterpret_cast<void**>(&solver->pinned_download),
            sizeof(float) * solver->pinned_download_floats
        );
        ok = ok && set_cuda_error(err, "pinned download allocation");
    }
    if (ok) {
        long long requested_capacity = std::clamp(
            static_cast<long long>(vertex_count) * 4ll,
            static_cast<long long>(kExternalContactCapacityMin),
            static_cast<long long>(kExternalContactCapacityMax)
        );
        solver->external_contact_capacity = static_cast<int>(requested_capacity);
        cudaError_t err = cudaMalloc(
            reinterpret_cast<void**>(&solver->external_contacts),
            sizeof(ExternalContact) * solver->external_contact_capacity
        );
        ok = ok && set_cuda_error(err, "external contact cache allocation");
        if (ok) {
            ok = ok && clear_external_contact_cache(solver, "external contact cache reset");
        }
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
    if (ok && (config->use_volume_pressure || solver->cfg.jitter_stabilizer_enabled)) {
        ok = ok && build_volume_vertex_triangles(solver, config, mesh);
    }
    solver->static_collider_complex = static_collider_is_complex(config, mesh);
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
        if (ok && config->edge_count > 0) {
            solver->self_edge_hash_table_size = next_power_of_two(std::max(1024, config->edge_count * 2));
            err = cudaMalloc(reinterpret_cast<void**>(&solver->self_edge_heads), sizeof(int) * solver->self_edge_hash_table_size);
            ok = ok && set_cuda_error(err, "self edge hash allocation");
            err = cudaMalloc(reinterpret_cast<void**>(&solver->self_edge_next), sizeof(int) * config->edge_count);
            ok = ok && set_cuda_error(err, "self edge link allocation");
        }
        err = cudaMalloc(reinterpret_cast<void**>(&solver->self_recovery_touched), sizeof(int) * config->vertex_count);
        ok = ok && set_cuda_error(err, "self recovery touched allocation");
        ok = ok && set_cuda_error(
            cudaMemset(solver->self_recovery_touched, 0, sizeof(int) * config->vertex_count),
            "self recovery touched reset"
        );
        err = cudaMalloc(reinterpret_cast<void**>(&solver->self_recovery_delta), sizeof(Vec3) * config->vertex_count);
        ok = ok && set_cuda_error(err, "self recovery delta allocation");
        ok = ok && set_cuda_error(
            cudaMemset(solver->self_recovery_delta, 0, sizeof(Vec3) * config->vertex_count),
            "self recovery delta reset"
        );
        ok = ok && build_self_sleep_regions(solver, config, mesh, host_pos);
        if (ok && solver->cfg.self_compaction_enabled) {
            err = cudaMalloc(reinterpret_cast<void**>(&solver->self_active_vertices), sizeof(int) * config->vertex_count);
            ok = ok && set_cuda_error(err, "self active vertex allocation");
            err = cudaMalloc(reinterpret_cast<void**>(&solver->self_suspect_vertices), sizeof(int) * config->vertex_count);
            ok = ok && set_cuda_error(err, "self suspect vertex allocation");
            err = cudaMalloc(reinterpret_cast<void**>(&solver->self_active_vertex_flags), sizeof(int) * config->vertex_count);
            ok = ok && set_cuda_error(err, "self active vertex flag allocation");
            err = cudaMalloc(reinterpret_cast<void**>(&solver->self_suspect_vertex_flags), sizeof(int) * config->vertex_count);
            ok = ok && set_cuda_error(err, "self suspect vertex flag allocation");
            err = cudaMalloc(reinterpret_cast<void**>(&solver->self_compaction_counts), sizeof(int) * kSelfCompactionCountSlots);
            ok = ok && set_cuda_error(err, "self compaction count allocation");
            if (ok) {
                ok = ok && set_cuda_error(
                    cudaMemset(solver->self_compaction_counts, 0, sizeof(int) * kSelfCompactionCountSlots),
                    "self compaction count reset"
                );
                ok = ok && set_cuda_error(
                    cudaMemset(solver->self_active_vertex_flags, 0, sizeof(int) * config->vertex_count),
                    "self active vertex flag reset"
                );
                ok = ok && set_cuda_error(
                    cudaMemset(solver->self_suspect_vertex_flags, 0, sizeof(int) * config->vertex_count),
                    "self suspect vertex flag reset"
                );
            }
        }
        if (ok && config->triangle_count > 0) {
            solver->self_sample_count = config->triangle_count * solver->self_samples_per_triangle;
            solver->self_sample_hash_table_size = next_power_of_two(std::max(1024, solver->self_sample_count * 2));
            err = cudaMalloc(reinterpret_cast<void**>(&solver->self_sample_heads), sizeof(int) * solver->self_sample_hash_table_size);
            ok = ok && set_cuda_error(err, "self surface sample hash allocation");
            err = cudaMalloc(reinterpret_cast<void**>(&solver->self_sample_next), sizeof(int) * solver->self_sample_count);
            ok = ok && set_cuda_error(err, "self surface sample link allocation");
            err = cudaMalloc(reinterpret_cast<void**>(&solver->self_sample_hash_dirty), sizeof(int));
            ok = ok && set_cuda_error(err, "self surface sample dirty allocation");
            ok = ok && set_cuda_error(cudaMemset(solver->self_sample_hash_dirty, 0, sizeof(int)), "self surface sample dirty reset");
            solver->self_tri_hash_table_size = next_power_of_two(std::max(1024, config->triangle_count * 4));
            solver->self_tri_entry_capacity = static_cast<int>(std::min(
                static_cast<long long>(config->triangle_count) * static_cast<long long>(kMaxSelfTriangleHashCells),
                static_cast<long long>(kSelfTriangleEntryCapacityMax)
            ));
            err = cudaMalloc(reinterpret_cast<void**>(&solver->self_tri_heads), sizeof(int) * solver->self_tri_hash_table_size);
            ok = ok && set_cuda_error(err, "self triangle hash allocation");
            err = cudaMalloc(reinterpret_cast<void**>(&solver->self_tri_entry_next), sizeof(int) * solver->self_tri_entry_capacity);
            ok = ok && set_cuda_error(err, "self triangle hash link allocation");
            err = cudaMalloc(reinterpret_cast<void**>(&solver->self_tri_entry_index), sizeof(int) * solver->self_tri_entry_capacity);
            ok = ok && set_cuda_error(err, "self triangle hash index allocation");
            err = cudaMalloc(reinterpret_cast<void**>(&solver->self_tri_entry_count), sizeof(int));
            ok = ok && set_cuda_error(err, "self triangle hash count allocation");
            if (ok) {
                ok = ok && set_cuda_error(
                    cudaMemset(solver->self_tri_heads, 0xff, sizeof(int) * solver->self_tri_hash_table_size),
                    "self triangle hash reset"
                );
                ok = ok && set_cuda_error(cudaMemset(solver->self_tri_entry_count, 0, sizeof(int)), "self triangle hash count reset");
            }
            if (ok && solver->cfg.self_compaction_enabled) {
                err = cudaMalloc(reinterpret_cast<void**>(&solver->self_active_samples), sizeof(int) * solver->self_sample_count);
                ok = ok && set_cuda_error(err, "self active sample allocation");
                err = cudaMalloc(reinterpret_cast<void**>(&solver->self_suspect_samples), sizeof(int) * solver->self_sample_count);
                ok = ok && set_cuda_error(err, "self suspect sample allocation");
            }
            if (ok && solver->cfg.self_pair_compaction_enabled) {
                long long requested_capacity = std::max(
                    static_cast<long long>(config->vertex_count) * 32ll,
                    static_cast<long long>(solver->self_sample_count) * 2ll
                );
                requested_capacity = std::clamp(
                    requested_capacity,
                    static_cast<long long>(kSelfVsPairCapacityMin),
                    static_cast<long long>(kSelfVsPairCapacityMax)
                );
                solver->self_vs_pair_capacity = static_cast<int>(requested_capacity);
                err = cudaMalloc(reinterpret_cast<void**>(&solver->self_vs_pairs), sizeof(Int2) * solver->self_vs_pair_capacity);
                ok = ok && set_cuda_error(err, "self vertex-surface pair allocation");
                err = cudaMalloc(reinterpret_cast<void**>(&solver->self_vs_pair_counts), sizeof(int) * kSelfVsPairCountSlots);
                ok = ok && set_cuda_error(err, "self vertex-surface pair counter allocation");
                if (ok) {
                    ok = ok && set_cuda_error(
                        cudaMemset(solver->self_vs_pair_counts, 0, sizeof(int) * kSelfVsPairCountSlots),
                        "self vertex-surface pair counter reset"
                    );
                }
            }
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
    if (solver->self_recovery_touched
        && !set_cuda_error(
            cudaMemset(solver->self_recovery_touched, 0, sizeof(int) * n),
            "reset self recovery touched flags")) {
        return 0;
    }
    if (solver->self_recovery_delta
        && !set_cuda_error(
            cudaMemset(solver->self_recovery_delta, 0, sizeof(Vec3) * n),
            "reset self recovery delta")) {
        return 0;
    }
    if (solver->self_sleep_prev_pos
        && !set_cuda_error(
            cudaMemcpy(solver->self_sleep_prev_pos, solver->rest, sizeof(Vec3) * n, cudaMemcpyDeviceToDevice),
            "reset self sleep previous positions")) {
        return 0;
    }
    if (solver->self_sleep_region_count > 0) {
        int r = solver->self_sleep_region_count;
        if (solver->self_sleep_region_still_frames
            && !set_cuda_error(cudaMemset(solver->self_sleep_region_still_frames, 0, sizeof(int) * r), "reset self sleep still frames")) {
            return 0;
        }
        if (solver->self_sleep_region_sleeping
            && !set_cuda_error(cudaMemset(solver->self_sleep_region_sleeping, 0, sizeof(int) * r), "reset self sleep states")) {
            return 0;
        }
        if (solver->self_sleep_region_motion
            && !set_cuda_error(cudaMemset(solver->self_sleep_region_motion, 0, sizeof(int) * r), "reset self sleep motion flags")) {
            return 0;
        }
        if (solver->self_sleep_region_touched
            && !set_cuda_error(cudaMemset(solver->self_sleep_region_touched, 0, sizeof(int) * r), "reset self sleep touch flags")) {
            return 0;
        }
        if (solver->self_sleep_has_sleeping
            && !set_cuda_error(cudaMemset(solver->self_sleep_has_sleeping, 0, sizeof(int)), "reset self sleep summary")) {
            return 0;
        }
        solver->self_sleep_frame_count = 0;
        solver->self_sleep_force_active = 0;
    }
    if (solver->jitter_frame_start_pos
        && !set_cuda_error(
            cudaMemcpy(solver->jitter_frame_start_pos, solver->rest, sizeof(Vec3) * n, cudaMemcpyDeviceToDevice),
            "reset jitter frame-start positions")) {
        return 0;
    }
    if (solver->jitter_prev_delta
        && !set_cuda_error(cudaMemset(solver->jitter_prev_delta, 0, sizeof(Vec3) * n), "reset jitter previous deltas")) {
        return 0;
    }
    if (solver->jitter_score
        && !set_cuda_error(cudaMemset(solver->jitter_score, 0, sizeof(int) * n), "reset jitter scores")) {
        return 0;
    }
    if (!reset_jitter_diagnostics(solver)) {
        return 0;
    }
    if (!clear_external_contact_cache(solver, "reset external contact cache")) {
        return 0;
    }
    reset_self_compaction_state(solver);
    return 1;
}

extern "C" SSBL_API int ssbl_update_pin_targets(void* handle, const int* indices, const float* positions, int count) {
    g_last_error.clear();
    return update_pin_targets_internal(reinterpret_cast<Solver*>(handle), indices, positions, count) ? 1 : 0;
}

extern "C" SSBL_API int ssbl_update_runtime_colliders(void* handle, const SsblXpbdRuntimeColliders* inputs) {
    g_last_error.clear();
    return update_runtime_colliders_internal(reinterpret_cast<Solver*>(handle), inputs) ? 1 : 0;
}

extern "C" SSBL_API int ssbl_update_positions(void* handle, const float* positions, int max_floats) {
    g_last_error.clear();
    return update_positions_internal(reinterpret_cast<Solver*>(handle), positions, max_floats) ? 1 : 0;
}

extern "C" SSBL_API int ssbl_update_static_triangles(void* handle, const float* triangles, int triangle_count) {
    g_last_error.clear();
    return update_static_triangles_internal(reinterpret_cast<Solver*>(handle), triangles, triangle_count) ? 1 : 0;
}

extern "C" SSBL_API int ssbl_update_dynamic_triangles(void* handle, const float* triangles, int triangle_count) {
    g_last_error.clear();
    return update_dynamic_triangles_internal(reinterpret_cast<Solver*>(handle), triangles, triangle_count) ? 1 : 0;
}

extern "C" SSBL_API int ssbl_update_frame_inputs(void* handle, const SsblXpbdFrameInputs* inputs) {
    g_last_error.clear();
    Solver* solver = reinterpret_cast<Solver*>(handle);
    if (!solver || !inputs) {
        return set_error("invalid frame input update") ? 1 : 0;
    }
    if (inputs->update_runtime_colliders && !update_runtime_colliders_internal(solver, &inputs->runtime_colliders)) {
        return 0;
    }
    if (inputs->update_pin_targets
        && !update_pin_targets_internal(solver, inputs->pin_indices, inputs->pin_positions, inputs->pin_count)) {
        return 0;
    }
    if (inputs->update_static_triangles
        && !update_static_triangles_internal(solver, inputs->static_triangles, inputs->static_triangle_count)) {
        return 0;
    }
    if (inputs->update_dynamic_triangles
        && !update_dynamic_triangles_internal(solver, inputs->dynamic_triangles, inputs->dynamic_triangle_count)) {
        return 0;
    }
    if (inputs->update_dynamic_particles
        && !update_dynamic_particles_internal(
            solver,
            inputs->dynamic_particle_positions,
            inputs->dynamic_particle_radii,
            inputs->dynamic_particle_inv_mass,
            inputs->dynamic_particle_slot_ids,
            inputs->dynamic_particle_phases,
            inputs->dynamic_particle_count)) {
        return 0;
    }
    if (inputs->update_force_fields
        && !update_force_fields_internal(
            solver,
            inputs->force_fields,
            inputs->force_field_count,
            inputs->unsupported_force_field_count)) {
        return 0;
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
    Solver* solver = reinterpret_cast<Solver*>(handle);
    if (!solver) {
        return set_error("invalid solver handle") ? 1 : 0;
    }
    if (!reset_step_diagnostics(solver)) {
        return 0;
    }
    if (!reset_jitter_diagnostics(solver)) {
        return 0;
    }
    reset_self_compaction_state(solver);
    if (solver->external_contacts && solver->external_contact_capacity > 0) {
        begin_external_contact_cache_step_kernel<<<block_count(solver->external_contact_capacity), 256>>>(*solver);
        if (!set_cuda_error(cudaGetLastError(), "begin external contact cache step")) {
            return 0;
        }
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
    const bool needs_sync = force_sync != 0 || fetch_diagnostics != 0;
    std::vector<TimedSegment> timings;
    std::vector<TimedSegment>* timing_ptr = needs_sync ? &timings : nullptr;
    const bool use_self_sleep = solver->cfg.self_sleep_enabled
        && solver->cfg.self_collision
        && solver->self_sleep_region_count > 0;
    if (solver->cfg.jitter_stabilizer_enabled && solver->jitter_frame_start_pos) {
        if (!set_cuda_error(
                cudaMemcpy(
                    solver->jitter_frame_start_pos,
                    solver->pos,
                    sizeof(Vec3) * solver->cfg.vertex_count,
                    cudaMemcpyDeviceToDevice
                ),
                "save jitter stabilizer frame start positions")) {
            destroy_timing_records(timing_ptr);
            return 0;
        }
    }
    solver->self_sleep_force_active = 0;
    if (use_self_sleep) {
        clear_self_sleep_frame_flags_kernel<<<block_count(solver->self_sleep_region_count), 256>>>(*solver);
        int full_scan_interval = std::max(solver->cfg.self_sleep_full_scan_interval, 1);
        bool force_full_scan = solver->self_sleep_frame_count > 0
            && (solver->self_sleep_frame_count % full_scan_interval) == 0;
        if (force_full_scan) {
            solver->self_sleep_force_active = 1;
            wake_self_sleep_regions_kernel<<<block_count(solver->self_sleep_region_count), 256>>>(*solver);
        }
        if (!set_cuda_error(cudaGetLastError(), "launch self sleep frame prep")) {
            destroy_timing_records(timing_ptr);
            return 0;
        }
    }
    if (!prepare_self_compaction_lists(solver, v_blocks)) {
        destroy_timing_records(timing_ptr);
        return 0;
    }

    for (int s = 0; s < substeps; ++s) {
        int interval = std::max(solver->cfg.self_collision_interval, 1);
        bool run_self_collision = solver->cfg.self_collision
            && (((s + 1) % interval) == 0 || s == substeps - 1);
        int volume_interval = std::max(solver->cfg.volume_solve_interval, 1);
        bool run_volume_pressure = solver->cfg.use_volume_pressure
            && (((s + 1) % volume_interval) == 0 || s == substeps - 1);
        bool run_jitter_filter = solver->cfg.jitter_stabilizer_enabled && s == substeps - 1;
        bool fast_triangle_cleanup_substep = s >= std::max(0, substeps - interval * kFastSelfTriangleCleanupRuns);
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
                true,
                fast_triangle_cleanup_substep,
                run_jitter_filter,
                timing_ptr)) {
            destroy_timing_records(timing_ptr);
            return 0;
        }
    }
    if (use_self_sleep) {
        solver->self_sleep_force_active = 0;
        float motion_threshold = std::max(1.0e-4f, solver->cfg.cloth_thickness * solver->cfg.self_sleep_motion_scale);
        update_self_sleep_motion_kernel<<<v_blocks, 256>>>(*solver, motion_threshold);
        clear_self_sleep_summary_kernel<<<1, 1>>>(*solver);
        finalize_self_sleep_regions_kernel<<<block_count(solver->self_sleep_region_count), 256>>>(*solver);
        ++solver->self_sleep_frame_count;
        if (!set_cuda_error(cudaGetLastError(), "launch self sleep frame update")) {
            destroy_timing_records(timing_ptr);
            return 0;
        }
    }
    if (solver->external_contacts && solver->external_contact_capacity > 0) {
        finalize_external_contact_cache_step_kernel<<<block_count(solver->external_contact_capacity), 256>>>(*solver);
        if (!set_cuda_error(cudaGetLastError(), "finalize external contact cache step")) {
            destroy_timing_records(timing_ptr);
            return 0;
        }
    }

    if (needs_sync) {
        const auto sync_start = std::chrono::high_resolution_clock::now();
        if (!set_cuda_error(cudaDeviceSynchronize(), "solver step")) {
            destroy_timing_records(timing_ptr);
            return 0;
        }
        solver->diag.sync_ms = elapsed_ms_since(sync_start);
        if (!collect_timing_records(solver, timing_ptr)) {
            return 0;
        }
    } else {
        if (!set_cuda_error(cudaGetLastError(), "launch solver step")) {
            return 0;
        }
    }
    solver->diag.step_ms = elapsed_ms_since(step_start);
    solver->diag.hash_build_ms = pending_hash_build_ms;
    if (fetch_diagnostics != 0) {
        const auto diagnostics_start = std::chrono::high_resolution_clock::now();
        if (!fetch_step_diagnostics(solver)) {
            return 0;
        }
        if (!fetch_self_compaction_diagnostics(solver)) {
            return 0;
        }
        solver->diag.diagnostics_fetch_ms = elapsed_ms_since(diagnostics_start);
    }
    solver->diag.recovery_passes = recovery_passes_total;
    solver->diag.local_retry_count = local_retry_total;
    return 1;
}

extern "C" SSBL_API int ssbl_step_solver(void* handle, int substeps, int iterations) {
    return ssbl_step_solver_ex(handle, substeps, iterations, 1, 1);
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
