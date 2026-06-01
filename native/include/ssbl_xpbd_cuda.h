#pragma once

#ifdef _WIN32
#define SSBL_API __declspec(dllexport)
#else
#define SSBL_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SsblXpbdConfig {
    int vertex_count;
    int edge_count;
    int bend_count;
    int lra_count;
    int triangle_count;
    int static_triangle_count;
    int edge_color_count;
    int bend_color_count;
    int lra_color_count;
    float dt;
    float damping;
    float gravity[3];
    float stretch_compliance;
    float bend_compliance;
    float lra_compliance;
    float collision_margin;
    int use_ground;
    float ground_height;
    int use_wall;
    float wall_origin[3];
    float wall_normal[3];
    int use_sphere;
    float sphere_center[3];
    float sphere_radius;
    int self_collision;
    int self_collision_mode;
    float cloth_thickness;
    int self_collision_interval;
    int max_self_collision_neighbors;
    int use_volume_pressure;
    float rest_volume;
    float volume_compliance;
    float pressure_strength;
    float volume_target_scale;
    int volume_solve_interval;
    int self_probe_interval;
    int self_surface_pair_interval;
} SsblXpbdConfig;

typedef struct SsblXpbdDiagnostics {
    float step_ms;
    float hash_build_ms;
    float constraints_ms;
    float volume_ms;
    float static_collision_ms;
    float dynamic_collision_ms;
    float self_hash_ms;
    float self_solve_ms;
    float self_probe_ms;
    float self_recovery_ms;
    float sync_ms;
    float diagnostics_fetch_ms;
    long long candidate_count;
    long long resolved_contacts;
    float min_gap;
    long long ccd_clamp_count;
    long long recovery_passes;
    long long local_retry_count;
    int finite_flag;
} SsblXpbdDiagnostics;

typedef struct SsblXpbdMesh {
    const float* positions;
    const float* inv_mass;
    const int* edges;
    const float* edge_rest_lengths;
    const int* edge_color_offsets;
    const int* bends;
    const float* bend_rest_lengths;
    const int* bend_color_offsets;
    const int* lra_edges;
    const float* lra_rest_lengths;
    const int* lra_color_offsets;
    const int* triangles;
    const float* static_triangles;
} SsblXpbdMesh;

typedef struct SsblXpbdRuntimeColliders {
    int use_ground;
    float ground_height;
    int use_wall;
    float wall_origin[3];
    float wall_normal[3];
    int use_sphere;
    float sphere_center[3];
    float sphere_radius;
} SsblXpbdRuntimeColliders;

typedef struct SsblXpbdFrameInputs {
    int update_pin_targets;
    const int* pin_indices;
    const float* pin_positions;
    int pin_count;
    int update_runtime_colliders;
    SsblXpbdRuntimeColliders runtime_colliders;
    int update_static_triangles;
    const float* static_triangles;
    int static_triangle_count;
    int update_dynamic_triangles;
    const float* dynamic_triangles;
    int dynamic_triangle_count;
} SsblXpbdFrameInputs;

SSBL_API void* ssbl_create_solver(const SsblXpbdConfig* config, const SsblXpbdMesh* mesh);
SSBL_API int ssbl_destroy_solver(void* handle);
SSBL_API int ssbl_reset_solver(void* handle);
SSBL_API int ssbl_update_pin_targets(void* handle, const int* indices, const float* positions, int count);
SSBL_API int ssbl_update_runtime_colliders(void* handle, const SsblXpbdRuntimeColliders* inputs);
SSBL_API int ssbl_update_static_triangles(void* handle, const float* triangles, int triangle_count);
SSBL_API int ssbl_update_dynamic_triangles(void* handle, const float* triangles, int triangle_count);
SSBL_API int ssbl_update_frame_inputs(void* handle, const SsblXpbdFrameInputs* inputs);
SSBL_API int ssbl_step_solver(void* handle, int substeps, int iterations);
SSBL_API int ssbl_step_solver_ex(void* handle, int substeps, int iterations, int fetch_diagnostics, int force_sync);
SSBL_API int ssbl_download_positions(void* handle, float* out_positions, int max_floats);
SSBL_API int ssbl_get_diagnostics(void* handle, SsblXpbdDiagnostics* out_diag);
SSBL_API const char* ssbl_last_error(void);

#ifdef __cplusplus
}
#endif
