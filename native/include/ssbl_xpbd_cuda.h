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
    int jitter_stabilizer_enabled;
    float contact_friction;
    float contact_tangent_damping;
    float contact_compliance;
    int fast_self_collision_passes;
    int stretch_optimization_enabled;
    float stretch_optimization_strength;
    float static_sdf_voxel_size;
    int static_sdf_band_voxels;
    int static_sdf_max_resolution;
} SsblXpbdConfig;

enum {
    SSBL_CAP_STRETCH_OPTIMIZATION = 1u << 0,
    SSBL_CAP_PIN_WEIGHTS = 1u << 1
};

typedef struct SsblXpbdDiagnostics {
    float step_ms;
    float hash_build_ms;
    float constraints_ms;
    float volume_ms;
    float analytic_collision_ms;
    float static_collision_ms;
    float dynamic_collision_ms;
    float dynamic_particle_collision_ms;
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
    long long jitter_stabilized_vertices;
    long long jitter_rejected_vertices;
    float jitter_max_correction;
    long long external_contact_cache_hits;
    long long external_contact_cache_misses;
    long long external_contact_cache_count;
    long long external_contact_cache_overflow;
    long long external_friction_corrections;
    long long force_field_count;
    long long unsupported_force_field_count;
    long long dynamic_particle_count;
    long long dynamic_particle_candidate_count;
    long long dynamic_particle_contacts;
    long long dynamic_particle_overflow;
    long long dynamic_triangle_count;
    long long static_triangle_count;
    int finite_flag;
    long long fast_exact_vt_candidates;
    long long fast_exact_vt_projected;
    long long fast_exact_vt_guarded;
    long long fast_exact_vt_skipped_rest;
    long long fast_soft_repulsion_candidates;
    long long fast_soft_repulsion_applied;
    float fast_soft_repulsion_max_push;
    long long fast_hard_projection_count;
    long long fast_manifold_contacts;
    long long fast_manifold_reused;
    long long fast_barrier_projected;
    long long fast_barrier_smoothed_vertices;
    long long fast_barrier_overflow;
    float fast_barrier_max_delta;
    long long fast_edge_edge_candidates;
    long long fast_edge_edge_contacts;
    long long fast_triangle_pair_candidates;
    long long fast_triangle_pair_contacts;
    long long fast_triangle_pair_skipped_rest;
    long long fast_contact_classification_guarded;
    long long fast_region_cluster_candidates;
    long long fast_region_cluster_contacts;
    long long fast_region_cluster_guarded;
    long long fast_overlap_island_candidates;
    long long fast_overlap_island_clusters;
    long long fast_overlap_island_vertex_refs;
    long long fast_overlap_island_applied_vertices;
    long long fast_overlap_island_guarded;
    float fast_overlap_island_max_delta;
    long long fast_cc_overlap_components;
    long long fast_cc_overlap_seed_triangles;
    long long fast_cc_overlap_owned_vertices;
    long long fast_cc_overlap_union_edges;
    long long fast_cc_overlap_guarded;
    long long fast_cc_overlap_applied_vertices;
    float fast_cc_overlap_max_delta;
    long long abi41_soft_contact_count;
    long long abi41_exact_impulse_contact_count;
    long long abi41_edge_edge_contact_count;
    float abi41_max_smoothed_delta;
    long long abi41_hard_projection_fallbacks;
    long long static_sdf_rebuild_count;
    long long static_sdf_voxel_count;
    long long static_sdf_grid_x;
    long long static_sdf_grid_y;
    long long static_sdf_grid_z;
    float static_sdf_build_ms;
    long long static_sdf_contact_count;
    long long static_sdf_unsigned_fallback_count;
    long long abi41_pcg_iterations;
    long long abi41_pcg_guarded;
    long long abi41_pcg_csr_nnz;
    long long abi41_pcg_texture_ready;
    float abi41_pcg_initial_residual;
    float abi41_pcg_final_residual;
    float abi41_pcg_max_delta;
    long long abi41_lra_tack_count;
    long long abi41_bending_wing_count;
    long long abi41_bending_texture_ready;
    long long abi41_tack_jitter_guarded;
    long long abi41_bending_guarded;
    float dynamic_collider_pack_ms;
    float dynamic_triangle_upload_ms;
    float dynamic_particle_upload_ms;
    long long dynamic_collider_cache_hits;
    long long dynamic_collider_cache_misses;
    float abi41_pcg_solve_ms;
    float abi41_pcg_system_ms;
    float abi41_pcg_ad_ms;
    float abi41_direct_stretch_ms;
    long long dynamic_pair_cache_hits;
    long long dynamic_pair_cache_misses;
    long long dynamic_pair_cache_reused_triangles;
    long long dynamic_pair_cache_reused_particles;
    long long dynamic_collision_skipped_launches;
    long long self_collision_skipped_launches;
    long long self_candidate_count;
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

typedef struct SsblXpbdForceField {
    int type;
    int use_min_distance;
    int use_max_distance;
    int seed;
    float strength;
    float origin[3];
    float direction[3];
    float axis[3];
    float falloff_power;
    float distance_min;
    float distance_max;
    float radial_min;
    float radial_max;
    float noise;
    float linear_drag;
    float quadratic_drag;
    float harmonic_damping;
    float flow;
    float size;
    float rest_length;
    float radial_falloff;
    float texture_nabla;
    int use_radial_min;
    int use_radial_max;
    int use_2d_force;
} SsblXpbdForceField;

typedef struct SsblXpbdFrameInputs {
    int update_pin_targets;
    const int* pin_indices;
    const float* pin_positions;
    const float* pin_weights;
    int pin_count;
    int update_runtime_colliders;
    SsblXpbdRuntimeColliders runtime_colliders;
    int update_static_triangles;
    const float* static_triangles;
    int static_triangle_count;
    int update_dynamic_triangles;
    const float* dynamic_triangles;
    int dynamic_triangle_count;
    int update_dynamic_particles;
    const float* dynamic_particle_positions;
    const float* dynamic_particle_radii;
    const float* dynamic_particle_inv_mass;
    const int* dynamic_particle_slot_ids;
    const int* dynamic_particle_phases;
    int dynamic_particle_count;
    int update_force_fields;
    const SsblXpbdForceField* force_fields;
    int force_field_count;
    int unsupported_force_field_count;
} SsblXpbdFrameInputs;

SSBL_API void* ssbl_create_solver(const SsblXpbdConfig* config, const SsblXpbdMesh* mesh);
SSBL_API int ssbl_destroy_solver(void* handle);
SSBL_API int ssbl_reset_solver(void* handle);
SSBL_API int ssbl_update_pin_targets(void* handle, const int* indices, const float* positions, const float* weights, int count);
SSBL_API int ssbl_update_runtime_colliders(void* handle, const SsblXpbdRuntimeColliders* inputs);
SSBL_API int ssbl_update_positions(void* handle, const float* positions, int max_floats);
SSBL_API int ssbl_update_static_triangles(void* handle, const float* triangles, int triangle_count);
SSBL_API int ssbl_update_dynamic_triangles(void* handle, const float* triangles, int triangle_count);
SSBL_API int ssbl_update_frame_inputs(void* handle, const SsblXpbdFrameInputs* inputs);
SSBL_API int ssbl_step_solver(void* handle, int substeps, int iterations);
SSBL_API int ssbl_step_solver_ex(void* handle, int substeps, int iterations, int fetch_diagnostics, int force_sync);
SSBL_API int ssbl_download_positions(void* handle, float* out_positions, int max_floats);
SSBL_API int ssbl_get_diagnostics(void* handle, SsblXpbdDiagnostics* out_diag);
SSBL_API unsigned int ssbl_capabilities(void);
SSBL_API const char* ssbl_last_error(void);

#ifdef __cplusplus
}
#endif
