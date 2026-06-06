from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import numpy as np

from .xpbd_core import ClothBuildData, SolverOptions


class NativeBackendUnavailable(RuntimeError):
    pass


class NativeSolverError(RuntimeError):
    pass


class _NativeConfig(ctypes.Structure):
    _fields_ = [
        ("vertex_count", ctypes.c_int),
        ("edge_count", ctypes.c_int),
        ("bend_count", ctypes.c_int),
        ("lra_count", ctypes.c_int),
        ("triangle_count", ctypes.c_int),
        ("static_triangle_count", ctypes.c_int),
        ("edge_color_count", ctypes.c_int),
        ("bend_color_count", ctypes.c_int),
        ("lra_color_count", ctypes.c_int),
        ("dt", ctypes.c_float),
        ("damping", ctypes.c_float),
        ("gravity", ctypes.c_float * 3),
        ("stretch_compliance", ctypes.c_float),
        ("bend_compliance", ctypes.c_float),
        ("lra_compliance", ctypes.c_float),
        ("collision_margin", ctypes.c_float),
        ("use_ground", ctypes.c_int),
        ("ground_height", ctypes.c_float),
        ("use_wall", ctypes.c_int),
        ("wall_origin", ctypes.c_float * 3),
        ("wall_normal", ctypes.c_float * 3),
        ("use_sphere", ctypes.c_int),
        ("sphere_center", ctypes.c_float * 3),
        ("sphere_radius", ctypes.c_float),
        ("self_collision", ctypes.c_int),
        ("self_collision_mode", ctypes.c_int),
        ("cloth_thickness", ctypes.c_float),
        ("self_collision_interval", ctypes.c_int),
        ("max_self_collision_neighbors", ctypes.c_int),
        ("use_volume_pressure", ctypes.c_int),
        ("rest_volume", ctypes.c_float),
        ("volume_compliance", ctypes.c_float),
        ("pressure_strength", ctypes.c_float),
        ("volume_target_scale", ctypes.c_float),
        ("volume_solve_interval", ctypes.c_int),
        ("self_probe_interval", ctypes.c_int),
        ("self_surface_pair_interval", ctypes.c_int),
        ("jitter_stabilizer_enabled", ctypes.c_int),
        ("contact_friction", ctypes.c_float),
        ("contact_tangent_damping", ctypes.c_float),
        ("contact_compliance", ctypes.c_float),
        ("fast_self_collision_passes", ctypes.c_int),
        ("stretch_optimization_enabled", ctypes.c_int),
        ("stretch_optimization_strength", ctypes.c_float),
        ("static_sdf_voxel_size", ctypes.c_float),
        ("static_sdf_band_voxels", ctypes.c_int),
        ("static_sdf_max_resolution", ctypes.c_int),
    ]


class _NativeMesh(ctypes.Structure):
    _fields_ = [
        ("positions", ctypes.POINTER(ctypes.c_float)),
        ("inv_mass", ctypes.POINTER(ctypes.c_float)),
        ("edges", ctypes.POINTER(ctypes.c_int)),
        ("edge_rest_lengths", ctypes.POINTER(ctypes.c_float)),
        ("edge_color_offsets", ctypes.POINTER(ctypes.c_int)),
        ("bends", ctypes.POINTER(ctypes.c_int)),
        ("bend_rest_lengths", ctypes.POINTER(ctypes.c_float)),
        ("bend_color_offsets", ctypes.POINTER(ctypes.c_int)),
        ("lra_edges", ctypes.POINTER(ctypes.c_int)),
        ("lra_rest_lengths", ctypes.POINTER(ctypes.c_float)),
        ("lra_color_offsets", ctypes.POINTER(ctypes.c_int)),
        ("triangles", ctypes.POINTER(ctypes.c_int)),
        ("static_triangles", ctypes.POINTER(ctypes.c_float)),
    ]


class _NativeRuntimeColliders(ctypes.Structure):
    _fields_ = [
        ("use_ground", ctypes.c_int),
        ("ground_height", ctypes.c_float),
        ("use_wall", ctypes.c_int),
        ("wall_origin", ctypes.c_float * 3),
        ("wall_normal", ctypes.c_float * 3),
        ("use_sphere", ctypes.c_int),
        ("sphere_center", ctypes.c_float * 3),
        ("sphere_radius", ctypes.c_float),
    ]


class _NativeForceField(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("use_min_distance", ctypes.c_int),
        ("use_max_distance", ctypes.c_int),
        ("seed", ctypes.c_int),
        ("strength", ctypes.c_float),
        ("origin", ctypes.c_float * 3),
        ("direction", ctypes.c_float * 3),
        ("axis", ctypes.c_float * 3),
        ("falloff_power", ctypes.c_float),
        ("distance_min", ctypes.c_float),
        ("distance_max", ctypes.c_float),
        ("radial_min", ctypes.c_float),
        ("radial_max", ctypes.c_float),
        ("noise", ctypes.c_float),
        ("linear_drag", ctypes.c_float),
        ("quadratic_drag", ctypes.c_float),
        ("harmonic_damping", ctypes.c_float),
        ("flow", ctypes.c_float),
        ("size", ctypes.c_float),
        ("rest_length", ctypes.c_float),
        ("radial_falloff", ctypes.c_float),
        ("texture_nabla", ctypes.c_float),
        ("use_radial_min", ctypes.c_int),
        ("use_radial_max", ctypes.c_int),
        ("use_2d_force", ctypes.c_int),
    ]


class _NativeFrameInputs(ctypes.Structure):
    _fields_ = [
        ("update_pin_targets", ctypes.c_int),
        ("pin_indices", ctypes.POINTER(ctypes.c_int)),
        ("pin_positions", ctypes.POINTER(ctypes.c_float)),
        ("pin_weights", ctypes.POINTER(ctypes.c_float)),
        ("pin_count", ctypes.c_int),
        ("update_runtime_colliders", ctypes.c_int),
        ("runtime_colliders", _NativeRuntimeColliders),
        ("update_static_triangles", ctypes.c_int),
        ("static_triangles", ctypes.POINTER(ctypes.c_float)),
        ("static_triangle_count", ctypes.c_int),
        ("update_dynamic_triangles", ctypes.c_int),
        ("dynamic_triangles", ctypes.POINTER(ctypes.c_float)),
        ("dynamic_triangle_count", ctypes.c_int),
        ("update_dynamic_particles", ctypes.c_int),
        ("dynamic_particle_positions", ctypes.POINTER(ctypes.c_float)),
        ("dynamic_particle_radii", ctypes.POINTER(ctypes.c_float)),
        ("dynamic_particle_inv_mass", ctypes.POINTER(ctypes.c_float)),
        ("dynamic_particle_slot_ids", ctypes.POINTER(ctypes.c_int)),
        ("dynamic_particle_phases", ctypes.POINTER(ctypes.c_int)),
        ("dynamic_particle_count", ctypes.c_int),
        ("update_force_fields", ctypes.c_int),
        ("force_fields", ctypes.POINTER(_NativeForceField)),
        ("force_field_count", ctypes.c_int),
        ("unsupported_force_field_count", ctypes.c_int),
    ]


class _NativeDiagnostics(ctypes.Structure):
    _fields_ = [
        ("step_ms", ctypes.c_float),
        ("hash_build_ms", ctypes.c_float),
        ("constraints_ms", ctypes.c_float),
        ("volume_ms", ctypes.c_float),
        ("analytic_collision_ms", ctypes.c_float),
        ("static_collision_ms", ctypes.c_float),
        ("dynamic_collision_ms", ctypes.c_float),
        ("dynamic_particle_collision_ms", ctypes.c_float),
        ("self_hash_ms", ctypes.c_float),
        ("self_solve_ms", ctypes.c_float),
        ("self_probe_ms", ctypes.c_float),
        ("self_recovery_ms", ctypes.c_float),
        ("sync_ms", ctypes.c_float),
        ("diagnostics_fetch_ms", ctypes.c_float),
        ("candidate_count", ctypes.c_longlong),
        ("resolved_contacts", ctypes.c_longlong),
        ("min_gap", ctypes.c_float),
        ("ccd_clamp_count", ctypes.c_longlong),
        ("recovery_passes", ctypes.c_longlong),
        ("local_retry_count", ctypes.c_longlong),
        ("jitter_stabilized_vertices", ctypes.c_longlong),
        ("jitter_rejected_vertices", ctypes.c_longlong),
        ("jitter_max_correction", ctypes.c_float),
        ("external_contact_cache_hits", ctypes.c_longlong),
        ("external_contact_cache_misses", ctypes.c_longlong),
        ("external_contact_cache_count", ctypes.c_longlong),
        ("external_contact_cache_overflow", ctypes.c_longlong),
        ("external_friction_corrections", ctypes.c_longlong),
        ("force_field_count", ctypes.c_longlong),
        ("unsupported_force_field_count", ctypes.c_longlong),
        ("dynamic_particle_count", ctypes.c_longlong),
        ("dynamic_particle_candidate_count", ctypes.c_longlong),
        ("dynamic_particle_contacts", ctypes.c_longlong),
        ("dynamic_particle_overflow", ctypes.c_longlong),
        ("dynamic_triangle_count", ctypes.c_longlong),
        ("static_triangle_count", ctypes.c_longlong),
        ("finite_flag", ctypes.c_int),
        ("fast_exact_vt_candidates", ctypes.c_longlong),
        ("fast_exact_vt_projected", ctypes.c_longlong),
        ("fast_exact_vt_guarded", ctypes.c_longlong),
        ("fast_exact_vt_skipped_rest", ctypes.c_longlong),
        ("fast_soft_repulsion_candidates", ctypes.c_longlong),
        ("fast_soft_repulsion_applied", ctypes.c_longlong),
        ("fast_soft_repulsion_max_push", ctypes.c_float),
        ("fast_hard_projection_count", ctypes.c_longlong),
        ("fast_manifold_contacts", ctypes.c_longlong),
        ("fast_manifold_reused", ctypes.c_longlong),
        ("fast_barrier_projected", ctypes.c_longlong),
        ("fast_barrier_smoothed_vertices", ctypes.c_longlong),
        ("fast_barrier_overflow", ctypes.c_longlong),
        ("fast_barrier_max_delta", ctypes.c_float),
        ("fast_edge_edge_candidates", ctypes.c_longlong),
        ("fast_edge_edge_contacts", ctypes.c_longlong),
        ("fast_triangle_pair_candidates", ctypes.c_longlong),
        ("fast_triangle_pair_contacts", ctypes.c_longlong),
        ("fast_triangle_pair_skipped_rest", ctypes.c_longlong),
        ("fast_contact_classification_guarded", ctypes.c_longlong),
        ("fast_region_cluster_candidates", ctypes.c_longlong),
        ("fast_region_cluster_contacts", ctypes.c_longlong),
        ("fast_region_cluster_guarded", ctypes.c_longlong),
        ("fast_overlap_island_candidates", ctypes.c_longlong),
        ("fast_overlap_island_clusters", ctypes.c_longlong),
        ("fast_overlap_island_vertex_refs", ctypes.c_longlong),
        ("fast_overlap_island_applied_vertices", ctypes.c_longlong),
        ("fast_overlap_island_guarded", ctypes.c_longlong),
        ("fast_overlap_island_max_delta", ctypes.c_float),
        ("fast_cc_overlap_components", ctypes.c_longlong),
        ("fast_cc_overlap_seed_triangles", ctypes.c_longlong),
        ("fast_cc_overlap_owned_vertices", ctypes.c_longlong),
        ("fast_cc_overlap_union_edges", ctypes.c_longlong),
        ("fast_cc_overlap_guarded", ctypes.c_longlong),
        ("fast_cc_overlap_applied_vertices", ctypes.c_longlong),
        ("fast_cc_overlap_max_delta", ctypes.c_float),
        ("abi41_soft_contact_count", ctypes.c_longlong),
        ("abi41_exact_impulse_contact_count", ctypes.c_longlong),
        ("abi41_edge_edge_contact_count", ctypes.c_longlong),
        ("abi41_max_smoothed_delta", ctypes.c_float),
        ("abi41_hard_projection_fallbacks", ctypes.c_longlong),
        ("static_sdf_rebuild_count", ctypes.c_longlong),
        ("static_sdf_voxel_count", ctypes.c_longlong),
        ("static_sdf_grid_x", ctypes.c_longlong),
        ("static_sdf_grid_y", ctypes.c_longlong),
        ("static_sdf_grid_z", ctypes.c_longlong),
        ("static_sdf_build_ms", ctypes.c_float),
        ("static_sdf_contact_count", ctypes.c_longlong),
        ("static_sdf_unsigned_fallback_count", ctypes.c_longlong),
        ("abi41_pcg_iterations", ctypes.c_longlong),
        ("abi41_pcg_guarded", ctypes.c_longlong),
        ("abi41_pcg_csr_nnz", ctypes.c_longlong),
        ("abi41_pcg_texture_ready", ctypes.c_longlong),
        ("abi41_pcg_initial_residual", ctypes.c_float),
        ("abi41_pcg_final_residual", ctypes.c_float),
        ("abi41_pcg_max_delta", ctypes.c_float),
        ("abi41_lra_tack_count", ctypes.c_longlong),
        ("abi41_bending_wing_count", ctypes.c_longlong),
        ("abi41_bending_texture_ready", ctypes.c_longlong),
        ("abi41_tack_jitter_guarded", ctypes.c_longlong),
        ("abi41_bending_guarded", ctypes.c_longlong),
        ("dynamic_collider_pack_ms", ctypes.c_float),
        ("dynamic_triangle_upload_ms", ctypes.c_float),
        ("dynamic_particle_upload_ms", ctypes.c_float),
        ("dynamic_collider_cache_hits", ctypes.c_longlong),
        ("dynamic_collider_cache_misses", ctypes.c_longlong),
        ("abi41_pcg_solve_ms", ctypes.c_float),
        ("abi41_pcg_system_ms", ctypes.c_float),
        ("abi41_pcg_ad_ms", ctypes.c_float),
        ("abi41_direct_stretch_ms", ctypes.c_float),
        ("dynamic_pair_cache_hits", ctypes.c_longlong),
        ("dynamic_pair_cache_misses", ctypes.c_longlong),
        ("dynamic_pair_cache_reused_triangles", ctypes.c_longlong),
        ("dynamic_pair_cache_reused_particles", ctypes.c_longlong),
        ("dynamic_collision_skipped_launches", ctypes.c_longlong),
        ("self_collision_skipped_launches", ctypes.c_longlong),
        ("self_candidate_count", ctypes.c_longlong),
    ]


@dataclass
class NativeStatus:
    available: bool
    message: str
    dll_path: str


@dataclass(frozen=True)
class NativeStepDiagnostics:
    step_ms: float = 0.0
    hash_build_ms: float = 0.0
    constraints_ms: float = 0.0
    volume_ms: float = 0.0
    analytic_collision_ms: float = 0.0
    static_collision_ms: float = 0.0
    dynamic_collision_ms: float = 0.0
    dynamic_particle_collision_ms: float = 0.0
    self_hash_ms: float = 0.0
    self_solve_ms: float = 0.0
    self_probe_ms: float = 0.0
    self_recovery_ms: float = 0.0
    sync_ms: float = 0.0
    diagnostics_fetch_ms: float = 0.0
    candidate_count: int = 0
    resolved_contacts: int = 0
    min_gap: float | None = None
    ccd_clamp_count: int = 0
    recovery_passes: int = 0
    local_retry_count: int = 0
    jitter_stabilized_vertices: int = 0
    jitter_rejected_vertices: int = 0
    jitter_max_correction: float = 0.0
    external_contact_cache_hits: int = 0
    external_contact_cache_misses: int = 0
    external_contact_cache_count: int = 0
    external_contact_cache_overflow: int = 0
    external_friction_corrections: int = 0
    force_field_count: int = 0
    unsupported_force_field_count: int = 0
    dynamic_particle_count: int = 0
    dynamic_particle_candidate_count: int = 0
    dynamic_particle_contacts: int = 0
    dynamic_particle_overflow: int = 0
    dynamic_triangle_count: int = 0
    static_triangle_count: int = 0
    finite: bool = True
    fast_exact_vt_candidates: int = 0
    fast_exact_vt_projected: int = 0
    fast_exact_vt_guarded: int = 0
    fast_exact_vt_skipped_rest: int = 0
    fast_soft_repulsion_candidates: int = 0
    fast_soft_repulsion_applied: int = 0
    fast_soft_repulsion_max_push: float = 0.0
    fast_hard_projection_count: int = 0
    fast_manifold_contacts: int = 0
    fast_manifold_reused: int = 0
    fast_barrier_projected: int = 0
    fast_barrier_smoothed_vertices: int = 0
    fast_barrier_overflow: int = 0
    fast_barrier_max_delta: float = 0.0
    fast_edge_edge_candidates: int = 0
    fast_edge_edge_contacts: int = 0
    fast_triangle_pair_candidates: int = 0
    fast_triangle_pair_contacts: int = 0
    fast_triangle_pair_skipped_rest: int = 0
    fast_contact_classification_guarded: int = 0
    fast_region_cluster_candidates: int = 0
    fast_region_cluster_contacts: int = 0
    fast_region_cluster_guarded: int = 0
    fast_overlap_island_candidates: int = 0
    fast_overlap_island_clusters: int = 0
    fast_overlap_island_vertex_refs: int = 0
    fast_overlap_island_applied_vertices: int = 0
    fast_overlap_island_guarded: int = 0
    fast_overlap_island_max_delta: float = 0.0
    fast_cc_overlap_components: int = 0
    fast_cc_overlap_seed_triangles: int = 0
    fast_cc_overlap_owned_vertices: int = 0
    fast_cc_overlap_union_edges: int = 0
    fast_cc_overlap_guarded: int = 0
    fast_cc_overlap_applied_vertices: int = 0
    fast_cc_overlap_max_delta: float = 0.0
    abi41_soft_contact_count: int = 0
    abi41_exact_impulse_contact_count: int = 0
    abi41_edge_edge_contact_count: int = 0
    abi41_max_smoothed_delta: float = 0.0
    abi41_hard_projection_fallbacks: int = 0
    static_sdf_rebuild_count: int = 0
    static_sdf_voxel_count: int = 0
    static_sdf_grid_x: int = 0
    static_sdf_grid_y: int = 0
    static_sdf_grid_z: int = 0
    static_sdf_build_ms: float = 0.0
    static_sdf_contact_count: int = 0
    static_sdf_unsigned_fallback_count: int = 0
    abi41_pcg_iterations: int = 0
    abi41_pcg_guarded: int = 0
    abi41_pcg_csr_nnz: int = 0
    abi41_pcg_texture_ready: int = 0
    abi41_pcg_initial_residual: float = 0.0
    abi41_pcg_final_residual: float = 0.0
    abi41_pcg_max_delta: float = 0.0
    abi41_lra_tack_count: int = 0
    abi41_bending_wing_count: int = 0
    abi41_bending_texture_ready: int = 0
    abi41_tack_jitter_guarded: int = 0
    abi41_bending_guarded: int = 0
    dynamic_collider_pack_ms: float = 0.0
    dynamic_triangle_upload_ms: float = 0.0
    dynamic_particle_upload_ms: float = 0.0
    dynamic_collider_cache_hits: int = 0
    dynamic_collider_cache_misses: int = 0
    abi41_pcg_solve_ms: float = 0.0
    abi41_pcg_system_ms: float = 0.0
    abi41_pcg_ad_ms: float = 0.0
    abi41_direct_stretch_ms: float = 0.0
    dynamic_pair_cache_hits: int = 0
    dynamic_pair_cache_misses: int = 0
    dynamic_pair_cache_reused_triangles: int = 0
    dynamic_pair_cache_reused_particles: int = 0
    dynamic_collision_skipped_launches: int = 0
    self_collision_skipped_launches: int = 0
    self_candidate_count: int = 0
    frame_ms: float = 0.0
    frame_set_ms: float = 0.0
    input_refresh_ms: float = 0.0
    pin_upload_ms: float = 0.0
    runtime_upload_ms: float = 0.0
    static_upload_ms: float = 0.0
    dynamic_upload_ms: float = 0.0
    cuda_step_call_ms: float = 0.0
    download_ms: float = 0.0
    writeback_ms: float = 0.0
    writeback_to_local_ms: float = 0.0
    writeback_foreach_set_ms: float = 0.0
    writeback_mesh_update_ms: float = 0.0
    frame_input_upload_ms: float = 0.0
    writeback_performed: bool = False
    diagnostics_ms: float = 0.0
    viewport_tag_ms: float = 0.0

    @property
    def penetration_depth(self) -> float:
        if self.min_gap is None:
            return 0.0
        return max(0.0, -float(self.min_gap))


_LIB = None
_LOAD_ERROR = ""
_ABI41_DLL_NAME = "ssbl_xpbd_cuda_abi40.dll"
_CAP_STRETCH_OPTIMIZATION = 1 << 0
_CAP_PIN_WEIGHTS = 1 << 1


def dll_path() -> str:
    override = os.environ.get("SSBL_NATIVE_DLL_PATH")
    if override:
        return override
    root = os.path.dirname(os.path.abspath(__file__))
    native_bin = os.path.join(root, "native", "bin")
    return os.path.join(native_bin, _ABI41_DLL_NAME)


def status() -> NativeStatus:
    path = dll_path()
    if not os.path.exists(path):
        return NativeStatus(False, f"Missing CUDA solver DLL: {path}", path)
    try:
        _load_library()
    except NativeBackendUnavailable as exc:
        return NativeStatus(False, str(exc), path)
    return NativeStatus(True, "CUDA solver DLL loaded.", path)


def _load_library():
    global _LIB, _LOAD_ERROR
    if _LIB is not None:
        return _LIB

    path = dll_path()
    if not os.path.exists(path):
        raise NativeBackendUnavailable(
            "Missing ABI40 CUDA solver DLL. Install CUDA Toolkit, CMake, and VS Build Tools, then run native/build_recon.ps1."
        )

    try:
        if os.name == "nt":
            os.add_dll_directory(os.path.dirname(path))
        lib = ctypes.WinDLL(path) if os.name == "nt" else ctypes.CDLL(path)
    except OSError as exc:
        _LOAD_ERROR = str(exc)
        raise NativeBackendUnavailable(f"Unable to load CUDA solver DLL: {exc}") from exc

    lib.ssbl_create_solver.argtypes = [ctypes.POINTER(_NativeConfig), ctypes.POINTER(_NativeMesh)]
    lib.ssbl_create_solver.restype = ctypes.c_void_p
    lib.ssbl_destroy_solver.argtypes = [ctypes.c_void_p]
    lib.ssbl_destroy_solver.restype = ctypes.c_int
    lib.ssbl_reset_solver.argtypes = [ctypes.c_void_p]
    lib.ssbl_reset_solver.restype = ctypes.c_int
    lib.ssbl_update_pin_targets.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.ssbl_update_pin_targets.restype = ctypes.c_int
    lib.ssbl_update_runtime_colliders.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NativeRuntimeColliders)]
    lib.ssbl_update_runtime_colliders.restype = ctypes.c_int
    if hasattr(lib, "ssbl_update_positions"):
        lib.ssbl_update_positions.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        lib.ssbl_update_positions.restype = ctypes.c_int
    lib.ssbl_update_static_triangles.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.ssbl_update_static_triangles.restype = ctypes.c_int
    lib.ssbl_update_dynamic_triangles.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    lib.ssbl_update_dynamic_triangles.restype = ctypes.c_int
    if hasattr(lib, "ssbl_update_frame_inputs"):
        lib.ssbl_update_frame_inputs.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NativeFrameInputs)]
        lib.ssbl_update_frame_inputs.restype = ctypes.c_int
    lib.ssbl_step_solver.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.ssbl_step_solver.restype = ctypes.c_int
    lib.ssbl_step_solver_ex.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.ssbl_step_solver_ex.restype = ctypes.c_int
    lib.ssbl_download_positions.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.ssbl_download_positions.restype = ctypes.c_int
    lib.ssbl_get_diagnostics.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NativeDiagnostics)]
    lib.ssbl_get_diagnostics.restype = ctypes.c_int
    if hasattr(lib, "ssbl_capabilities"):
        lib.ssbl_capabilities.argtypes = []
        lib.ssbl_capabilities.restype = ctypes.c_uint
    lib.ssbl_last_error.argtypes = []
    lib.ssbl_last_error.restype = ctypes.c_char_p
    _LIB = lib
    return lib


def _as_float_ptr(arr: np.ndarray):
    if arr.size == 0:
        return ctypes.POINTER(ctypes.c_float)()
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _as_int_ptr(arr: np.ndarray):
    if arr.size == 0:
        return ctypes.POINTER(ctypes.c_int)()
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def _last_error(lib) -> str:
    raw = lib.ssbl_last_error()
    if not raw:
        return "Native CUDA solver failed without an error message."
    return raw.decode("utf-8", errors="replace")


def _capabilities(lib) -> int:
    if hasattr(lib, "ssbl_capabilities"):
        return int(lib.ssbl_capabilities())
    return 0


def _config_from_options(
    cloth: ClothBuildData,
    options: SolverOptions,
    static_triangles: np.ndarray,
) -> _NativeConfig:
    cfg = _NativeConfig()
    cfg.vertex_count = int(len(cloth.positions_world))
    cfg.edge_count = int(len(cloth.edges))
    cfg.bend_count = int(len(cloth.bends))
    cfg.lra_count = int(len(cloth.lra_edges))
    cfg.triangle_count = int(len(cloth.triangles))
    cfg.static_triangle_count = int(len(static_triangles))
    cfg.edge_color_count = max(int(len(cloth.edge_color_offsets)) - 1, 0)
    cfg.bend_color_count = max(int(len(cloth.bend_color_offsets)) - 1, 0)
    cfg.lra_color_count = max(int(len(cloth.lra_color_offsets)) - 1, 0)
    cfg.dt = float(options.dt)
    cfg.damping = float(options.damping)
    cfg.gravity = (ctypes.c_float * 3)(*map(float, options.gravity))
    cfg.stretch_compliance = float(options.stretch_compliance)
    cfg.bend_compliance = float(options.bend_compliance)
    cfg.lra_compliance = float(options.lra_compliance)
    cfg.collision_margin = float(options.collision_margin)
    cfg.use_ground = int(options.use_ground)
    cfg.ground_height = float(options.ground_height)
    cfg.use_wall = int(options.use_wall)
    cfg.wall_origin = (ctypes.c_float * 3)(*map(float, options.wall_origin))
    cfg.wall_normal = (ctypes.c_float * 3)(*map(float, options.wall_normal))
    cfg.use_sphere = int(options.use_sphere)
    cfg.sphere_center = (ctypes.c_float * 3)(*map(float, options.sphere_center))
    cfg.sphere_radius = float(options.sphere_radius)
    cfg.self_collision = int(options.self_collision)
    cfg.self_collision_mode = int(options.self_collision_mode)
    cfg.cloth_thickness = float(getattr(options, 'cloth_thickness', 0.02))
    cfg.self_collision_interval = int(options.self_collision_interval)
    cfg.max_self_collision_neighbors = int(options.max_self_collision_neighbors)
    cfg.use_volume_pressure = int(options.use_volume_pressure)
    cfg.rest_volume = float(cloth.rest_volume)
    cfg.volume_compliance = float(options.volume_compliance)
    cfg.pressure_strength = float(options.pressure_strength)
    cfg.volume_target_scale = float(options.volume_target_scale)
    cfg.volume_solve_interval = int(options.volume_solve_interval)
    cfg.self_probe_interval = int(options.self_probe_interval)
    cfg.self_surface_pair_interval = int(options.self_surface_pair_interval)
    cfg.jitter_stabilizer_enabled = int(options.jitter_stabilizer_enabled)
    cfg.contact_friction = float(getattr(options, "contact_friction", 0.35))
    cfg.contact_tangent_damping = float(getattr(options, "contact_tangent_damping", 0.2))
    cfg.contact_compliance = float(getattr(options, "contact_compliance", 0.0))
    cfg.fast_self_collision_passes = int(getattr(options, "fast_self_collision_passes", 4))
    cfg.stretch_optimization_enabled = int(getattr(options, "stretch_optimization_enabled", False))
    cfg.stretch_optimization_strength = float(getattr(options, "stretch_optimization_strength", 0.35))
    cfg.static_sdf_voxel_size = float(getattr(options, "static_sdf_voxel_size", 0.0))
    cfg.static_sdf_band_voxels = int(getattr(options, "static_sdf_band_voxels", 4))
    cfg.static_sdf_max_resolution = int(getattr(options, "static_sdf_max_resolution", 160))
    return cfg


def _runtime_colliders_from_options(options: SolverOptions) -> _NativeRuntimeColliders:
    inputs = _NativeRuntimeColliders()
    inputs.use_ground = int(options.use_ground)
    inputs.ground_height = float(options.ground_height)
    inputs.use_wall = int(options.use_wall)
    inputs.wall_origin = (ctypes.c_float * 3)(*map(float, options.wall_origin))
    inputs.wall_normal = (ctypes.c_float * 3)(*map(float, options.wall_normal))
    inputs.use_sphere = int(options.use_sphere)
    inputs.sphere_center = (ctypes.c_float * 3)(*map(float, options.sphere_center))
    inputs.sphere_radius = float(options.sphere_radius)
    return inputs


def _force_field_array(force_fields):
    samples = tuple(getattr(force_fields, "fields", ()) or ()) if force_fields is not None else ()
    array_type = _NativeForceField * len(samples)
    array = array_type()
    for index, sample in enumerate(samples):
        item = array[index]
        item.type = int(getattr(sample, "field_type", 0))
        item.use_min_distance = int(getattr(sample, "use_min_distance", 0))
        item.use_max_distance = int(getattr(sample, "use_max_distance", 0))
        item.seed = int(getattr(sample, "seed", 0))
        item.strength = float(getattr(sample, "strength", 0.0))
        item.origin = (ctypes.c_float * 3)(*map(float, getattr(sample, "origin", (0.0, 0.0, 0.0))))
        item.direction = (ctypes.c_float * 3)(*map(float, getattr(sample, "direction", (0.0, 0.0, 1.0))))
        item.axis = (ctypes.c_float * 3)(*map(float, getattr(sample, "axis", (0.0, 0.0, 1.0))))
        item.falloff_power = float(getattr(sample, "falloff_power", 0.0))
        item.distance_min = float(getattr(sample, "distance_min", 0.0))
        item.distance_max = float(getattr(sample, "distance_max", 0.0))
        item.radial_min = float(getattr(sample, "radial_min", 0.0))
        item.radial_max = float(getattr(sample, "radial_max", 0.0))
        item.noise = float(getattr(sample, "noise", 0.0))
        item.linear_drag = float(getattr(sample, "linear_drag", 0.0))
        item.quadratic_drag = float(getattr(sample, "quadratic_drag", 0.0))
        item.harmonic_damping = float(getattr(sample, "harmonic_damping", 0.0))
        item.flow = float(getattr(sample, "flow", 0.0))
        item.size = float(getattr(sample, "size", 0.0))
        item.rest_length = float(getattr(sample, "rest_length", 0.0))
        item.radial_falloff = float(getattr(sample, "radial_falloff", 0.0))
        item.texture_nabla = float(getattr(sample, "texture_nabla", 0.0))
        item.use_radial_min = int(getattr(sample, "use_radial_min", 0))
        item.use_radial_max = int(getattr(sample, "use_radial_max", 0))
        item.use_2d_force = int(getattr(sample, "use_2d_force", 0))
    return array


def _as_force_field_ptr(arr):
    if len(arr) == 0:
        return ctypes.POINTER(_NativeForceField)()
    return ctypes.cast(arr, ctypes.POINTER(_NativeForceField))


class NativeXpbdSolver:
    def __init__(self, cloth: ClothBuildData, options: SolverOptions, static_triangles: np.ndarray):
        self._lib = _load_library()
        self._is_abi41_abi40 = "abi40" in os.path.basename(dll_path()).lower()
        if bool(getattr(options, "stretch_optimization_enabled", False)):
            if (_capabilities(self._lib) & _CAP_STRETCH_OPTIMIZATION) == 0:
                raise NativeSolverError(
                    "Loaded CUDA solver DLL does not support hard stretch optimization. "
                    "Rebuild native/build_recon.ps1 to generate the ABI40 DLL."
                )
        if (_capabilities(self._lib) & _CAP_PIN_WEIGHTS) == 0:
            raise NativeSolverError(
                "Loaded CUDA solver DLL does not support vertex-group pin weights. "
                "Rebuild native/build_recon.ps1 to generate the ABI40 DLL."
            )
        self._vertex_count = int(len(cloth.positions_world))
        self._positions_out = np.empty((self._vertex_count, 3), dtype=np.float32)
        self._static_triangle_count = int(len(static_triangles))
        self._dynamic_triangle_count: int | None = None
        self._last_diagnostics = NativeStepDiagnostics()
        static_flat = np.ascontiguousarray(static_triangles.reshape((-1, 3)), dtype=np.float32)
        cfg = _config_from_options(cloth, options, static_triangles)
        self._runtime_colliders = _runtime_colliders_from_options(options)
        mesh = _NativeMesh(
            positions=_as_float_ptr(cloth.positions_world),
            inv_mass=_as_float_ptr(cloth.inv_mass),
            edges=_as_int_ptr(cloth.edges),
            edge_rest_lengths=_as_float_ptr(cloth.edge_rest_lengths),
            edge_color_offsets=_as_int_ptr(cloth.edge_color_offsets),
            bends=_as_int_ptr(cloth.bends),
            bend_rest_lengths=_as_float_ptr(cloth.bend_rest_lengths),
            bend_color_offsets=_as_int_ptr(cloth.bend_color_offsets),
            lra_edges=_as_int_ptr(cloth.lra_edges),
            lra_rest_lengths=_as_float_ptr(cloth.lra_rest_lengths),
            lra_color_offsets=_as_int_ptr(cloth.lra_color_offsets),
            triangles=_as_int_ptr(cloth.triangles),
            static_triangles=_as_float_ptr(static_flat),
        )
        self._handle = self._lib.ssbl_create_solver(ctypes.byref(cfg), ctypes.byref(mesh))
        if not self._handle:
            raise NativeSolverError(_last_error(self._lib))
        if len(cloth.pin_indices) > 0:
            self.update_pin_targets(cloth.pin_indices, cloth.pin_targets_world, cloth.pin_weights)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.ssbl_destroy_solver(self._handle)
            self._handle = None

    def reset(self) -> None:
        if not self._lib.ssbl_reset_solver(self._handle):
            raise NativeSolverError(_last_error(self._lib))

    def update_pin_targets(
        self,
        indices: np.ndarray,
        positions: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        indices = np.ascontiguousarray(indices, dtype=np.int32)
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        if weights is None:
            weights = np.ones(len(indices), dtype=np.float32)
        weights = np.ascontiguousarray(weights, dtype=np.float32).reshape((-1,))
        if len(indices) != len(positions) or len(indices) != len(weights):
            raise NativeSolverError("Pin index, target, and weight arrays must have matching lengths.")
        ok = self._lib.ssbl_update_pin_targets(
            self._handle,
            _as_int_ptr(indices),
            _as_float_ptr(positions),
            _as_float_ptr(weights),
            int(len(indices)),
        )
        if not ok:
            raise NativeSolverError(_last_error(self._lib))

    def update_runtime_colliders(self, options: SolverOptions) -> None:
        self._runtime_colliders = _runtime_colliders_from_options(options)
        if not self._lib.ssbl_update_runtime_colliders(self._handle, ctypes.byref(self._runtime_colliders)):
            raise NativeSolverError(_last_error(self._lib))

    def update_positions(self, positions: np.ndarray) -> None:
        if not hasattr(self._lib, "ssbl_update_positions"):
            raise NativeSolverError("Loaded CUDA solver DLL does not support runtime position upload.")
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        if positions.shape != (self._vertex_count, 3):
            raise NativeSolverError(
                f"Position upload expected shape ({self._vertex_count}, 3), got {tuple(positions.shape)}."
            )
        if not self._lib.ssbl_update_positions(self._handle, _as_float_ptr(positions), int(positions.size)):
            raise NativeSolverError(_last_error(self._lib))

    def update_static_triangles(self, static_triangles: np.ndarray) -> None:
        triangle_count = int(len(static_triangles))
        if triangle_count <= 0 and self._static_triangle_count <= 0:
            return
        static_triangles = np.ascontiguousarray(static_triangles.reshape((-1, 3)), dtype=np.float32)
        if not self._lib.ssbl_update_static_triangles(
            self._handle,
            _as_float_ptr(static_triangles),
            triangle_count,
        ):
            raise NativeSolverError(_last_error(self._lib))
        self._static_triangle_count = triangle_count

    def update_dynamic_triangles(self, dynamic_triangles: np.ndarray) -> None:
        triangle_count = int(len(dynamic_triangles))
        dynamic_triangles = np.ascontiguousarray(dynamic_triangles.reshape((-1, 3)), dtype=np.float32)
        if not self._lib.ssbl_update_dynamic_triangles(
            self._handle,
            _as_float_ptr(dynamic_triangles),
            triangle_count,
        ):
            raise NativeSolverError(_last_error(self._lib))
        self._dynamic_triangle_count = triangle_count

    def update_frame_inputs(
        self,
        *,
        pin_indices: np.ndarray | None,
        pin_positions: np.ndarray | None,
        pin_weights: np.ndarray | None,
        update_pin: bool,
        options: SolverOptions | None,
        update_runtime: bool,
        static_triangles: np.ndarray | None,
        update_static: bool,
        dynamic_triangles: np.ndarray | None,
        update_dynamic: bool,
        dynamic_particles: dict[str, np.ndarray] | None = None,
        update_dynamic_particles: bool = False,
        force_fields=None,
        update_force_fields: bool = False,
    ) -> None:
        if not hasattr(self._lib, "ssbl_update_frame_inputs"):
            if update_force_fields and force_fields is not None and len(getattr(force_fields, "fields", ()) or ()) > 0:
                raise NativeSolverError("Loaded CUDA solver DLL does not support Blender force fields.")
            if update_pin:
                self.update_pin_targets(
                    np.asarray(pin_indices if pin_indices is not None else [], dtype=np.int32),
                    np.asarray(pin_positions if pin_positions is not None else np.empty((0, 3), dtype=np.float32), dtype=np.float32),
                    None if pin_weights is None else np.asarray(pin_weights, dtype=np.float32),
                )
            if update_runtime and options is not None:
                self.update_runtime_colliders(options)
            if update_static and static_triangles is not None:
                self.update_static_triangles(static_triangles)
            if update_dynamic:
                dyn = dynamic_triangles if dynamic_triangles is not None else np.empty((0, 3, 3), dtype=np.float32)
                self.update_dynamic_triangles(dyn)
            return

        pin_indices_arr = np.ascontiguousarray(pin_indices if pin_indices is not None else np.empty(0, dtype=np.int32), dtype=np.int32)
        pin_positions_arr = np.ascontiguousarray(
            pin_positions if pin_positions is not None else np.empty((0, 3), dtype=np.float32),
            dtype=np.float32,
        )
        if pin_weights is None:
            pin_weights_arr = np.ones(len(pin_indices_arr), dtype=np.float32)
        else:
            pin_weights_arr = np.ascontiguousarray(pin_weights, dtype=np.float32).reshape((-1,))
        if len(pin_indices_arr) != len(pin_positions_arr) or len(pin_indices_arr) != len(pin_weights_arr):
            raise NativeSolverError("Pin index, target, and weight arrays must have matching lengths.")
        static_triangle_count = int(len(static_triangles)) if static_triangles is not None else 0
        dynamic_triangle_count = int(len(dynamic_triangles)) if dynamic_triangles is not None else 0
        static_arr = np.ascontiguousarray(
            static_triangles.reshape((-1, 3)) if static_triangles is not None else np.empty((0, 3), dtype=np.float32),
            dtype=np.float32,
        )
        dynamic_arr = np.ascontiguousarray(
            dynamic_triangles.reshape((-1, 3)) if dynamic_triangles is not None else np.empty((0, 3), dtype=np.float32),
            dtype=np.float32,
        )
        particle_positions = np.ascontiguousarray(
            (dynamic_particles or {}).get("positions", np.empty((0, 3), dtype=np.float32)).reshape((-1, 3)),
            dtype=np.float32,
        )
        dynamic_particle_count = int(len(particle_positions))
        particle_radii = np.ascontiguousarray(
            (dynamic_particles or {}).get("radii", np.empty(0, dtype=np.float32)),
            dtype=np.float32,
        ).reshape((-1,))
        needs_particle_extras = dynamic_particle_count > 0 and not bool(getattr(self, "_is_abi41_abi40", False))
        if needs_particle_extras:
            particle_inv_mass = np.ascontiguousarray(
                (dynamic_particles or {}).get("inv_mass", np.ones(dynamic_particle_count, dtype=np.float32)),
                dtype=np.float32,
            ).reshape((-1,))
            particle_slot_ids = np.ascontiguousarray(
                (dynamic_particles or {}).get("slot_ids", np.ones(dynamic_particle_count, dtype=np.int32)),
                dtype=np.int32,
            ).reshape((-1,))
            particle_phases = np.ascontiguousarray(
                (dynamic_particles or {}).get("phases", np.ones(dynamic_particle_count, dtype=np.int32)),
                dtype=np.int32,
            ).reshape((-1,))
        else:
            particle_inv_mass = np.empty(0, dtype=np.float32)
            particle_slot_ids = np.empty(0, dtype=np.int32)
            particle_phases = np.empty(0, dtype=np.int32)
        if dynamic_particle_count <= 0:
            particle_radii = np.empty(0, dtype=np.float32)
            particle_inv_mass = np.empty(0, dtype=np.float32)
            particle_slot_ids = np.empty(0, dtype=np.int32)
            particle_phases = np.empty(0, dtype=np.int32)
        elif not (
            len(particle_radii) == dynamic_particle_count
            and (not needs_particle_extras or len(particle_inv_mass) == dynamic_particle_count)
            and (not needs_particle_extras or len(particle_slot_ids) == dynamic_particle_count)
            and (not needs_particle_extras or len(particle_phases) == dynamic_particle_count)
        ):
            raise NativeSolverError("Dynamic particle arrays must have matching lengths.")
        force_field_arr = _force_field_array(force_fields)
        force_field_count = int(len(force_field_arr))
        unsupported_force_field_count = int(getattr(force_fields, "unsupported_count", 0) if force_fields is not None else 0)
        runtime_inputs = _runtime_colliders_from_options(options) if options is not None else self._runtime_colliders
        frame_inputs = _NativeFrameInputs(
            update_pin_targets=int(update_pin),
            pin_indices=_as_int_ptr(pin_indices_arr),
            pin_positions=_as_float_ptr(pin_positions_arr),
            pin_weights=_as_float_ptr(pin_weights_arr),
            pin_count=int(len(pin_indices_arr)),
            update_runtime_colliders=int(update_runtime),
            runtime_colliders=runtime_inputs,
            update_static_triangles=int(update_static),
            static_triangles=_as_float_ptr(static_arr),
            static_triangle_count=static_triangle_count,
            update_dynamic_triangles=int(update_dynamic),
            dynamic_triangles=_as_float_ptr(dynamic_arr),
            dynamic_triangle_count=dynamic_triangle_count,
            update_dynamic_particles=int(update_dynamic_particles),
            dynamic_particle_positions=_as_float_ptr(particle_positions),
            dynamic_particle_radii=_as_float_ptr(particle_radii),
            dynamic_particle_inv_mass=_as_float_ptr(particle_inv_mass),
            dynamic_particle_slot_ids=_as_int_ptr(particle_slot_ids),
            dynamic_particle_phases=_as_int_ptr(particle_phases),
            dynamic_particle_count=dynamic_particle_count,
            update_force_fields=int(update_force_fields),
            force_fields=_as_force_field_ptr(force_field_arr),
            force_field_count=force_field_count,
            unsupported_force_field_count=unsupported_force_field_count,
        )
        ok = self._lib.ssbl_update_frame_inputs(self._handle, ctypes.byref(frame_inputs))
        if not ok:
            raise NativeSolverError(_last_error(self._lib))
        if update_runtime:
            self._runtime_colliders = runtime_inputs
        if update_static:
            self._static_triangle_count = static_triangle_count
        if update_dynamic:
            self._dynamic_triangle_count = dynamic_triangle_count

    def step(self, substeps: int, iterations: int, diagnostics: bool = True, synchronize: bool = True) -> None:
        fetch_diagnostics = 1 if diagnostics else 0
        force_sync = 1 if (synchronize or diagnostics) else 0
        if not self._lib.ssbl_step_solver_ex(
            self._handle,
            int(substeps),
            int(iterations),
            fetch_diagnostics,
            force_sync,
        ):
            raise NativeSolverError(_last_error(self._lib))
        if diagnostics:
            self._last_diagnostics = self.diagnostics()

    def cached_diagnostics(self) -> NativeStepDiagnostics:
        return self._last_diagnostics

    def download_positions(self) -> np.ndarray:
        out = self._positions_out
        ok = self._lib.ssbl_download_positions(
            self._handle,
            _as_float_ptr(out),
            int(out.size),
        )
        if not ok:
            raise NativeSolverError(_last_error(self._lib))
        return out

    def diagnostics(self) -> NativeStepDiagnostics:
        if not getattr(self, "_handle", None):
            return self._last_diagnostics
        raw = _NativeDiagnostics()
        if not self._lib.ssbl_get_diagnostics(self._handle, ctypes.byref(raw)):
            raise NativeSolverError(_last_error(self._lib))
        min_gap = float(raw.min_gap)
        if not np.isfinite(min_gap) or min_gap >= 1.0e29:
            min_gap = None
        diag = NativeStepDiagnostics(
            step_ms=float(raw.step_ms),
            hash_build_ms=float(raw.hash_build_ms),
            constraints_ms=float(raw.constraints_ms),
            volume_ms=float(raw.volume_ms),
            analytic_collision_ms=float(raw.analytic_collision_ms),
            static_collision_ms=float(raw.static_collision_ms),
            dynamic_collision_ms=float(raw.dynamic_collision_ms),
            dynamic_particle_collision_ms=float(raw.dynamic_particle_collision_ms),
            self_hash_ms=float(raw.self_hash_ms),
            self_solve_ms=float(raw.self_solve_ms),
            self_probe_ms=float(raw.self_probe_ms),
            self_recovery_ms=float(raw.self_recovery_ms),
            sync_ms=float(raw.sync_ms),
            diagnostics_fetch_ms=float(raw.diagnostics_fetch_ms),
            candidate_count=int(raw.candidate_count),
            resolved_contacts=int(raw.resolved_contacts),
            min_gap=min_gap,
            ccd_clamp_count=int(raw.ccd_clamp_count),
            recovery_passes=int(raw.recovery_passes),
            local_retry_count=int(raw.local_retry_count),
            jitter_stabilized_vertices=int(raw.jitter_stabilized_vertices),
            jitter_rejected_vertices=int(raw.jitter_rejected_vertices),
            jitter_max_correction=float(raw.jitter_max_correction),
            external_contact_cache_hits=int(raw.external_contact_cache_hits),
            external_contact_cache_misses=int(raw.external_contact_cache_misses),
            external_contact_cache_count=int(raw.external_contact_cache_count),
            external_contact_cache_overflow=int(raw.external_contact_cache_overflow),
            external_friction_corrections=int(raw.external_friction_corrections),
            force_field_count=int(raw.force_field_count),
            unsupported_force_field_count=int(raw.unsupported_force_field_count),
            dynamic_particle_count=int(raw.dynamic_particle_count),
            dynamic_particle_candidate_count=int(raw.dynamic_particle_candidate_count),
            dynamic_particle_contacts=int(raw.dynamic_particle_contacts),
            dynamic_particle_overflow=int(raw.dynamic_particle_overflow),
            dynamic_triangle_count=int(raw.dynamic_triangle_count),
            static_triangle_count=int(raw.static_triangle_count),
            finite=bool(raw.finite_flag),
            fast_exact_vt_candidates=int(raw.fast_exact_vt_candidates),
            fast_exact_vt_projected=int(raw.fast_exact_vt_projected),
            fast_exact_vt_guarded=int(raw.fast_exact_vt_guarded),
            fast_exact_vt_skipped_rest=int(raw.fast_exact_vt_skipped_rest),
            fast_soft_repulsion_candidates=int(raw.fast_soft_repulsion_candidates),
            fast_soft_repulsion_applied=int(raw.fast_soft_repulsion_applied),
            fast_soft_repulsion_max_push=float(raw.fast_soft_repulsion_max_push),
            fast_hard_projection_count=int(raw.fast_hard_projection_count),
            fast_manifold_contacts=int(raw.fast_manifold_contacts),
            fast_manifold_reused=int(raw.fast_manifold_reused),
            fast_barrier_projected=int(raw.fast_barrier_projected),
            fast_barrier_smoothed_vertices=int(raw.fast_barrier_smoothed_vertices),
            fast_barrier_overflow=int(raw.fast_barrier_overflow),
            fast_barrier_max_delta=float(raw.fast_barrier_max_delta),
            fast_edge_edge_candidates=int(raw.fast_edge_edge_candidates),
            fast_edge_edge_contacts=int(raw.fast_edge_edge_contacts),
            fast_triangle_pair_candidates=int(raw.fast_triangle_pair_candidates),
            fast_triangle_pair_contacts=int(raw.fast_triangle_pair_contacts),
            fast_triangle_pair_skipped_rest=int(raw.fast_triangle_pair_skipped_rest),
            fast_contact_classification_guarded=int(raw.fast_contact_classification_guarded),
            fast_region_cluster_candidates=int(raw.fast_region_cluster_candidates),
            fast_region_cluster_contacts=int(raw.fast_region_cluster_contacts),
            fast_region_cluster_guarded=int(raw.fast_region_cluster_guarded),
            fast_overlap_island_candidates=int(raw.fast_overlap_island_candidates),
            fast_overlap_island_clusters=int(raw.fast_overlap_island_clusters),
            fast_overlap_island_vertex_refs=int(raw.fast_overlap_island_vertex_refs),
            fast_overlap_island_applied_vertices=int(raw.fast_overlap_island_applied_vertices),
            fast_overlap_island_guarded=int(raw.fast_overlap_island_guarded),
            fast_overlap_island_max_delta=float(raw.fast_overlap_island_max_delta),
            fast_cc_overlap_components=int(raw.fast_cc_overlap_components),
            fast_cc_overlap_seed_triangles=int(raw.fast_cc_overlap_seed_triangles),
            fast_cc_overlap_owned_vertices=int(raw.fast_cc_overlap_owned_vertices),
            fast_cc_overlap_union_edges=int(raw.fast_cc_overlap_union_edges),
            fast_cc_overlap_guarded=int(raw.fast_cc_overlap_guarded),
            fast_cc_overlap_applied_vertices=int(raw.fast_cc_overlap_applied_vertices),
            fast_cc_overlap_max_delta=float(raw.fast_cc_overlap_max_delta),
            abi41_soft_contact_count=int(raw.abi41_soft_contact_count),
            abi41_exact_impulse_contact_count=int(raw.abi41_exact_impulse_contact_count),
            abi41_edge_edge_contact_count=int(raw.abi41_edge_edge_contact_count),
            abi41_max_smoothed_delta=float(raw.abi41_max_smoothed_delta),
            abi41_hard_projection_fallbacks=int(raw.abi41_hard_projection_fallbacks),
            static_sdf_rebuild_count=int(raw.static_sdf_rebuild_count),
            static_sdf_voxel_count=int(raw.static_sdf_voxel_count),
            static_sdf_grid_x=int(raw.static_sdf_grid_x),
            static_sdf_grid_y=int(raw.static_sdf_grid_y),
            static_sdf_grid_z=int(raw.static_sdf_grid_z),
            static_sdf_build_ms=float(raw.static_sdf_build_ms),
            static_sdf_contact_count=int(raw.static_sdf_contact_count),
            static_sdf_unsigned_fallback_count=int(raw.static_sdf_unsigned_fallback_count),
            abi41_pcg_iterations=int(raw.abi41_pcg_iterations),
            abi41_pcg_guarded=int(raw.abi41_pcg_guarded),
            abi41_pcg_csr_nnz=int(raw.abi41_pcg_csr_nnz),
            abi41_pcg_texture_ready=int(raw.abi41_pcg_texture_ready),
            abi41_pcg_initial_residual=float(raw.abi41_pcg_initial_residual),
            abi41_pcg_final_residual=float(raw.abi41_pcg_final_residual),
            abi41_pcg_max_delta=float(raw.abi41_pcg_max_delta),
            abi41_lra_tack_count=int(raw.abi41_lra_tack_count),
            abi41_bending_wing_count=int(raw.abi41_bending_wing_count),
            abi41_bending_texture_ready=int(raw.abi41_bending_texture_ready),
            abi41_tack_jitter_guarded=int(raw.abi41_tack_jitter_guarded),
            abi41_bending_guarded=int(raw.abi41_bending_guarded),
            dynamic_collider_pack_ms=float(raw.dynamic_collider_pack_ms),
            dynamic_triangle_upload_ms=float(raw.dynamic_triangle_upload_ms),
            dynamic_particle_upload_ms=float(raw.dynamic_particle_upload_ms),
            dynamic_collider_cache_hits=int(raw.dynamic_collider_cache_hits),
            dynamic_collider_cache_misses=int(raw.dynamic_collider_cache_misses),
            abi41_pcg_solve_ms=float(raw.abi41_pcg_solve_ms),
            abi41_pcg_system_ms=float(raw.abi41_pcg_system_ms),
            abi41_pcg_ad_ms=float(raw.abi41_pcg_ad_ms),
            abi41_direct_stretch_ms=float(raw.abi41_direct_stretch_ms),
            dynamic_pair_cache_hits=int(raw.dynamic_pair_cache_hits),
            dynamic_pair_cache_misses=int(raw.dynamic_pair_cache_misses),
            dynamic_pair_cache_reused_triangles=int(raw.dynamic_pair_cache_reused_triangles),
            dynamic_pair_cache_reused_particles=int(raw.dynamic_pair_cache_reused_particles),
            dynamic_collision_skipped_launches=int(raw.dynamic_collision_skipped_launches),
            self_collision_skipped_launches=int(raw.self_collision_skipped_launches),
            self_candidate_count=int(raw.self_candidate_count),
        )
        self._last_diagnostics = diag
        return diag

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

