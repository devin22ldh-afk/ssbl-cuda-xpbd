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


class _NativeFrameInputs(ctypes.Structure):
    _fields_ = [
        ("update_pin_targets", ctypes.c_int),
        ("pin_indices", ctypes.POINTER(ctypes.c_int)),
        ("pin_positions", ctypes.POINTER(ctypes.c_float)),
        ("pin_count", ctypes.c_int),
        ("update_runtime_colliders", ctypes.c_int),
        ("runtime_colliders", _NativeRuntimeColliders),
        ("update_static_triangles", ctypes.c_int),
        ("static_triangles", ctypes.POINTER(ctypes.c_float)),
        ("static_triangle_count", ctypes.c_int),
        ("update_dynamic_triangles", ctypes.c_int),
        ("dynamic_triangles", ctypes.POINTER(ctypes.c_float)),
        ("dynamic_triangle_count", ctypes.c_int),
    ]


class _NativeDiagnostics(ctypes.Structure):
    _fields_ = [
        ("step_ms", ctypes.c_float),
        ("hash_build_ms", ctypes.c_float),
        ("constraints_ms", ctypes.c_float),
        ("volume_ms", ctypes.c_float),
        ("static_collision_ms", ctypes.c_float),
        ("dynamic_collision_ms", ctypes.c_float),
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
        ("finite_flag", ctypes.c_int),
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
    static_collision_ms: float = 0.0
    dynamic_collision_ms: float = 0.0
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
    finite: bool = True
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


def dll_path() -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, "native", "bin", "ssbl_xpbd_cuda_abi25.dll")


def status() -> NativeStatus:
    path = dll_path()
    if not os.path.exists(path):
        return NativeStatus(False, f"缺少 CUDA 解算 DLL：{path}", path)
    try:
        _load_library()
    except NativeBackendUnavailable as exc:
        return NativeStatus(False, str(exc), path)
    return NativeStatus(True, "CUDA 解算 DLL 已加载", path)


def _load_library():
    global _LIB, _LOAD_ERROR
    if _LIB is not None:
        return _LIB

    path = dll_path()
    if not os.path.exists(path):
        raise NativeBackendUnavailable(
            "缺少 CUDA 解算 DLL。请先安装 CUDA Toolkit、CMake 和 VS Build Tools，然后运行 native/build.ps1 进行构建。"
        )

    try:
        if os.name == "nt":
            os.add_dll_directory(os.path.dirname(path))
        lib = ctypes.WinDLL(path) if os.name == "nt" else ctypes.CDLL(path)
    except OSError as exc:
        _LOAD_ERROR = str(exc)
        raise NativeBackendUnavailable(f"无法加载 CUDA 解算 DLL：{exc}") from exc

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
        ctypes.c_int,
    ]
    lib.ssbl_update_pin_targets.restype = ctypes.c_int
    lib.ssbl_update_runtime_colliders.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NativeRuntimeColliders)]
    lib.ssbl_update_runtime_colliders.restype = ctypes.c_int
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
        return "原生 CUDA 解算器执行失败，但没有返回错误信息"
    return raw.decode("utf-8", errors="replace")


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


class NativeXpbdSolver:
    def __init__(self, cloth: ClothBuildData, options: SolverOptions, static_triangles: np.ndarray):
        self._lib = _load_library()
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

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.ssbl_destroy_solver(self._handle)
            self._handle = None

    def reset(self) -> None:
        if not self._lib.ssbl_reset_solver(self._handle):
            raise NativeSolverError(_last_error(self._lib))

    def update_pin_targets(self, indices: np.ndarray, positions: np.ndarray) -> None:
        indices = np.ascontiguousarray(indices, dtype=np.int32)
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        ok = self._lib.ssbl_update_pin_targets(
            self._handle,
            _as_int_ptr(indices),
            _as_float_ptr(positions),
            int(len(indices)),
        )
        if not ok:
            raise NativeSolverError(_last_error(self._lib))

    def update_runtime_colliders(self, options: SolverOptions) -> None:
        self._runtime_colliders = _runtime_colliders_from_options(options)
        if not self._lib.ssbl_update_runtime_colliders(self._handle, ctypes.byref(self._runtime_colliders)):
            raise NativeSolverError(_last_error(self._lib))

    def update_static_triangles(self, static_triangles: np.ndarray) -> None:
        triangle_count = int(len(static_triangles))
        if triangle_count <= 0 and self._static_triangle_count <= 0:
            return
        static_triangles = np.ascontiguousarray(static_triangles.reshape((-1, 3)), dtype=np.float32)
        if triangle_count != self._static_triangle_count:
            raise NativeSolverError(
                f"静态碰撞体拓扑发生变化：三角形数量从 {self._static_triangle_count} 变为 {triangle_count}。"
            )
        if not self._lib.ssbl_update_static_triangles(
            self._handle,
            _as_float_ptr(static_triangles),
            triangle_count,
        ):
            raise NativeSolverError(_last_error(self._lib))

    def update_dynamic_triangles(self, dynamic_triangles: np.ndarray) -> None:
        triangle_count = int(len(dynamic_triangles))
        if triangle_count <= 0 and (self._dynamic_triangle_count is None or self._dynamic_triangle_count <= 0):
            return
        dynamic_triangles = np.ascontiguousarray(dynamic_triangles.reshape((-1, 3)), dtype=np.float32)
        if self._dynamic_triangle_count is None and triangle_count > 0:
            self._dynamic_triangle_count = triangle_count
        elif triangle_count > 0 and self._dynamic_triangle_count != triangle_count:
            raise NativeSolverError(
                f"动态布料碰撞体拓扑发生变化：三角形数量从 {self._dynamic_triangle_count} 变为 {triangle_count}。"
            )
        if not self._lib.ssbl_update_dynamic_triangles(
            self._handle,
            _as_float_ptr(dynamic_triangles),
            triangle_count,
        ):
            raise NativeSolverError(_last_error(self._lib))

    def update_frame_inputs(
        self,
        *,
        pin_indices: np.ndarray | None,
        pin_positions: np.ndarray | None,
        update_pin: bool,
        options: SolverOptions | None,
        update_runtime: bool,
        static_triangles: np.ndarray | None,
        update_static: bool,
        dynamic_triangles: np.ndarray | None,
        update_dynamic: bool,
    ) -> None:
        if not hasattr(self._lib, "ssbl_update_frame_inputs"):
            if update_pin:
                self.update_pin_targets(
                    np.asarray(pin_indices if pin_indices is not None else [], dtype=np.int32),
                    np.asarray(pin_positions if pin_positions is not None else np.empty((0, 3), dtype=np.float32), dtype=np.float32),
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
        static_triangle_count = int(len(static_triangles)) if static_triangles is not None else 0
        dynamic_triangle_count = int(len(dynamic_triangles)) if dynamic_triangles is not None else 0
        if update_dynamic and dynamic_triangle_count > 0:
            if self._dynamic_triangle_count is None:
                self._dynamic_triangle_count = dynamic_triangle_count
            elif self._dynamic_triangle_count != dynamic_triangle_count:
                raise NativeSolverError(
                    f"鍔ㄦ€佸竷鏂欑鎾炰綋鎷撴墤鍙戠敓鍙樺寲锛氫笁瑙掑舰鏁伴噺浠?{self._dynamic_triangle_count} 鍙樹负 {dynamic_triangle_count}銆?"
                )
        static_arr = np.ascontiguousarray(
            static_triangles.reshape((-1, 3)) if static_triangles is not None else np.empty((0, 3), dtype=np.float32),
            dtype=np.float32,
        )
        dynamic_arr = np.ascontiguousarray(
            dynamic_triangles.reshape((-1, 3)) if dynamic_triangles is not None else np.empty((0, 3), dtype=np.float32),
            dtype=np.float32,
        )
        runtime_inputs = _runtime_colliders_from_options(options) if options is not None else self._runtime_colliders
        frame_inputs = _NativeFrameInputs(
            update_pin_targets=int(update_pin),
            pin_indices=_as_int_ptr(pin_indices_arr),
            pin_positions=_as_float_ptr(pin_positions_arr),
            pin_count=int(len(pin_indices_arr)),
            update_runtime_colliders=int(update_runtime),
            runtime_colliders=runtime_inputs,
            update_static_triangles=int(update_static),
            static_triangles=_as_float_ptr(static_arr),
            static_triangle_count=static_triangle_count,
            update_dynamic_triangles=int(update_dynamic),
            dynamic_triangles=_as_float_ptr(dynamic_arr),
            dynamic_triangle_count=dynamic_triangle_count,
        )
        ok = self._lib.ssbl_update_frame_inputs(self._handle, ctypes.byref(frame_inputs))
        if not ok:
            raise NativeSolverError(_last_error(self._lib))
        if update_runtime:
            self._runtime_colliders = runtime_inputs

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
            static_collision_ms=float(raw.static_collision_ms),
            dynamic_collision_ms=float(raw.dynamic_collision_ms),
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
            finite=bool(raw.finite_flag),
        )
        self._last_diagnostics = diag
        return diag

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
