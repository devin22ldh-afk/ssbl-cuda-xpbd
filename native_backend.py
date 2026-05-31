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


@dataclass
class NativeStatus:
    available: bool
    message: str
    dll_path: str


_LIB = None
_LOAD_ERROR = ""


def dll_path() -> str:
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, "native", "bin", "ssbl_xpbd_cuda_abi17.dll")


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
    lib.ssbl_step_solver.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.ssbl_step_solver.restype = ctypes.c_int
    lib.ssbl_download_positions.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.ssbl_download_positions.restype = ctypes.c_int
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

    def step(self, substeps: int, iterations: int) -> None:
        if not self._lib.ssbl_step_solver(self._handle, int(substeps), int(iterations)):
            raise NativeSolverError(_last_error(self._lib))

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

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
