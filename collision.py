from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib

import bmesh
import bpy
import numpy as np

from .xpbd_core import to_world


_STARTUP_CACHE_LIMIT = 16


@dataclass
class _StaticCollisionCacheEntry:
    triangles_world: np.ndarray
    signature: tuple[tuple[str, int, int], ...]


@dataclass
class _StaticSource:
    name: str
    mesh: bpy.types.Mesh
    local: np.ndarray
    matrix_world: object
    eval_obj: bpy.types.Object | None


_STATIC_COLLISION_CACHE: OrderedDict[tuple, _StaticCollisionCacheEntry] = OrderedDict()
_STATIC_COLLISION_CACHE_STATS = {
    "hits": 0,
    "misses": 0,
    "last_hit": False,
}


def clear_static_collision_cache() -> None:
    _STATIC_COLLISION_CACHE.clear()
    _STATIC_COLLISION_CACHE_STATS["hits"] = 0
    _STATIC_COLLISION_CACHE_STATS["misses"] = 0
    _STATIC_COLLISION_CACHE_STATS["last_hit"] = False


def static_collision_cache_stats() -> dict[str, int | bool]:
    return {
        "hits": int(_STATIC_COLLISION_CACHE_STATS["hits"]),
        "misses": int(_STATIC_COLLISION_CACHE_STATS["misses"]),
        "last_hit": bool(_STATIC_COLLISION_CACHE_STATS["last_hit"]),
        "size": int(len(_STATIC_COLLISION_CACHE)),
    }


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _mesh_loop_signature(mesh: bpy.types.Mesh) -> tuple[int, str]:
    loop_indices = np.empty(len(mesh.loops), dtype=np.int32)
    if len(loop_indices) > 0:
        mesh.loops.foreach_get("vertex_index", loop_indices)
    return len(loop_indices), _array_digest(loop_indices)


def _matrix_signature(matrix) -> tuple[float, ...]:
    return tuple(round(float(value), 6) for row in matrix for value in row)


def _source_signature(source: _StaticSource, original_obj: bpy.types.Object, use_evaluated_mesh: bool) -> tuple:
    return (
        original_obj.name,
        int(original_obj.as_pointer()),
        int(original_obj.data.as_pointer()),
        bool(use_evaluated_mesh),
        len(source.mesh.vertices),
        len(source.mesh.polygons),
        len(source.mesh.loops),
        _mesh_loop_signature(source.mesh),
        _matrix_signature(source.matrix_world),
        _array_digest(np.asarray(source.local, dtype=np.float32)),
    )


def _cache_key(
    collection: bpy.types.Collection,
    exclude_obj: bpy.types.Object | None,
    use_evaluated_mesh: bool,
    source_signatures: tuple,
) -> tuple:
    return (
        int(collection.as_pointer()),
        int(exclude_obj.as_pointer()) if exclude_obj is not None else 0,
        bool(use_evaluated_mesh),
        source_signatures,
    )


def _get_static_cache_entry(key: tuple) -> _StaticCollisionCacheEntry | None:
    entry = _STATIC_COLLISION_CACHE.get(key)
    if entry is None:
        _STATIC_COLLISION_CACHE_STATS["misses"] = int(_STATIC_COLLISION_CACHE_STATS["misses"]) + 1
        _STATIC_COLLISION_CACHE_STATS["last_hit"] = False
        return None
    _STATIC_COLLISION_CACHE.move_to_end(key)
    _STATIC_COLLISION_CACHE_STATS["hits"] = int(_STATIC_COLLISION_CACHE_STATS["hits"]) + 1
    _STATIC_COLLISION_CACHE_STATS["last_hit"] = True
    return entry


def _store_static_cache_entry(key: tuple, entry: _StaticCollisionCacheEntry) -> None:
    _STATIC_COLLISION_CACHE[key] = entry
    _STATIC_COLLISION_CACHE.move_to_end(key)
    while len(_STATIC_COLLISION_CACHE) > _STARTUP_CACHE_LIMIT:
        _STATIC_COLLISION_CACHE.popitem(last=False)


def _collect_static_triangles_uncached(
    collection: bpy.types.Collection | None,
    exclude_obj: bpy.types.Object | None,
    depsgraph: bpy.types.Depsgraph | None = None,
    use_evaluated_mesh: bool = False,
) -> tuple[np.ndarray, tuple[tuple[str, int, int], ...]]:
    if collection is None:
        return np.empty((0, 3, 3), dtype=np.float32), ()

    triangles_world: list[np.ndarray] = []
    signature_entries: list[tuple[str, int, int]] = []
    objects = sorted(collection.objects, key=lambda item: item.name)
    for obj in objects:
        if obj is None or obj == exclude_obj or obj.type != "MESH":
            continue
        local, faces, matrix_world = _mesh_input_data(obj, depsgraph, use_evaluated_mesh)
        if len(faces) == 0:
            continue
        signature_entries.append((obj.name, len(local), len(faces)))
        world, _mat = to_world(local, matrix_world)
        triangles_world.append(world[faces])

    if not triangles_world:
        return np.empty((0, 3, 3), dtype=np.float32), tuple(signature_entries)
    return np.ascontiguousarray(np.concatenate(triangles_world, axis=0), dtype=np.float32), tuple(signature_entries)


def collect_static_triangles(
    collection: bpy.types.Collection | None,
    exclude_obj: bpy.types.Object | None,
    depsgraph: bpy.types.Depsgraph | None = None,
    use_evaluated_mesh: bool = False,
) -> tuple[np.ndarray, tuple[tuple[str, int, int], ...]]:
    if collection is None:
        return np.empty((0, 3, 3), dtype=np.float32), ()

    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    sources: list[_StaticSource] = []
    source_signatures: list[tuple] = []
    try:
        objects = sorted(collection.objects, key=lambda item: item.name)
        for obj in objects:
            if obj is None or obj == exclude_obj or obj.type != "MESH":
                continue
            if use_evaluated_mesh:
                eval_obj = obj.evaluated_get(depsgraph)
                mesh = eval_obj.to_mesh()
                source = _StaticSource(
                    name=obj.name,
                    mesh=mesh,
                    local=_mesh_local_positions(mesh),
                    matrix_world=eval_obj.matrix_world.copy(),
                    eval_obj=eval_obj,
                )
            else:
                source = _StaticSource(
                    name=obj.name,
                    mesh=obj.data,
                    local=_mesh_local_positions(obj.data),
                    matrix_world=obj.matrix_world.copy(),
                    eval_obj=None,
                )
            sources.append(source)
            source_signatures.append(_source_signature(source, obj, use_evaluated_mesh))

        key = _cache_key(collection, exclude_obj, use_evaluated_mesh, tuple(source_signatures))
        cache_entry = _get_static_cache_entry(key)
        if cache_entry is not None:
            return (
                np.array(cache_entry.triangles_world, dtype=np.float32, copy=True),
                tuple(cache_entry.signature),
            )

        triangles_world: list[np.ndarray] = []
        signature_entries: list[tuple[str, int, int]] = []
        for source in sources:
            faces = _triangulated_faces(source.mesh)
            if len(faces) == 0:
                continue
            signature_entries.append((source.name, len(source.local), len(faces)))
            world, _mat = to_world(source.local, source.matrix_world)
            triangles_world.append(world[faces])

        if triangles_world:
            triangles = np.ascontiguousarray(np.concatenate(triangles_world, axis=0), dtype=np.float32)
        else:
            triangles = np.empty((0, 3, 3), dtype=np.float32)
        signature = tuple(signature_entries)
        _store_static_cache_entry(
            key,
            _StaticCollisionCacheEntry(
                triangles_world=np.ascontiguousarray(triangles, dtype=np.float32),
                signature=signature,
            ),
        )
        return np.array(triangles, dtype=np.float32, copy=True), signature
    finally:
        for source in sources:
            if source.eval_obj is not None:
                source.eval_obj.to_mesh_clear()


def _mesh_input_data(
    obj: bpy.types.Object,
    depsgraph: bpy.types.Depsgraph | None,
    use_evaluated_mesh: bool,
) -> tuple[np.ndarray, np.ndarray, bpy.types.Matrix]:
    if not use_evaluated_mesh:
        return _mesh_local_positions(obj.data), _triangulated_faces(obj.data), obj.matrix_world.copy()

    depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        return _mesh_local_positions(mesh), _triangulated_faces(mesh), eval_obj.matrix_world.copy()
    finally:
        eval_obj.to_mesh_clear()


def _mesh_local_positions(mesh_or_obj) -> np.ndarray:
    mesh = mesh_or_obj.data if hasattr(mesh_or_obj, "data") else mesh_or_obj
    coords = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    if "position" in mesh.attributes:
        mesh.attributes["position"].data.foreach_get("vector", coords)
    else:
        mesh.vertices.foreach_get("co", coords)
    return coords.reshape((-1, 3))


def _triangulated_faces(mesh: bpy.types.Mesh) -> np.ndarray:
    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.verts.index_update()
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
        bm.faces.ensure_lookup_table()
        triangles = [[vert.index for vert in face.verts] for face in bm.faces if len(face.verts) == 3]
        return np.asarray(triangles, dtype=np.int32).reshape((-1, 3))
    finally:
        bm.free()
