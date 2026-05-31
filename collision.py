from __future__ import annotations

import bmesh
import bpy
import numpy as np

from .xpbd_core import to_world


def collect_static_triangles(
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
