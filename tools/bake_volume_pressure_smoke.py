import json
import os
import struct
import sys

import bpy


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

import ssbl


def main():
    try:
        ssbl.unregister()
    except Exception:
        pass
    ssbl.register()

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.0, location=(0.0, 0.0, 2.0))
    obj = bpy.context.object
    obj.name = "SSBL_Bake_Volume_Sphere"

    settings = bpy.context.scene.ssbl_preview
    settings.use_volume_pressure = True
    settings.volume_compliance = 1e-6
    settings.pressure_strength = 1.0
    settings.volume_target_scale = 1.0
    settings.self_collision_mode = "fast"
    settings.self_collision_interval = 2
    settings.max_self_collision_neighbors = 32
    settings.use_ground = False
    settings.substeps = 8
    settings.iterations = 2
    settings.bake_start = 1
    settings.bake_end = 30
    settings.pin_vertex_group = ""

    path = ssbl.solver.bake_xpbd_cache(bpy.context, obj)
    modifier = obj.modifiers.get("SSBL XPBD Cache")
    with open(path, "rb") as handle:
        signature, version, vertex_count, start, sample_rate, sample_count = struct.unpack("<12siiffi", handle.read(32))
    exists_before_clear = os.path.exists(path)
    cleared = ssbl.solver.clear_xpbd_cache(obj)
    exists_after_clear = os.path.exists(path)
    print(
        "SSBL_BAKE_VOLUME_SMOKE",
        json.dumps(
            {
                "path": path,
                "signature": signature.decode("ascii", errors="replace").rstrip("\0"),
                "version": version,
                "vertex_count": vertex_count,
                "object_vertex_count": len(obj.data.vertices),
                "start": start,
                "sample_rate": sample_rate,
                "sample_count": sample_count,
                "modifier_cache_format": getattr(modifier, "cache_format", None) if modifier else None,
                "modifier_frame_start": getattr(modifier, "frame_start", None) if modifier else None,
                "exists_before_clear": exists_before_clear,
                "cleared": cleared,
                "exists_after_clear": exists_after_clear,
            },
            ensure_ascii=False,
        ),
    )
    ssbl.unregister()


if __name__ == "__main__":
    main()
