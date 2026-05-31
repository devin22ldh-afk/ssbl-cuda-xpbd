import json
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
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=7, y_subdivisions=7, size=1.0, location=(0.0, 0.0, 1.0))
    obj = bpy.context.object
    obj.name = "SSBL_Topology_Guard"
    group = obj.vertex_groups.new(name="ssbl_pin")
    group.add([vert.index for vert in obj.data.vertices if vert.co.y > 0.45], 1.0, "ADD")
    modifier = obj.modifiers.new("Subd", "SUBSURF")
    modifier.levels = 1

    settings = bpy.context.scene.ssbl_preview
    settings.use_evaluated_mesh = True
    settings.pin_vertex_group = "ssbl_pin"

    error_text = ""
    try:
        ssbl.solver.start_preview(bpy.context, obj)
    except Exception as exc:
        error_text = str(exc)
    else:
        raise RuntimeError("Expected topology-changing evaluated cloth input to be rejected")

    print(
        "SSBL_EVALUATED_CLOTH_GUARD",
        json.dumps(
            {
                "error": error_text,
                "matched": "same vertex count" in error_text.lower() or "fixed topology" in error_text.lower(),
            },
            ensure_ascii=False,
        ),
    )
    ssbl.unregister()


if __name__ == "__main__":
    main()
