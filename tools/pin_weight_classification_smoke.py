from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import bpy
import numpy as np


ADDONS_ROOT = r"C:\Users\Administrator\AppData\Roaming\Blender Foundation\Blender\5.0\scripts\addons"
if ADDONS_ROOT not in sys.path:
    sys.path.insert(0, ADDONS_ROOT)

from ssbl.xpbd_core import PIN_HARD_WEIGHT_THRESHOLD, build_cloth_data


def _make_settings() -> SimpleNamespace:
    return SimpleNamespace(
        use_evaluated_mesh=False,
        pin_vertex_group="ssbl_pin",
        density=1.0,
        hardness=0.0,
        hardness_initialized=True,
        stretch_compliance=1.0e-6,
        bend_compliance=1.0e-4,
    )


def main() -> None:
    mesh = bpy.data.meshes.new("SSBL_PinWeightSmokeMesh")
    mesh.from_pydata(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ],
        [],
        [(0, 1, 2), (1, 3, 2)],
    )
    mesh.update()
    obj = bpy.data.objects.new("SSBL_PinWeightSmoke", mesh)
    bpy.context.collection.objects.link(obj)
    group = obj.vertex_groups.new(name="ssbl_pin")
    group.add([1], 0.375, "ADD")
    group.add([2], PIN_HARD_WEIGHT_THRESHOLD, "ADD")
    group.add([3], 1.0, "ADD")

    cloth = build_cloth_data(obj, _make_settings())
    pin_indices = [int(value) for value in cloth.pin_indices.tolist()]
    pin_weights = [round(float(value), 6) for value in cloth.pin_weights.tolist()]
    inv_mass = np.asarray(cloth.inv_mass, dtype=np.float32)

    expected_indices = [1, 2, 3]
    expected_weights = [0.375, 0.75, 1.0]
    if pin_indices != expected_indices:
        raise RuntimeError(f"Pin indices mismatch: {pin_indices}")
    if pin_weights != expected_weights:
        raise RuntimeError(f"Pin weights mismatch: {pin_weights}")
    if not (float(inv_mass[0]) > 0.0 and float(inv_mass[1]) > 0.0):
        raise RuntimeError(f"Unpinned/soft vertices should keep mass: {inv_mass.tolist()}")
    if not (float(inv_mass[2]) == 0.0 and float(inv_mass[3]) == 0.0):
        raise RuntimeError(f"Hard-pinned vertices should have zero inverse mass: {inv_mass.tolist()}")

    print(
        "PIN_WEIGHT_CLASSIFICATION_OK "
        + json.dumps(
            {
                "pin_indices": pin_indices,
                "pin_weights": pin_weights,
                "inv_mass": [round(float(value), 6) for value in inv_mass.tolist()],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
