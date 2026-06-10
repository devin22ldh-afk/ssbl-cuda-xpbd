# SSBL Parameters

This document covers the SSBL CUDA XPBD parameters visible in the Blender Physics panel. It is written for artists and technical users tuning cloth behavior, not for native ABI maintenance.

本文件说明 Blender Physics 面板中可见的 SSBL CUDA XPBD 参数，面向调节布料行为的美术和技术用户，不涉及 native ABI 维护细节。

Hidden compatibility fields, bake-progress internals, legacy LRA and volume fields, hidden multi-cloth switches, legacy wall and sphere fields, and object-level internal collision flags are intentionally excluded.

以下内容有意排除了隐藏兼容字段、烘焙进度内部字段、旧版 LRA 和体积字段、隐藏多布料开关、旧版墙体和球体字段，以及对象级内部碰撞标记。

## Reading Notes / 阅读说明

- Default values and ranges come from the current `SSBL_PreviewSettings` definitions. `soft_max` is Blender UI guidance, not a hard clamp.
- Some controls appear only when related features are enabled, such as volume pressure, self-collision, or static collider collections.
- Pin weights are multiplied by global `pin_hardness`. The native solver ignores effective pin weights below `0.05`, while `0.75` or higher usually behaves like a hard pin.

## Main Toggle / 主开关

| UI label (English / 中文) | Field | Default and range | Effect | Tuning notes |
| --- | --- | --- | --- | --- |
| `Enable SSBL Cloth Simulation` / `启用 SSBL 布料模拟` | `enabled` | `false` | Enables SSBL cloth preview and bake controls on the active Mesh object. | Turn this on before tuning. |
| `SSBL Cloth Simulation On` / `SSBL 布料解算中` | `enabled` | runtime state | Label shown while preview is enabled. | Indicates the object is participating in SSBL simulation. |

## Physics & Material / 物理与材质

### Base Properties / 基础属性

| UI label (English / 中文) | Field | Default and range | Effect | Tuning notes |
| --- | --- | --- | --- | --- |
| `Bend Stiffness` / `弯曲硬度` | `hardness` | `0.4`, `0.0..1.0` | Overall cloth stiffness control that also drives hidden stretch, bend, and tether settings. | Lower is softer, higher feels closer to heavy fabric or leather. |
| `Surface Thickness` / `表面厚度` | `cloth_thickness` | `0.02`, min `0.001`, soft max `0.1` | Thickness used by self-collision and cloth-surface spacing. | Increase if cloth intersects itself; too high can look inflated or detached. |
| `Material Density` / `材质密度` | `density` | `1.0`, min `1e-4`, soft max `100.0` | Affects vertex mass and inertia. | Higher feels heavier and resists forces; lower reacts more strongly to fields and collisions. |

### Spatial Constraints / 空间约束

| UI label (English / 中文) | Field | Default and range | Effect | Tuning notes |
| --- | --- | --- | --- | --- |
| `Pin Vertex Group` / `钉固顶点组` | `pin_vertex_group` | `"ssbl_pin"` | Vertex group used for fixed or softly constrained cloth vertices. | Leave empty to disable pinning by group. |
| `Pin Stiffness` / `钉固硬度` | `pin_hardness` | `1.0`, `0.0..1.0` | Global multiplier for pin vertex-group weights. | `0` disables pinning; `1` preserves the original weights. |

### Pressure & Volume / 充气与体积

| UI label (English / 中文) | Field | Default and range | Effect | Tuning notes |
| --- | --- | --- | --- | --- |
| `Enable Volume Pressure` / `启用体积气压` | `use_volume_pressure` | `false` | Adds outward pressure along cloth surface normals. | Enable for inflated cloth, cushions, or balloon-like effects. |
| `Internal Pressure` / `内压强度` | `pressure_strength` | `0.02`, min `0.0`, soft max `0.2` | Controls the strength of volume pressure. | High values on dense meshes can jitter or overinflate. |

## Environment & Collision / 环境与碰撞

### Scene Interaction / 场景交互

| UI label (English / 中文) | Field | Default and range | Effect | Tuning notes |
| --- | --- | --- | --- | --- |
| `Collision Margin` / `碰撞容差` | `collision_margin` | `0.005`, min `0.0`, soft max `0.2` | Minimum separation distance preserved during collision projection. | Increase if penetration is visible; too high can make cloth float above colliders. |
| `Infinite Ground Plane` / `无限平面底座` | `use_ground` | `true` | Enables an infinite world-space ground plane collider. | Good for quick drop tests and floor contact checks. |
| `Ground Height` / `平面高度` | `ground_height` | `0.0` | World Z height of the infinite ground plane. | Match it to the scene floor. |
| `Static Collider Collection` / `静态碰撞体集合` | `static_collider_collection` | none | Mesh collection used as static triangle or SDF colliders. | Use for non-deforming props, bodies, and environment pieces. |
| `Dynamic Collider Collection` / `动画碰撞体集合` | `dynamic_collider_collection` | none | Mesh collection used as one-way dynamic collision sources. | Complex colliders increase upload and collision cost. |
| `Voxel Size` / `体素尺寸` | `static_sdf_voxel_size` | `0.0`, min `0.0`, soft max `0.05` | Static SDF voxel size. `0` uses `max(collision_margin * 0.5, 0.002)`. | Smaller is more accurate but slower and heavier. |
| `Band Width` / `拓展带` | `static_sdf_band_voxels` | `4`, `1..32`, soft max `16` | Padding band around static SDF surfaces. | Increase for unstable or fast contacts; too high increases build cost. |
| `Max Resolution` / `最大分辨率` | `static_sdf_max_resolution` | `160`, `16..512`, soft max `256` | Maximum grid resolution per axis for static SDFs. | Raise for large complex colliders, lower for performance or memory pressure. |

### Self-Collision / 自碰撞

| UI label (English / 中文) | Field | Default and range | Effect | Tuning notes |
| --- | --- | --- | --- | --- |
| `Enable Self-Collision` / `启用自碰撞` | `self_collision` | `true` | Resolves contacts between the cloth surface and itself. | Recommended for skirts, capes, and folded cloth. |
| `Mode` / `算法模式` | `self_collision_mode` | `"fast"`, options `fast`, `strict` | Chooses the self-collision solve path. | `fast` targets realtime preview; `strict` is heavier and more conservative. |
| `Self-Collision Distance` / `自碰撞距离` | `self_collision_distance` | `0.0`, min `0.0`, soft max `0.1` | Overrides self-collision contact distance. `0` means automatic. | Keep `0` in most cases; raise it for thicker separation. |
| `Fast Passes` / `快速遍数` | `fast_self_collision_passes` | `4`, `1..8` | Native fast-mode self-collision pass count. | Increase for more intersections; lower it if preview becomes too slow. |
| `Max Neighbor Checks` / `最大邻居探测数` | `max_self_collision_neighbors` | `64`, min `4`, soft max `256` | Candidate neighbor budget per vertex or edge during self collision. | Increase for dense or heavily folded cloth. |

## Force Fields / 力场

### Effectors / 效果器

| UI label (English / 中文) | Field | Default and range | Effect | Tuning notes |
| --- | --- | --- | --- | --- |
| `Effector Collection` / `效果器集合` | `force_field_collection` | none | Blender force-field collection to sample. | Leave empty to read force fields from the current scene. |
| `Gravity` / `重力` | `force_field_weight_gravity` | `1.0`, `0.0..1.0` | Scales scene gravity for SSBL cloth. | Set to `0` to isolate non-gravity effects. |
| `All` / `全部` | `force_field_weight_all` | `1.0`, `0.0..1.0` | Scales all non-gravity force fields. | Use this first for a global boost or reduction. |
| `Force` / `常力` | `force_field_weight_force` | `1.0`, `0.0..1.0` | Scales Force effectors. | Useful for directional pushes and pulls. |
| `Vortex` / `涡流` | `force_field_weight_vortex` | `1.0`, `0.0..1.0` | Scales Vortex effectors. | Good for swirling and twisting motion. |
| `Magnetic` / `磁力` | `force_field_weight_magnetic` | `1.0`, `0.0..1.0` | Scales Magnetic effectors. | Adjust when using magnetic attraction or repulsion. |
| `Harmonic` / `谐振` | `force_field_weight_harmonic` | `1.0`, `0.0..1.0` | Scales Harmonic effectors. | Useful for elastic pulls and periodic recovery. |
| `Charge` / `电荷` | `force_field_weight_charge` | `1.0`, `0.0..1.0` | Scales Charge effectors. | Use for point-like attraction or repulsion. |
| `Lennard-Jones` / `兰纳琼斯分子力` | `force_field_weight_lennardjones` | `1.0`, `0.0..1.0` | Scales Lennard-Jones effectors. | Useful for short-range attraction or repulsion. |
| `Wind` / `风力` | `force_field_weight_wind` | `1.0`, `0.0..1.0` | Scales Wind effectors. | Start low when adding wind to cloth. |
| `Texture` / `纹理` | `force_field_weight_texture` | `1.0`, `0.0..1.0` | Scales Texture effectors. | Use for spatial noise or texture-driven disturbance. |
| `Turbulence` / `紊流` | `force_field_weight_turbulence` | `1.0`, `0.0..1.0` | Scales Turbulence effectors. | High values can destabilize the simulation. |
| `Drag` / `拖拽` | `force_field_weight_drag` | `1.0`, `0.0..1.0` | Scales Drag effectors. | Useful for slowdown or air-resistance feel. |

## Solver Tuning / 解算器调优

### Solver Core / 引擎核心

| UI label (English / 中文) | Field | Default and range | Effect | Tuning notes |
| --- | --- | --- | --- | --- |
| `Time Step` / `步长` | `dt` | `0.02`, min `1e-4`, soft max `0.1` | Simulation time represented by each output frame. | Smaller is more stable but advances less time per frame. |
| `Substeps` / `子步数` | `substeps` | `14`, min `1`, soft max `32` | XPBD substeps per frame. | Increase for penetration, jitter, or fast collisions. |
| `Constraint Iterations` / `约束迭代` | `iterations` | `2`, min `1`, soft max `128` | Constraint projection iterations per substep. | Raise it if constraints look too soft or under-converged. |
| `Global Damping` / `全局阻尼` | `damping` | `1.0`, `0.0..1.0` | Velocity damping applied each substep. | Lower values dissipate motion faster and can calm jitter. |

### Contact Dynamics / 接触动力学

| UI label (English / 中文) | Field | Default and range | Effect | Tuning notes |
| --- | --- | --- | --- | --- |
| `Surface Friction` / `表面摩擦力` | `contact_friction` | `0.35`, `0.0..4.0` | Friction coefficient for static and dynamic triangle contacts. | Increase if cloth slides too much; decrease if it sticks too heavily. |
| `Tangential Damping` / `切向滑动阻尼` | `contact_tangent_damping` | `0.2`, `0.0..1.0` | Tangential velocity damping at contacts. | Increase if cloth jitters or slides too quickly across surfaces. |
| `Contact Compliance` / `接触顺应性` | `contact_compliance` | `0.0`, min `0.0`, soft max `1e-4` | XPBD compliance for static and dynamic triangle contacts. | Default is the stiffest contact; raise slightly for softer response. |

### Advanced Self-Collision / 高级自碰撞设定

| UI label (English / 中文) | Field | Default and range | Effect | Tuning notes |
| --- | --- | --- | --- | --- |
| `Intersection Check Interval` / `求交判定间隔` | `self_collision_interval` | `1`, min `1`, soft max `8` | Runs self collision every N substeps. | Higher values save performance but can miss contacts. |
| `Broadphase Probe Interval` / `宽相 Probe 间隔` | `self_probe_interval` | `1`, min `1`, soft max `8` | Runs expensive probe and recovery every N self-collision passes. | Raise for speed, keep `1` for stricter recovery. |
| `Surface Pair Refresh Interval` / `表面配对刷新间隔` | `self_surface_pair_interval` | `1`, min `1`, soft max `8` | Refreshes sample-sample surface pairs every N passes. | Keep low for jitter-prone self collision. |

## Cache & Bake / 缓存与烘焙

### Preview Settings / 预览设定

| UI label (English / 中文) | Field | Default and range | Effect | Tuning notes |
| --- | --- | --- | --- | --- |
| `Target Preview FPS` / `目标预览帧率` | `preview_target_fps` | `30.0`, min `1.0`, soft max `120.0` | Target FPS for the viewport preview timer. | Very high values increase realtime pressure. |
| `Auto-Cache Real-Time Simulation` / `实时模拟自动缓存` | `auto_cache_realtime` | `false` | Writes a PC2 cache while realtime preview or timeline playback is running. | Enable when preview results need to be recorded during playback. |
| `Use Evaluated Mesh` / `使用修改器形态` | `use_evaluated_mesh` | `false` | Reads mesh input after Blender modifiers and bindings. | Vertex count and topology must remain unchanged. |

### Bake Workflow / 烘焙工作流

| UI label (English / 中文) | Field | Default and range | Effect | Tuning notes |
| --- | --- | --- | --- | --- |
| `Start Frame` / `起始帧` | `bake_start` | `1` | First frame written to PC2 bake output. | Match the scene timeline start. |
| `End Frame` / `结束帧` | `bake_end` | `120`, min `1` | Last frame written to PC2 bake output. | Long ranges increase bake time and file size. |
