# SSBL Parameters

This document covers the SSBL CUDA XPBD parameters that are visible in the Blender Physics panel. It is intended for artists and technical users tuning cloth behavior, not for native ABI maintenance.

Hidden compatibility fields, bake progress fields, legacy LRA/volume fields, hidden multi-cloth switches, legacy wall/sphere fields, and object-level internal collision flags are intentionally excluded.

## 中文版

### 阅读规则

- “默认值与范围”来自当前 `SSBL_PreviewSettings` 属性定义；`soft_max` 是 Blender UI 的软上限，不是绝对限制。
- 条件显示的参数会在“调参建议”中标出，例如仅在启用体积气压、自碰撞或静态碰撞体集合后显示。
- pin 权重会先乘以全局 `pin_hardness`；native 解算器会忽略低于 `0.05` 的有效 pin 权重，`0.75` 或更高通常接近硬 pin 行为。

### 启用

| 面板标签 | 字段名 | 默认值与范围 | 作用 | 调参建议 |
| --- | --- | --- | --- | --- |
| 启用 SSBL 布料模拟 / SSBL 布料解算中 | `enabled` | `false` | 在当前 Mesh 对象上启用 SSBL 布料预览和烘焙入口。 | 开始调参前先启用；关闭后对象不参与 SSBL 预览。 |

### 物理与材质

| 面板标签 | 字段名 | 默认值与范围 | 作用 | 调参建议 |
| --- | --- | --- | --- | --- |
| 弯曲硬度 | `hardness` | `0.4`, `0.0..1.0` | 综合控制布料软硬，并同步到底层拉伸、弯曲和隐藏 tether 设置。 | 低值更柔软，高值更像厚布或皮革；先用此参数做大方向调节。 |
| 表面厚度 | `cloth_thickness` | `0.02`, min `0.001`, soft max `0.1` | 设置自碰撞和布料表面间隔使用的厚度。 | 穿插明显时增大；过大可能让布料看起来膨胀或难以贴合。 |
| 材质密度 | `density` | `1.0`, min `1e-4`, soft max `100.0` | 影响顶点质量和惯性。 | 更大更沉、更抗外力；更小更轻、更容易被力场或碰撞推开。 |
| 钉固顶点组 | `pin_vertex_group` | `"ssbl_pin"` | 指定用于固定或软约束布料顶点的 Blender 顶点组。 | 顶点组权重决定局部 pin 强度；空名称等同于不使用 pin。 |
| 钉固硬度 | `pin_hardness` | `1.0`, `0.0..1.0` | 全局缩放 `pin_vertex_group` 的权重。 | `0` 关闭 pin，`1` 保留顶点组权重；需要整体放松边缘时降低它。 |
| 启用体积气压 | `use_volume_pressure` | `false` | 沿布料表面法线添加向外压力。 | 做膨胀布料、气囊或鼓起效果时启用；普通披挂布料通常关闭。 |
| 内压强度 | `pressure_strength` | `0.02`, min `0.0`, soft max `0.2` | 控制体积气压的强度。 | 仅在启用体积气压后显示；高值在高密网格上要谨慎，容易抖动或过度膨胀。 |

### 环境与碰撞

| 面板标签 | 字段名 | 默认值与范围 | 作用 | 调参建议 |
| --- | --- | --- | --- | --- |
| 碰撞容差 (Margin) | `collision_margin` | `0.005`, min `0.0`, soft max `0.2` | 碰撞投影时保留的最小分离距离。 | 穿透时适当增大；太大可能让布料悬浮在碰撞体外。 |
| 无限平面底座 | `use_ground` | `true` | 启用世界空间的无限地面平面碰撞。 | 地面接触测试和简单落布建议开启；复杂场景可改用静态碰撞体集合。 |
| 平面高度 | `ground_height` | `0.0` | 设置无限地面平面的世界 Z 高度。 | 仅在启用地面后显示；与场景地板高度保持一致。 |
| 静态碰撞体集合 | `static_collider_collection` | none | 指定作为静态三角形/SDF 碰撞体的 Mesh 集合。 | 用于不变形的桌面、身体、道具等；集合为空时不上传静态碰撞体。 |
| 动画碰撞体集合 | `dynamic_collider_collection` | none | 指定作为单向动态碰撞源的动画 Mesh 集合。 | 用于移动角色或动画障碍物；碰撞体越复杂，上传和碰撞成本越高。 |
| 体素尺寸 | `static_sdf_voxel_size` | `0.0`, min `0.0`, soft max `0.05` | 控制静态 SDF 的体素大小；`0` 自动使用 `max(collision_margin * 0.5, 0.002)`。 | 仅在指定静态碰撞体集合后显示；更小更精细但更慢、更占显存。 |
| 拓展带 (Band) | `static_sdf_band_voxels` | `4`, `1..32`, soft max `16` | 静态 SDF 表面外扩的体素带宽。 | 接触不稳定或快速移动时可增大；过大增加构建和采样成本。 |
| 最大分辨率 | `static_sdf_max_resolution` | `160`, `16..512`, soft max `256` | 限制静态 SDF 每个轴向的最大网格分辨率。 | 复杂大碰撞体可增大；性能优先或显存紧张时降低。 |

### 自碰撞

| 面板标签 | 字段名 | 默认值与范围 | 作用 | 调参建议 |
| --- | --- | --- | --- | --- |
| 启用自碰撞 | `self_collision` | `true` | 启用布料自身表面之间的碰撞处理。 | 容易自穿插的裙摆、披风建议开启；性能压力大时可关闭。 |
| 算法模式 | `self_collision_mode` | `"fast"`, options `fast`, `strict` | 选择自碰撞算法路径。 | `fast` 适合实时预览；`strict` 更重，适合需要更少相交的质量检查。 |
| 自碰撞距离 | `self_collision_distance` | `0.0`, min `0.0`, soft max `0.1` | 覆盖自碰撞接触距离；`0` 表示自动。 | 多数情况下保持 `0`；需要更厚的分离距离时手动提高。 |
| 快速遍数 (Passes) | `fast_self_collision_passes` | `4`, `1..8` | fast 模式下 native 自碰撞 pass 数。 | 仅在 fast 模式显示；相交多时增加，性能不足时降低。 |
| 最大邻居探测数 | `max_self_collision_neighbors` | `64`, min `4`, soft max `256` | 每个顶点或边在自碰撞中允许处理的候选邻居数量。 | 高密网格或复杂折叠可增大；实时预览卡顿时降低。 |

### 力场

| 面板标签 | 字段名 | 默认值与范围 | 作用 | 调参建议 |
| --- | --- | --- | --- | --- |
| 效果器集合 | `force_field_collection` | none | 指定要采样的 Blender Force Field 集合。 | 为空时读取当前场景中的力场；指定集合可隔离某组力场。 |
| 重力 | `force_field_weight_gravity` | `1.0`, `0.0..1.0` | 缩放场景重力对 SSBL 布料的影响。 | `0` 可临时关闭重力，便于检查其他力场效果。 |
| 全部 | `force_field_weight_all` | `1.0`, `0.0..1.0` | 缩放所有非重力 Blender 力场。 | 想整体减弱或增强力场响应时先调它。 |
| 常力 | `force_field_weight_force` | `1.0`, `0.0..1.0` | 缩放 Force 效果器。 | 用于定向推拉布料。 |
| 涡流 | `force_field_weight_vortex` | `1.0`, `0.0..1.0` | 缩放 Vortex 效果器。 | 用于旋转、卷动类效果。 |
| 磁力 | `force_field_weight_magnetic` | `1.0`, `0.0..1.0` | 缩放 Magnetic 效果器。 | 需要磁力式吸引/排斥时调整。 |
| 谐振 | `force_field_weight_harmonic` | `1.0`, `0.0..1.0` | 缩放 Harmonic 效果器。 | 用于弹性牵引和周期性回弹效果。 |
| 电荷 | `force_field_weight_charge` | `1.0`, `0.0..1.0` | 缩放 Charge 效果器。 | 用于点状吸引或排斥。 |
| 兰纳琼斯分子力 | `force_field_weight_lennardjones` | `1.0`, `0.0..1.0` | 缩放 Lennard-Jones 效果器。 | 适合短距离吸引/排斥类效果，通常谨慎使用。 |
| 风力 | `force_field_weight_wind` | `1.0`, `0.0..1.0` | 缩放 Wind 效果器。 | 做风吹布料时使用；先从较低强度开始。 |
| 纹理 | `force_field_weight_texture` | `1.0`, `0.0..1.0` | 缩放 Texture 效果器。 | 用于空间噪声或贴图驱动的扰动。 |
| 紊流 | `force_field_weight_turbulence` | `1.0`, `0.0..1.0` | 缩放 Turbulence 效果器。 | 增加随机抖动；高值可能造成模拟不稳定。 |
| 拖拽 | `force_field_weight_drag` | `1.0`, `0.0..1.0` | 缩放 Drag 效果器。 | 用于减速或空气阻力感。 |

### Solver Tuning

| 面板标签 | 字段名 | 默认值与范围 | 作用 | 调参建议 |
| --- | --- | --- | --- | --- |
| 步长 (Time Step) | `dt` | `0.02`, min `1e-4`, soft max `0.1` | 每个输出帧对应的模拟时间步长。 | 通常保持默认；更小更稳定但需要更多步骤才能达到同等时间。 |
| 子步数 (Substeps) | `substeps` | `14`, min `1`, soft max `32` | 每帧拆分的 XPBD 子步数量。 | 穿透、抖动或高速碰撞时增加；实时性能不足时降低。 |
| 约束迭代 (Iterations) | `iterations` | `2`, min `1`, soft max `128` | 每个子步内的约束投影迭代次数。 | 形变过软或约束不收敛时增加；高值成本明显上升。 |
| 全局阻尼 (Damping) | `damping` | `1.0`, `0.0..1.0` | 每个子步施加的速度阻尼。 | 高值保留更多速度，低值更快耗散运动；异常抖动时适当降低。 |
| 表面摩擦力 | `contact_friction` | `0.35`, `0.0..4.0` | 静态/动态三角形接触的摩擦系数。 | 滑动太多时增大；卡住或拖拽感过强时降低。 |
| 切向滑动阻尼 | `contact_tangent_damping` | `0.2`, `0.0..1.0` | 接触切向速度阻尼。 | 接触面上抖动或滑移过快时增大。 |
| 接触顺应性 (Compliance) | `contact_compliance` | `0.0`, min `0.0`, soft max `1e-4` | 静态/动态三角形接触的 XPBD 顺应性。 | 默认最硬；需要更软的接触响应时小幅增大。 |
| 求交判定间隔 | `self_collision_interval` | `1`, min `1`, soft max `8` | 每 N 个子步运行一次自碰撞。 | 仅在自碰撞开启时显示；提高间隔可省性能但更容易漏碰撞。 |
| 宽相 Probe 间隔 | `self_probe_interval` | `1`, min `1`, soft max `8` | 每 N 次自碰撞 pass 运行昂贵的 probe/recovery。 | 性能优先时增大；需要更严谨恢复时保持 `1`。 |
| 表面配对刷新间隔 | `self_surface_pair_interval` | `1`, min `1`, soft max `8` | 每 N 次自碰撞 pass 刷新 sample-sample 表面配对。 | 高频自碰撞抖动时保持较低；性能优先时提高。 |

### 预览与烘焙

| 面板标签 | 字段名 | 默认值与范围 | 作用 | 调参建议 |
| --- | --- | --- | --- | --- |
| 目标预览帧率 | `preview_target_fps` | `30.0`, min `1.0`, soft max `120.0` | 视口预览计时器的目标 FPS。 | 与目标播放体验匹配；设太高会增加实时压力。 |
| 实时模拟自动缓存 | `auto_cache_realtime` | `false` | 实时预览或时间轴播放时同步写 PC2 缓存。 | 需要边预览边记录结果时启用；普通调参建议关闭。 |
| 使用修改器形态 (Evaluated) | `use_evaluated_mesh` | `false` | 读取经过 Blender 修改器和绑定后的 evaluated mesh。 | 使用 Hook、Armature 或其他形变输入时启用；顶点数和拓扑必须保持不变。 |
| 起始帧 | `bake_start` | `1` | PC2 烘焙输出的第一帧。 | 与场景时间轴起点保持一致。 |
| 结束帧 | `bake_end` | `120`, min `1` | PC2 烘焙输出的最后一帧。 | 必须大于或等于起始帧；长区间会增加缓存时间和文件大小。 |

## English Version

### Reading Notes

- “Default and range” comes from the current `SSBL_PreviewSettings` property definitions; `soft_max` is Blender UI guidance, not a hard clamp.
- Conditional parameters call out their trigger in the tuning notes, for example volume pressure, self collision, or static collider collection.
- Pin weights are multiplied by global `pin_hardness`; the native solver ignores effective pin weights below `0.05`, while `0.75` or higher usually behaves like a hard pin.

### Enable

| UI label | Field | Default and range | Effect | Tuning notes |
| --- | --- | --- | --- | --- |
| Enable SSBL cloth simulation / SSBL cloth solving | `enabled` | `false` | Enables SSBL cloth preview and bake controls on the active Mesh object. | Turn this on before tuning; disabled objects do not participate in SSBL preview. |

### Physics And Material

| UI label | Field | Default and range | Effect | Tuning notes |
| --- | --- | --- | --- | --- |
| Bend hardness | `hardness` | `0.4`, `0.0..1.0` | Overall cloth stiffness control that drives stretch, bend, and hidden tether settings. | Use this first for broad material feel: lower is softer, higher is closer to heavy fabric or leather. |
| Surface thickness | `cloth_thickness` | `0.02`, min `0.001`, soft max `0.1` | Thickness used by self collision and cloth surface spacing. | Increase when cloth intersects itself; too high can look inflated or detached. |
| Material density | `density` | `1.0`, min `1e-4`, soft max `100.0` | Affects vertex mass and inertia. | Higher values feel heavier and resist forces; lower values react more strongly to fields and collisions. |
| Pin vertex group | `pin_vertex_group` | `"ssbl_pin"` | Blender vertex group used for fixed or softly constrained cloth vertices. | Vertex-group weights define local pin strength; an empty name means no pin group. |
| Pin hardness | `pin_hardness` | `1.0`, `0.0..1.0` | Global multiplier for `pin_vertex_group` weights. | `0` disables pins, `1` preserves vertex-group weights; reduce it to loosen all pins. |
| Enable volume pressure | `use_volume_pressure` | `false` | Adds outward pressure along cloth surface normals. | Enable for inflated cloth, cushions, or balloon-like effects; keep off for normal draped cloth. |
| Pressure strength | `pressure_strength` | `0.02`, min `0.0`, soft max `0.2` | Controls volume pressure strength. | Visible only when volume pressure is enabled; high values on dense meshes can jitter or overinflate. |

### Environment And Collision

| UI label | Field | Default and range | Effect | Tuning notes |
| --- | --- | --- | --- | --- |
| Collision margin | `collision_margin` | `0.005`, min `0.0`, soft max `0.2` | Minimum separation distance kept during collision projection. | Increase if penetration is visible; too high can make cloth float above colliders. |
| Infinite ground plane | `use_ground` | `true` | Enables an infinite world-space ground plane collider. | Useful for simple drop tests; use static collider collections for complex environments. |
| Ground height | `ground_height` | `0.0` | World Z height of the infinite ground plane. | Visible only when ground collision is enabled; match it to the scene floor. |
| Static collider collection | `static_collider_collection` | none | Mesh collection used as static triangle/SDF colliders. | Use for non-deforming tables, bodies, and props; empty means no static colliders are uploaded. |
| Animated collider collection | `dynamic_collider_collection` | none | Mesh collection used as one-way evaluated animated collision sources. | Use for moving characters or obstacles; complex colliders increase upload and collision cost. |
| Voxel size | `static_sdf_voxel_size` | `0.0`, min `0.0`, soft max `0.05` | Static SDF voxel size; `0` uses `max(collision_margin * 0.5, 0.002)`. | Visible only when a static collider collection is set; smaller is more accurate but slower and heavier. |
| Band | `static_sdf_band_voxels` | `4`, `1..32`, soft max `16` | Voxel padding band around static SDF collision surfaces. | Increase for unstable or fast contacts; too high increases build and sampling cost. |
| Max resolution | `static_sdf_max_resolution` | `160`, `16..512`, soft max `256` | Maximum grid resolution per axis for static SDFs. | Increase for large complex colliders; lower for performance or memory pressure. |

### Self Collision

| UI label | Field | Default and range | Effect | Tuning notes |
| --- | --- | --- | --- | --- |
| Enable self collision | `self_collision` | `true` | Resolves contacts between the cloth surface and itself. | Recommended for skirts, capes, and folded cloth; disable when performance is more important. |
| Algorithm mode | `self_collision_mode` | `"fast"`, options `fast`, `strict` | Selects the self-collision solve path. | `fast` targets realtime preview; `strict` is heavier and suited for quality checks. |
| Self collision distance | `self_collision_distance` | `0.0`, min `0.0`, soft max `0.1` | Overrides the self-collision contact distance; `0` means automatic. | Keep `0` for most cases; raise it when cloth needs a thicker separation. |
| Fast passes | `fast_self_collision_passes` | `4`, `1..8` | Native self-collision pass count in fast mode. | Visible only in fast mode; increase for intersections, reduce for realtime performance. |
| Max neighbor count | `max_self_collision_neighbors` | `64`, min `4`, soft max `256` | Candidate neighbor budget per vertex or edge during self collision. | Increase for dense or heavily folded cloth; lower when preview is too slow. |

### Force Fields

| UI label | Field | Default and range | Effect | Tuning notes |
| --- | --- | --- | --- | --- |
| Effector collection | `force_field_collection` | none | Blender Force Field collection to sample. | Empty reads force fields from the current scene; set a collection to isolate a group. |
| Gravity | `force_field_weight_gravity` | `1.0`, `0.0..1.0` | Scales scene gravity for SSBL cloth. | Set to `0` to temporarily inspect non-gravity effects. |
| All | `force_field_weight_all` | `1.0`, `0.0..1.0` | Scales all non-gravity Blender force fields. | Use this first to globally attenuate or boost force-field response. |
| Force | `force_field_weight_force` | `1.0`, `0.0..1.0` | Scales Force effectors. | Use for directional pushes and pulls. |
| Vortex | `force_field_weight_vortex` | `1.0`, `0.0..1.0` | Scales Vortex effectors. | Use for swirling and twisting motion. |
| Magnetic | `force_field_weight_magnetic` | `1.0`, `0.0..1.0` | Scales Magnetic effectors. | Adjust when using magnetic attraction or repulsion. |
| Harmonic | `force_field_weight_harmonic` | `1.0`, `0.0..1.0` | Scales Harmonic effectors. | Useful for elastic pulls and periodic recovery. |
| Charge | `force_field_weight_charge` | `1.0`, `0.0..1.0` | Scales Charge effectors. | Use for point-like attraction or repulsion. |
| Lennard-Jones | `force_field_weight_lennardjones` | `1.0`, `0.0..1.0` | Scales Lennard-Jones effectors. | Useful for short-range attraction/repulsion; tune carefully. |
| Wind | `force_field_weight_wind` | `1.0`, `0.0..1.0` | Scales Wind effectors. | Start low when adding wind to cloth. |
| Texture | `force_field_weight_texture` | `1.0`, `0.0..1.0` | Scales Texture effectors. | Use for spatial noise or texture-driven disturbance. |
| Turbulence | `force_field_weight_turbulence` | `1.0`, `0.0..1.0` | Scales Turbulence effectors. | Adds random motion; high values can destabilize simulation. |
| Drag | `force_field_weight_drag` | `1.0`, `0.0..1.0` | Scales Drag effectors. | Use for slowdown or air-resistance feel. |

### Solver Tuning

| UI label | Field | Default and range | Effect | Tuning notes |
| --- | --- | --- | --- | --- |
| Time step | `dt` | `0.02`, min `1e-4`, soft max `0.1` | Simulation time represented by each output frame. | Usually keep the default; smaller is more stable but advances less time per frame. |
| Substeps | `substeps` | `14`, min `1`, soft max `32` | XPBD substeps per frame. | Increase for penetration, jitter, or fast collisions; lower for realtime speed. |
| Iterations | `iterations` | `2`, min `1`, soft max `128` | Constraint projection iterations per substep. | Increase when constraints look too soft or under-converged; high values are expensive. |
| Damping | `damping` | `1.0`, `0.0..1.0` | Velocity damping applied each substep. | Higher keeps more velocity; lower dissipates motion faster and can calm jitter. |
| Surface friction | `contact_friction` | `0.35`, `0.0..4.0` | Friction coefficient for static and dynamic triangle contacts. | Increase if cloth slides too much; decrease if contacts stick or drag too heavily. |
| Tangential damping | `contact_tangent_damping` | `0.2`, `0.0..1.0` | Tangential velocity damping at contacts. | Increase if cloth jitters or slides too quickly across contact surfaces. |
| Contact compliance | `contact_compliance` | `0.0`, min `0.0`, soft max `1e-4` | XPBD compliance for static and dynamic triangle contacts. | Default is the stiffest contact; raise slightly for softer contact response. |
| Intersection interval | `self_collision_interval` | `1`, min `1`, soft max `8` | Runs self collision every N substeps. | Visible when self collision is enabled; higher saves performance but can miss contacts. |
| Broad-phase probe interval | `self_probe_interval` | `1`, min `1`, soft max `8` | Runs expensive probe/recovery every N self-collision passes. | Increase for performance, keep `1` for stricter recovery. |
| Surface pair refresh interval | `self_surface_pair_interval` | `1`, min `1`, soft max `8` | Refreshes sample-sample surface pairs every N self-collision passes. | Keep low for jitter-prone self collision; raise for performance. |

### Preview And Bake

| UI label | Field | Default and range | Effect | Tuning notes |
| --- | --- | --- | --- | --- |
| Target preview FPS | `preview_target_fps` | `30.0`, min `1.0`, soft max `120.0` | Target FPS for the viewport preview timer. | Match the desired preview experience; very high values add realtime pressure. |
| Realtime auto cache | `auto_cache_realtime` | `false` | Writes a PC2 cache while realtime preview or timeline playback is running. | Enable when preview results need to be recorded; keep off during ordinary tuning. |
| Use evaluated mesh | `use_evaluated_mesh` | `false` | Reads mesh input after Blender modifiers and bindings. | Enable for Hook, Armature, or other deformed inputs; vertex count and topology must remain unchanged. |
| Start frame | `bake_start` | `1` | First frame written to PC2 bake output. | Match the scene timeline start. |
| End frame | `bake_end` | `120`, min `1` | Last frame written to PC2 bake output. | Must be greater than or equal to start frame; long ranges increase bake time and file size. |
