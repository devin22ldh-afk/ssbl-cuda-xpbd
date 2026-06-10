# SSBL CUDA XPBD

SSBL CUDA XPBD 是一个面向 Blender 5.0 的本地 CUDA XPBD 布料插件，用于在视口中实时预览布料模拟，并将模拟结果烘焙为 PC2 点缓存。

SSBL CUDA XPBD is a native CUDA XPBD cloth add-on for Blender 5.0. It provides viewport cloth preview and can bake simulation results to PC2 point-cache files.

## 主要功能 / Features

- 实时布料预览：启用后播放 Blender 时间轴即可驱动 XPBD 布料模拟。
- 时间轴集成：插件会在播放、帧变更和停止播放时维护预览会话状态。
- PC2 缓存烘焙：可将活动网格的模拟结果写入 `ssbl_cache/<object>_xpbd.pc2` 并绑定到 Mesh Cache 修改器。
- 碰撞支持：支持地面平面、球体碰撞器、静态碰撞集合，以及基于 CUDA 的静态 SDF 碰撞。
- 自碰撞：提供 `fast` 预览优先模式和 `strict` 质量优先模式。
- 多布料预览：支持多对象布料会话、跨布料动态碰撞和碰撞层级。
- 材质与约束：支持硬度、布料厚度、密度、阻尼、接触摩擦、加权 pin 顶点组和体积气压。
- Blender 力场：可从力场集合采样并上传到 native 解算器，同时支持按类型设置权重。
- Native 后端：通过 `native/bin/ssbl_xpbd_cuda_abi41.dll` 使用 ABI41 CUDA 解算器。

- Real-time cloth preview: play the Blender timeline to drive XPBD cloth simulation.
- Timeline integration: preview sessions are managed through playback and frame-change handlers.
- PC2 baking: simulation output can be written to `ssbl_cache/<object>_xpbd.pc2` and bound through a Mesh Cache modifier.
- Collision support: ground plane, sphere collider, static collider collections, and CUDA static SDF collision.
- Self collision: `fast` preview-oriented mode and `strict` quality-oriented mode.
- Multi-cloth preview: multi-object cloth sessions, cross-cloth dynamic collision, and collision layers.
- Material and constraints: hardness, cloth thickness, density, damping, contact friction, weighted pin vertex groups, and volume pressure.
- Blender force fields: force-field collections can be sampled and uploaded to the native solver with per-type weights.
- Native backend: uses the ABI41 CUDA solver DLL at `native/bin/ssbl_xpbd_cuda_abi41.dll`.

## 环境要求 / Requirements

- Blender 5.0。
- Windows。
- NVIDIA GPU 和支持 CUDA 的显卡驱动。
- 已存在或可构建的 ABI41 native DLL：`native/bin/ssbl_xpbd_cuda_abi41.dll`。
- 如果需要重新构建 native 后端，还需要 CUDA Toolkit 12.6+、Visual Studio Build Tools 2022 x64 C++ 工具链和 CMake 3.25+。

- Blender 5.0.
- Windows.
- NVIDIA GPU with a CUDA-capable driver.
- A prebuilt or locally built ABI41 native DLL at `native/bin/ssbl_xpbd_cuda_abi41.dll`.
- To rebuild the native backend, install CUDA Toolkit 12.6+, Visual Studio Build Tools 2022 with the x64 C++ toolchain, and CMake 3.25+.

## 安装 / Installation

将本目录放到 Blender 5.0 的用户插件目录，并确保目录名为 `ssbl`：

Place this directory in the Blender 5.0 user add-ons folder and keep the directory name as `ssbl`:

```powershell
$addons = "$env:APPDATA\Blender Foundation\Blender\5.0\scripts\addons"
Copy-Item -Recurse -LiteralPath "C:\path\to\ssbl" -Destination "$addons\ssbl"
```

然后在 Blender 中启用插件：

Then enable the add-on in Blender:

1. 打开 `Edit > Preferences > Add-ons`。
2. 搜索 `SSBL CUDA XPBD`。
3. 勾选启用插件。

1. Open `Edit > Preferences > Add-ons`.
2. Search for `SSBL CUDA XPBD`.
3. Enable the add-on.

## 快速上手 / Quick Start

1. 在场景中选择一个 Mesh 对象。
2. 打开 `Properties > Physics > SSBL CUDA XPBD`。
3. 启用 `SSBL` 布料模拟。
4. 可选：创建名为 `ssbl_pin` 的顶点组，用于固定布料上的点。顶点组权重会驱动软 pin 强度；native 解算器会把低于 `0.05` 的有效权重视为未 pin。
5. 根据需要设置硬度、布料厚度、碰撞、力场、自碰撞和多布料选项。
6. 播放 Blender 时间轴，插件会在视口中更新布料预览。
7. 在 Cache & Bake 区域设置起止帧并执行 Bake，将结果写入 PC2 缓存。

1. Select a Mesh object in the scene.
2. Open `Properties > Physics > SSBL CUDA XPBD`.
3. Enable SSBL cloth simulation.
4. Optional: create a vertex group named `ssbl_pin` to pin cloth vertices. Vertex-group weights drive soft pin strength; effective weights below `0.05` are treated as unpinned by the native solver.
5. Configure hardness, cloth thickness, collision, force fields, self collision, and multi-cloth options as needed.
6. Play the Blender timeline to update the cloth preview in the viewport.
7. Use the Cache & Bake section to set frame range and bake the result to a PC2 cache.

## Native CUDA 后端 / Native CUDA Backend

插件默认加载：

The add-on loads this DLL by default:

```text
native/bin/ssbl_xpbd_cuda_abi41.dll
```

如果 DLL 缺失，插件会报告 native CUDA solver 不可用。更多构建说明见 `native/README.md`。

If the DLL is missing, the add-on reports that the native CUDA solver is unavailable. See `native/README.md` for detailed build notes.

检查工具链：

Check the build toolchain:

```powershell
Push-Location .\native
.\check_toolchain.ps1
Pop-Location
```

构建 ABI41 后端并运行 native smoke：

Build the ABI41 backend and run the native smoke check:

```powershell
Push-Location .\native
.\build_recon.ps1
Pop-Location
```

如果需要指定其他 ABI41 兼容 DLL：

To load another ABI41-compatible DLL:

```powershell
$env:SSBL_NATIVE_DLL_PATH = "C:\path\to\ssbl_xpbd_cuda_abi41.dll"
```

## 验证 / Validation

以下命令应在项目根目录运行。请根据本机安装位置调整 `blender.exe` 路径。

Run these commands from the add-on root. Adjust the `blender.exe` path for your machine.

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --python ".\tools\animated_inputs_smoke.py"
```

成功时会打印 `SSBL_ANIMATED_INPUTS_SMOKE`，并验证动画输入、pin 附着、预览、PC2 烘焙和缓存清理。

On success, this prints `SSBL_ANIMATED_INPUTS_SMOKE` and validates animated inputs, pin attachments, preview, PC2 baking, and cache cleanup.

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --python ".\tools\benchmark_v2_multicloth.py"
```

成功时会打印 `SSBL_V2_BENCHMARK`，并覆盖 10k 布料、自碰撞、多布料和静态碰撞集合场景。

On success, this prints `SSBL_V2_BENCHMARK` and covers 10k cloth, self collision, multi-cloth, and static collider collection scenarios.

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --python ".\tools\object_collision_smoke.py"
```

成功时会打印 `SSBL_OBJECT_COLLISION_SMOKE`，并验证解析地面/墙体接触、静态 SDF、静态碰撞体更新和摩擦行为。需要缩短本地检查时，可将 `SSBL_OBJECT_COLLISION_CASES` 设为逗号分隔的子集，例如 `ground,wall,static_mesh,moving_static_mesh`。

On success, this prints `SSBL_OBJECT_COLLISION_SMOKE` and validates analytic ground/wall contacts, static SDF cases, static collider updates, and friction behavior. To run a shorter subset locally, set `SSBL_OBJECT_COLLISION_CASES` to a comma-separated list such as `ground,wall,static_mesh,moving_static_mesh`.

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --python ".\tools\native_contact_group_probe.py" -- --case analytic_ground
& "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --python ".\tools\native_contact_group_probe.py" -- --case self_vv
```

成功时每个 probe 会打印 `SSBL_NATIVE_CONTACT_GROUP_PROBE`。可用 case 包括 `analytic_ground`、`analytic_wall`、`analytic_sphere`、`analytic_corner`、`static_sdf` 和 `self_vv`；它们适合在不搭建完整 Blender 场景时隔离验证 native contact grouping、摩擦或自接触变化。

On success, each probe prints `SSBL_NATIVE_CONTACT_GROUP_PROBE`. Available cases are `analytic_ground`, `analytic_wall`, `analytic_sphere`, `analytic_corner`, `static_sdf`, and `self_vv`; use these when isolating native contact grouping, friction, or self-contact changes without building a full Blender scene.

Native 后端 smoke 可通过：

Run the native backend smoke through:

```powershell
Push-Location .\native
.\build_recon.ps1
Pop-Location
```

成功输出应包含 `SSBL_ABI41_NATIVE_OK`、`SSBL_ABI41_STATIC_SDF_OK` 和 `SSBL_ABI41_PIN_WEIGHT_OK`。

Successful output should include `SSBL_ABI41_NATIVE_OK`, `SSBL_ABI41_STATIC_SDF_OK`, and `SSBL_ABI41_PIN_WEIGHT_OK`.

## 常见问题 / Troubleshooting

### 缺少 CUDA DLL / Missing CUDA DLL

确认 `native/bin/ssbl_xpbd_cuda_abi41.dll` 存在。也可以通过 `SSBL_NATIVE_DLL_PATH` 指向其他 ABI41 兼容 DLL。

Make sure `native/bin/ssbl_xpbd_cuda_abi41.dll` exists. You can also point `SSBL_NATIVE_DLL_PATH` to another ABI41-compatible DLL.

### 工具链检查失败 / Toolchain Check Fails

运行 `native/check_toolchain.ps1`。如果缺少 `nvcc`、`cl` 或 `cmake`，安装 CUDA Toolkit 12.6+、Visual Studio Build Tools 2022 x64 C++ 工具链和 CMake 3.25+。

Run `native/check_toolchain.ps1`. If `nvcc`, `cl`, or `cmake` is missing, install CUDA Toolkit 12.6+, Visual Studio Build Tools 2022 with the x64 C++ toolchain, and CMake 3.25+.

### 看不到面板 / Panel Is Not Visible

确认已选择 Mesh 对象。SSBL 面板位于 `Properties > Physics > SSBL CUDA XPBD`，只在活动对象是 Mesh 时显示。

Select a Mesh object. The SSBL panel is under `Properties > Physics > SSBL CUDA XPBD` and is only shown for active Mesh objects.

### Pin 顶点组不起作用 / Pin Vertex Group Does Not Work

默认 pin 顶点组名为 `ssbl_pin`。如果使用其他名称，请在 SSBL 材质与约束设置中填写对应顶点组。顶点组权重会乘以全局 pin 硬度；软 pin 的有效权重至少需要 `0.05`，硬 pin 行为通常使用 `0.75` 或更高权重。

The default pin vertex group is `ssbl_pin`. If you use another name, set it in the SSBL material and constraint settings. Vertex-group weights are multiplied by the global pin hardness; use effective weights of at least `0.05` for soft pins and `0.75` or higher for hard pin behavior.

### 自碰撞较慢 / Self Collision Is Slow

优先使用 `fast` 模式做视口预览；需要更高质量或更严格避免相交时再切换到 `strict` 模式。增加顶点数、启用自碰撞、多布料碰撞或静态 SDF 都会提高计算成本。

Use `fast` mode for viewport preview first. Switch to `strict` when higher quality or stricter intersection prevention is needed. Higher vertex counts, self collision, multi-cloth collision, and static SDF collision all increase compute cost.

## 项目结构 / Project Layout

```text
__init__.py          Blender add-on registration and settings
ui.py               Physics panel UI
operators.py        Preview, reset, bake, and cache operators
solver.py           Public solver facade
session_manager.py  Preview and bake session orchestration
xpbd_core.py        Cloth build data and XPBD settings helpers
collision.py        Static collider collection helpers
force_fields.py     Blender force-field sampling
native_backend.py   ctypes bridge to the CUDA DLL
parameters.md       User-facing guide for visible SSBL panel parameters
native/             CUDA source, ABI headers, build scripts, native README
tools/              Smoke tests, benchmarks, and preview recording scripts
```
