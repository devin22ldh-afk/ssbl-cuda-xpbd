# SSBL Native CUDA XPBD Backend

This directory contains the native CUDA solver used by the Blender add-on.
The add-on now prefers the reconstructed ABI38 solver at
`native/bin/ssbl_xpbd_cuda_abi38.dll` through `ctypes`.
If ABI38 is missing, it falls back to the legacy ABI36 solver at
`native/bin/ssbl_xpbd_cuda_abi36.dll`.

## Windows prerequisites

- NVIDIA driver with CUDA support.
- CUDA Toolkit 12.6 or newer, with `nvcc.exe` in `PATH`.
- Visual Studio Build Tools 2022 with the x64 C++ toolchain, with `cl.exe` in `PATH`.
- CMake 3.25 or newer, with `cmake.exe` in `PATH`.

For the reconstructed ABI38 smoke build, run:

```powershell
.\build_recon.ps1
```

This builds only the ABI38 target and runs `ssbl_abi41_smoke.exe`, which should
print `SSBL_ABI41_NATIVE_OK`.

For the legacy ABI36 backend, run:

```powershell
.\check_toolchain.ps1
.\build.ps1 -Config Release
```

The expected ABI38 output DLL is:

```text
native/bin/ssbl_xpbd_cuda_abi38.dll
```

Set `SSBL_LEGACY_NATIVE=1` to force ABI36, or `SSBL_NATIVE_DLL_PATH` to load a
specific candidate DLL.
