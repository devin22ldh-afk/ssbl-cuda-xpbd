# SSBL Native CUDA XPBD Backend

This directory contains the native CUDA solver used by the Blender add-on.
The add-on loads `native/bin/ssbl_xpbd_cuda_abi28.dll` through `ctypes`.

## Windows prerequisites

- NVIDIA driver with CUDA support.
- CUDA Toolkit 12.6 or newer, with `nvcc.exe` in `PATH`.
- Visual Studio Build Tools 2022 with the x64 C++ toolchain, with `cl.exe` in `PATH`.
- CMake 3.25 or newer, with `cmake.exe` in `PATH`.

Open a Visual Studio 2022 Developer PowerShell, then run:

```powershell
.\check_toolchain.ps1
.\build.ps1 -Config Release
```

The expected output DLL is:

```text
native/bin/ssbl_xpbd_cuda_abi28.dll
```

If Blender reports that the CUDA solver DLL is missing, build this native backend first.
