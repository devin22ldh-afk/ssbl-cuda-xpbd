param(
    [ValidateSet("Release", "Debug")]
    [string]$Config = "Release",
    [string]$CudaRoot = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $Root "build"

function Resolve-FirstExistingPath {
    param([string[]]$Paths)
    foreach ($path in $Paths) {
        if ($path -and (Test-Path -LiteralPath $path)) {
            return (Resolve-Path -LiteralPath $path).Path
        }
    }
    return $null
}

function Resolve-CommandPath {
    param(
        [string]$Name,
        [string[]]$Fallbacks
    )
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }
    return Resolve-FirstExistingPath $Fallbacks
}

function Get-CudaVersionFromPath {
    param([string]$Path)
    $leaf = Split-Path -Leaf $Path
    if ($leaf -match "^v?([0-9]+)\.([0-9]+)") {
        return [version]("{0}.{1}" -f $Matches[1], $Matches[2])
    }
    return $null
}

function Get-DriverCudaVersion {
    try {
        $text = & nvidia-smi 2>$null | Out-String
        if ($text -match "CUDA Version:\s*([0-9]+)\.([0-9]+)") {
            return [version]("{0}.{1}" -f $Matches[1], $Matches[2])
        }
    } catch {
        return $null
    }
    return $null
}

function Resolve-CudaRoot {
    param([string]$RequestedRoot)
    if ($RequestedRoot) {
        if (Test-Path -LiteralPath (Join-Path $RequestedRoot "bin\nvcc.exe")) {
            return (Resolve-Path -LiteralPath $RequestedRoot).Path
        }
        throw "Requested CUDA root does not contain bin\nvcc.exe: $RequestedRoot"
    }

    $rawCandidates = @()
    if ($env:CUDA_PATH) {
        $rawCandidates += $env:CUDA_PATH
    }
    $cudaBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path -LiteralPath $cudaBase) {
        $rawCandidates += Get-ChildItem -LiteralPath $cudaBase -Directory | ForEach-Object { $_.FullName }
    }

    $candidates = foreach ($candidate in ($rawCandidates | Select-Object -Unique)) {
        if (Test-Path -LiteralPath (Join-Path $candidate "bin\nvcc.exe")) {
            $resolved = (Resolve-Path -LiteralPath $candidate).Path
            [pscustomobject]@{
                Path = $resolved
                Version = Get-CudaVersionFromPath $resolved
            }
        }
    }
    if (-not $candidates) {
        return $null
    }

    $driverCuda = Get-DriverCudaVersion
    if ($driverCuda) {
        $compatible = $candidates | Where-Object { $_.Version -and $_.Version -le $driverCuda }
        if ($compatible) {
            return ($compatible | Sort-Object Version -Descending | Select-Object -First 1).Path
        }
    }
    return ($candidates | Sort-Object Version -Descending | Select-Object -First 1).Path
}

function Resolve-VsDevCmd {
    $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        $installPath = & $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -eq 0 -and $installPath) {
            $candidate = Join-Path $installPath "Common7\Tools\VsDevCmd.bat"
            if (Test-Path -LiteralPath $candidate) {
                return (Resolve-Path -LiteralPath $candidate).Path
            }
        }
    }
    return Resolve-FirstExistingPath @(
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
    )
}

$cmake = Resolve-CommandPath "cmake" @(
    "C:\Program Files\CMake\bin\cmake.exe",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
)
if (-not $cmake) {
    throw "CMake was not found. Install CMake 3.25+ and retry."
}

$cudaRoot = Resolve-CudaRoot $CudaRoot
if (-not $cudaRoot) {
    throw "CUDA Toolkit was not found. Install CUDA Toolkit 12.6+ and retry."
}

$vsDevCmd = Resolve-VsDevCmd
if (-not $vsDevCmd) {
    throw "VS 2022 C++ Build Tools were not found. Install Visual Studio Build Tools with the C++ workload."
}

$cudaBin = Join-Path $cudaRoot "bin"
$cudaToolkitDir = $cudaRoot.TrimEnd('\') + "\"
$tempCmd = Join-Path ([System.IO.Path]::GetTempPath()) ("ssbl_cuda_build_{0}.cmd" -f ([System.Guid]::NewGuid().ToString("N")))
$lines = @(
    "@echo off",
    "call ""$vsDevCmd"" -arch=x64 -host_arch=x64 || exit /b 1",
    "set ""PATH=$(Split-Path -Parent $cmake);$cudaBin;%PATH%""",
    "set ""CUDA_PATH=$cudaRoot""",
    "set ""CudaToolkitDir=$cudaToolkitDir""",
    """$cmake"" -S ""$Root"" -B ""$BuildDir"" -G ""Visual Studio 17 2022"" -A x64 -T ""cuda=$cudaRoot"" -DCMAKE_CUDA_COMPILER=""$cudaBin\nvcc.exe"" || exit /b 1",
    """$cmake"" --build ""$BuildDir"" --config $Config || exit /b 1"
)

try {
    Set-Content -LiteralPath $tempCmd -Value $lines -Encoding ASCII
    & cmd.exe /d /s /c """$tempCmd"""
    if ($LASTEXITCODE -ne 0) {
        throw "Native CUDA build failed with exit code $LASTEXITCODE."
    }
} finally {
    Remove-Item -LiteralPath $tempCmd -Force -ErrorAction SilentlyContinue
}

$dllPath = Join-Path $Root "bin\ssbl_xpbd_cuda_abi22.dll"
if (-not (Test-Path -LiteralPath $dllPath)) {
    throw "Build completed but DLL was not found at $dllPath."
}

Write-Host "Built $dllPath"
