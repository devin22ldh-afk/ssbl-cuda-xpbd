$ErrorActionPreference = "Stop"

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

function Resolve-CudaNvcc {
    $rawCandidates = @()
    if ($env:CUDA_PATH) {
        $rawCandidates += $env:CUDA_PATH
    }
    $cudaBase = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path -LiteralPath $cudaBase) {
        $rawCandidates += Get-ChildItem -LiteralPath $cudaBase -Directory | ForEach-Object { $_.FullName }
    }

    $candidates = foreach ($candidate in ($rawCandidates | Select-Object -Unique)) {
        $nvcc = Join-Path $candidate "bin\nvcc.exe"
        if (Test-Path -LiteralPath $nvcc) {
            $resolvedRoot = (Resolve-Path -LiteralPath $candidate).Path
            [pscustomobject]@{
                Nvcc = (Resolve-Path -LiteralPath $nvcc).Path
                Version = Get-CudaVersionFromPath $resolvedRoot
            }
        }
    }
    $driverCuda = Get-DriverCudaVersion
    if ($driverCuda -and $candidates) {
        $compatible = $candidates | Where-Object { $_.Version -and $_.Version -le $driverCuda }
        if ($compatible) {
            return ($compatible | Sort-Object Version -Descending | Select-Object -First 1).Nvcc
        }
    }
    if ($candidates) {
        return ($candidates | Sort-Object Version -Descending | Select-Object -First 1).Nvcc
    }
    return Resolve-CommandPath "nvcc" @()
}

function Resolve-Cl {
    $fallbacks = @(
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"
    )
    $expanded = foreach ($pattern in $fallbacks) {
        Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
    }
    return Resolve-CommandPath "cl" $expanded
}

$tools = @(
    @{ Name = "cmake"; Path = Resolve-CommandPath "cmake" @("C:\Program Files\CMake\bin\cmake.exe", "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"); Hint = "Install CMake 3.25+." },
    @{ Name = "nvcc"; Path = Resolve-CudaNvcc; Hint = "Install CUDA Toolkit 12.6+." },
    @{ Name = "cl"; Path = Resolve-Cl; Hint = "Install VS 2022 Build Tools with the C++ workload." }
)

$failed = $false
foreach ($tool in $tools) {
    if ($tool.Path) {
        Write-Host ("OK   {0}: {1}" -f $tool.Name, $tool.Path)
    } else {
        Write-Host ("MISS {0}: {1}" -f $tool.Name, $tool.Hint)
        $failed = $true
    }
}

try {
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} catch {
    Write-Host "MISS nvidia-smi: Install or repair the NVIDIA driver."
    $failed = $true
}

if ($failed) {
    exit 1
}

Write-Host "Native CUDA XPBD toolchain looks ready."
