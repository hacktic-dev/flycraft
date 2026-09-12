param([switch]$SkipDataAudit)
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
function CheckExit([string]$Step) { if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit $LASTEXITCODE)" } }
New-Item -ItemType Directory -Force .tools,vendor,artifacts | Out-Null
if (!(Test-Path .git)) { git init; CheckExit 'Git initialization' }
if (!(Test-Path vendor/doomfly/.git)) { git clone https://github.com/nftechie/doomfly.git vendor/doomfly; CheckExit 'DOOMFLY download'; git -C vendor/doomfly checkout 71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33; CheckExit 'DOOMFLY revision' }
if (!(Test-Path vendor/CraftGround/.git)) { git clone https://github.com/yhs0602/CraftGround.git vendor/CraftGround; CheckExit 'CraftGround download'; git -C vendor/CraftGround checkout 18eba01a87a8481bc4fbb54b42fee1428c0c3719; CheckExit 'CraftGround revision' }
if (!(Test-Path .tools/uv/bin/uv.exe)) { python -m pip install --target .tools/uv uv==0.12.13; CheckExit 'uv installation' }
$env:UV_PYTHON_INSTALL_DIR="$PSScriptRoot\.tools\python"
& .tools/uv/bin/uv.exe python install 3.11.16; CheckExit 'Python 3.11 installation'
if (!(Test-Path .venv/Scripts/python.exe)) { & .tools/uv/bin/uv.exe venv --python 3.11.16 .venv; CheckExit 'Python environment' }
& .tools/uv/bin/uv.exe pip install --python .venv/Scripts/python.exe -r requirements.lock.txt --build-constraint vendor/doomfly/neural-build-constraints.txt; CheckExit 'Python dependencies'
& .venv/Scripts/python.exe scripts/download.py; CheckExit 'Verified data/tool downloads'
if (!(Test-Path .tools/glew-2.2.0)) {
    $ProgressPreference='SilentlyContinue'
    Invoke-WebRequest https://github.com/nigels-com/glew/releases/download/glew-2.2.0/glew-2.2.0-win32.zip -OutFile .tools/glew.zip
    Expand-Archive .tools/glew.zip .tools -Force
}
. "$PSScriptRoot\env.ps1"
& $python scripts/build_kernel.py; CheckExit 'Native neural kernel'
if (!(Test-Path vendor/doomfly/outputs/doom/malecns_v1/graph.npz)) {
    & $python -m doom.connectome malecns_v1; CheckExit 'Connectome import'
    & $python -m doom.prepare; CheckExit 'Neural graph preparation'
}
if (!$SkipDataAudit) { & $python -m doom.audit_data; CheckExit 'Full graph audit' }
& $python scripts/prepare_runtime.py; CheckExit 'Windows runtime preparation'
$runtime="$PSScriptRoot\.venv\Lib\site-packages\craftground_runtime_mc121"
cmake -S "$runtime\src\main\cpp" -B "$runtime" -G Ninja "-DGLEW_INCLUDE_DIR=$root/.tools/glew-2.2.0/include" "-DGLEW_SHARED_LIBRARY_RELEASE=$root/.tools/glew-2.2.0/lib/Release/x64/glew32.lib" -DCMAKE_BUILD_TYPE=Release; CheckExit 'CraftGround native configuration'
cmake --build "$runtime" --parallel 4; CheckExit 'CraftGround native build'
Copy-Item "$root\.tools\glew-2.2.0\bin\Release\x64\glew32.dll" "$runtime\glew32.dll" -Force
Push-Location $runtime
try { .\gradlew.bat classes --no-daemon; CheckExit 'Minecraft Java build' } finally { Pop-Location }
& $python scripts/test_brain.py; CheckExit 'Independent neural test'
& $python -m flycraft.activity; CheckExit 'Read-only activity observer build'
& $python -m flycraft.anatomy; CheckExit 'Anatomical positions'
Write-Host 'Setup complete. Run .\test.ps1 for live Minecraft verification.'
