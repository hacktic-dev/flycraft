$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Set-Location $root
$env:JAVA_HOME = (Get-ChildItem "$root\.tools\jdk-*" -Directory | Select-Object -First 1).FullName
$llvm = (Get-ChildItem "$root\.tools\llvm-mingw-*" -Directory | Select-Object -First 1).FullName
$env:PATH = "$root\.venv\Scripts;$env:JAVA_HOME\bin;$llvm\bin;$env:PATH"
$env:PYTHONPATH = "$root\src;$root\vendor\doomfly"
$env:GRADLE_USER_HOME = "$root\.tools\gradle"
$env:PYTHONUNBUFFERED = '1'
$env:CRAFTGROUND_JVM_MAX_MEMORY = '3G'
$env:CC = "$llvm\bin\clang.exe"
$env:CXX = "$llvm\bin\clang++.exe"
$env:CMAKE_GENERATOR = 'Ninja'
$env:CMAKE_PREFIX_PATH = "$root\.tools\native"
$python = "$root\.venv\Scripts\python.exe"
if (!(Test-Path $python)) { throw 'Run .\setup.ps1 first.' }
$env:PATH = "$root\.tools\glew-2.2.0\bin\Release\x64;$env:PATH"
