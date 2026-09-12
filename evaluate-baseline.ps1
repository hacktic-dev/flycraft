param([int]$Steps=300,[switch]$Record)
& "$PSScriptRoot\run-baseline.ps1" -Steps $Steps -Debug -Record:$Record
if ($Steps -ge 100) {
    . "$PSScriptRoot\env.ps1"
    & $python scripts/verify_baseline.py
    if ($LASTEXITCODE -ne 0) { throw 'Closed-loop evidence verification failed' }
}
