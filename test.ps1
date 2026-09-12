param([switch]$NeuralOnly)
. "$PSScriptRoot\env.ps1"
& $python -m pytest vendor/doomfly/tests/test_doom_reference.py -q --disable-warnings
if ($LASTEXITCODE -ne 0) { throw 'Neural numerical reference failed' }
& $python scripts/test_brain.py
if ($LASTEXITCODE -ne 0) { throw 'Full-connectome test failed' }
if (!$NeuralOnly) {
    & $python scripts/test_game.py
    if ($LASTEXITCODE -ne 0) { throw 'CraftGround live action test failed' }
    & "$PSScriptRoot\evaluate-baseline.ps1" -Steps 300
    if ($LASTEXITCODE -ne 0) { throw 'Closed-loop baseline failed' }
}
Write-Host 'All requested tests passed. LEARNING: OFF'
