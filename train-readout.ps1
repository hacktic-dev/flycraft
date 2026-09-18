param(
    [ValidateSet('Fresh','Resume','Evaluate','Replay')]
    [string]$Mode = 'Fresh',
    [string]$Checkpoint = 'latest',
    [string]$Config = 'config/new-arch.json',
    [int]$Steps = 0,
    [int]$Episodes = 0,
    [switch]$NoPreview
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\env.ps1"

# Keep the existing CraftGround block-breaking telemetry preparation used by train.ps1.
& $python "$PSScriptRoot\scripts\prepare_training_runtime.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$args = @('-m','flycraft.imitation_train','--mode',$Mode.ToLower(),'--config',$Config)
if ($Checkpoint) { $args += @('--checkpoint',$Checkpoint) }
if ($Steps -gt 0) { $args += @('--steps',$Steps) }
if ($Episodes -gt 0) { $args += @('--episodes',$Episodes) }
if ($NoPreview) { $args += '--no-preview' }

& $python @args
exit $LASTEXITCODE
