param(
    [ValidateSet('Fresh','Resume','Branch','Evaluate','Replay')][string]$Mode='Fresh',
    [string]$Checkpoint='latest',
    [string]$Config,
    [int]$Steps,
    [int]$CheckpointEvery,
    [int]$Episodes,
    [ValidateSet('Spikes','Voltage','Combined')]
    [string]$ActivityMode='Spikes',
    [double]$VoltageSmoothingMs=80,
    [switch]$NoPreview
)
. "$PSScriptRoot\env.ps1"
$env:OPENBLAS_NUM_THREADS='1'
$arguments=@('-m','flycraft.train','--mode',$Mode.ToLower(),'--checkpoint',$Checkpoint,'--activity-mode',$ActivityMode.ToLower(),'--voltage-smoothing-ms',"$VoltageSmoothingMs")
if ($Config) { $arguments+=@('--config',$Config) }
if ($PSBoundParameters.ContainsKey('Steps')) { $arguments+=@('--steps',"$Steps") }
if ($PSBoundParameters.ContainsKey('CheckpointEvery')) { $arguments+=@('--checkpoint-every',"$CheckpointEvery") }
if ($PSBoundParameters.ContainsKey('Episodes')) { $arguments+=@('--episodes',"$Episodes") }
if ($NoPreview) { $arguments+='--no-preview' }
& $python scripts/prepare_training_runtime.py
if ($LASTEXITCODE -ne 0) { throw 'Training telemetry preparation failed.' }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw 'Training failed. Check the terminal and artifacts/training.' }
