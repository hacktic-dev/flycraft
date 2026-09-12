param(
    [switch]$Record,
    [switch]$RecordVisuals,
    [switch]$RecordActivity,
    [switch]$RecordTrajectory,
    [switch]$NoPreview,
    [switch]$Debug,
    [int]$Steps=0,
    [string]$Playback,
    [ValidateSet('Spikes','Voltage','Combined')]
    [string]$ActivityMode='Spikes',
    [ValidateRange(0,1000)]
    [int]$VoltageSmoothingMs=80
)
. "$PSScriptRoot\env.ps1"
$arguments=@('-m','flycraft.run','--steps',"$Steps",'--activity-mode',$ActivityMode.ToLower(),'--voltage-smoothing-ms',"$VoltageSmoothingMs")
if ($Record) { $arguments+='--record' }
if ($Debug) { $arguments+='--debug' }
if ($RecordVisuals) { $arguments+='--record-visuals' }
if ($NoPreview) { $arguments+='--no-preview' }
if ($RecordActivity) { $arguments+='--record-activity' }
if ($RecordTrajectory) { $arguments+='--record-trajectory' }
if ($Playback) { $arguments+=@('--playback',$Playback) }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "Baseline failed (exit $LASTEXITCODE). See artifacts and Minecraft runtime logs." }
