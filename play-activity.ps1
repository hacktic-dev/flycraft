param(
    [Parameter(Mandatory=$true,Position=0)]
    [string]$ActivityFile,
    [ValidateSet('Spikes','Voltage','Combined')]
    [string]$ActivityMode='Combined',
    [ValidateRange(0,1000)]
    [int]$VoltageSmoothingMs=80
)
. "$PSScriptRoot\env.ps1"
& $python -m flycraft.activity_player $ActivityFile --mode $ActivityMode.ToLower() --voltage-smoothing-ms $VoltageSmoothingMs
if ($LASTEXITCODE -ne 0) { throw "Activity player failed (exit $LASTEXITCODE)." }
