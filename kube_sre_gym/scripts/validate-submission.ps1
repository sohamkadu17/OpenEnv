param(
    [Parameter(Mandatory = $true)]
    [string]$PingUrl,

    [Parameter(Mandatory = $false)]
    [string]$RepoDir = ".",

    [Parameter(Mandatory = $false)]
    [int]$DockerBuildTimeoutSec = 600
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    $ts = (Get-Date).ToUniversalTime().ToString("HH:mm:ss")
    Write-Host "[$ts] $Message"
}

function Write-Pass {
    param([string]$Message)
    Write-Log "PASSED -- $Message"
}

function Write-Fail {
    param([string]$Message)
    Write-Log "FAILED -- $Message"
}

function Stop-At {
    param([string]$StepName)
    Write-Host ""
    Write-Host "Validation stopped at $StepName. Fix the above before continuing."
    exit 1
}

try {
    $resolvedRepoDir = (Resolve-Path -Path $RepoDir).Path
} catch {
    Write-Host "Error: directory '$RepoDir' not found"
    exit 1
}

$PingUrl = $PingUrl.TrimEnd('/')

Write-Host ""
Write-Host "========================================"
Write-Host "  OpenEnv Submission Validator (PS)"
Write-Host "========================================"
Write-Log "Repo:     $resolvedRepoDir"
Write-Log "Ping URL: $PingUrl"
Write-Host ""

# Step 1/3: Space ping
Write-Log "Step 1/3: Pinging HF Space ($PingUrl/reset) ..."
try {
    $resp = Invoke-WebRequest -Uri "$PingUrl/reset" -Method POST -ContentType "application/json" -Body "{}" -TimeoutSec 30 -UseBasicParsing
    if ($resp.StatusCode -eq 200) {
        Write-Pass "HF Space is live and responds to /reset"
    } else {
        Write-Fail "HF Space /reset returned HTTP $($resp.StatusCode) (expected 200)"
        Stop-At "Step 1"
    }
} catch {
    Write-Fail "HF Space not reachable or /reset failed: $($_.Exception.Message)"
    Stop-At "Step 1"
}

# Step 2/3: Docker build
Write-Log "Step 2/3: Running docker build ..."
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Fail "docker command not found"
    Stop-At "Step 2"
}

$dockerContext = $null
$rootDockerfile = Join-Path $resolvedRepoDir "Dockerfile"
$serverDockerfile = Join-Path $resolvedRepoDir "server\Dockerfile"

if (Test-Path $rootDockerfile) {
    $dockerContext = $resolvedRepoDir
} elseif (Test-Path $serverDockerfile) {
    $dockerContext = Join-Path $resolvedRepoDir "server"
} else {
    Write-Fail "No Dockerfile found in repo root or server/ directory"
    Stop-At "Step 2"
}

Write-Log "  Found Dockerfile in $dockerContext"

$dockerTimedOut = $false
$dockerJob = Start-Job -ScriptBlock {
    param($ctx)
    Set-Location $ctx
    docker build . 2>&1 | Out-String
} -ArgumentList $dockerContext

if (-not (Wait-Job -Job $dockerJob -Timeout $DockerBuildTimeoutSec)) {
    $dockerTimedOut = $true
    Stop-Job $dockerJob -Force | Out-Null
}

$dockerOutput = Receive-Job -Job $dockerJob -Keep -ErrorAction SilentlyContinue
$dockerState = (Get-Job -Id $dockerJob.Id).State
Remove-Job -Job $dockerJob -Force | Out-Null

if ($dockerTimedOut) {
    Write-Fail "Docker build failed (timeout=${DockerBuildTimeoutSec}s)"
    Stop-At "Step 2"
}

if ($dockerState -eq "Completed") {
    Write-Pass "Docker build succeeded"
} else {
    Write-Fail "Docker build failed"
    if ($dockerOutput) {
        ($dockerOutput -split "`n") | Select-Object -Last 20 | ForEach-Object { Write-Host $_ }
    }
    Stop-At "Step 2"
}

# Step 3/3: openenv validate
Write-Log "Step 3/3: Running openenv validate ..."

$openenvCmd = $null
if (Get-Command openenv -ErrorAction SilentlyContinue) {
    $openenvCmd = "openenv"
} elseif (Test-Path (Join-Path $resolvedRepoDir "..\.venv\Scripts\openenv.exe")) {
    $openenvCmd = (Join-Path $resolvedRepoDir "..\.venv\Scripts\openenv.exe")
} elseif (Test-Path (Join-Path $resolvedRepoDir ".venv\Scripts\openenv.exe")) {
    $openenvCmd = (Join-Path $resolvedRepoDir ".venv\Scripts\openenv.exe")
}

if (-not $openenvCmd) {
    Write-Fail "openenv command not found"
    Stop-At "Step 3"
}

try {
    Push-Location $resolvedRepoDir
    & $openenvCmd validate | Out-String | Write-Host
    Pop-Location
    Write-Pass "openenv validate passed"
} catch {
    Pop-Location
    Write-Fail "openenv validate failed: $($_.Exception.Message)"
    Stop-At "Step 3"
}

Write-Host ""
Write-Host "========================================"
Write-Host "  All 3/3 checks passed!"
Write-Host "  Your submission is ready to submit."
Write-Host "========================================"
Write-Host ""

exit 0
