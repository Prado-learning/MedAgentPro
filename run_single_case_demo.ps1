param(
    [Parameter(Mandatory = $true)]
    [string]$Image,
    [string]$DataRoot = "Glaucoma",
    [string]$OutputDir = "",
    [string]$ApiKey = "",
    [string]$BaseUrl = "",
    [string]$Model = ""
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$argsList = @(
    (Join-Path $scriptDir "single_case_demo.py"),
    "--image", $Image,
    "--data-root", $DataRoot
)

if ($OutputDir) {
    $argsList += @("--output-dir", $OutputDir)
}

if ($ApiKey) {
    $argsList += @("--api-key", $ApiKey)
}

if ($BaseUrl) {
    $argsList += @("--base-url", $BaseUrl)
}

if ($Model) {
    $argsList += @("--model", $Model)
}

python @argsList
