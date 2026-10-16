$ErrorActionPreference = "Stop"
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
./bin/aim-windows-amd64.exe analyze --weights checkpoints/sample_262p.pt --out results/sample
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "AIM program: results/sample/learned_program.aim"
