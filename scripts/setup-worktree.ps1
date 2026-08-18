param(
    [string]$Main = "",
    [switch]$Force
)

if (-not $Main) {
    $Main = (git worktree list --porcelain |
        Select-String "^worktree " |
        Select-Object -First 1).Line -replace "^worktree ", ""
}

$Target = git rev-parse --show-toplevel

if ((Test-Path "$Target\.env") -and -not $Force) {
    throw "$Target\.env 已存在；要覆盖就加 -Force"
}

Copy-Item "$Main\.env" "$Target\.env" -Force:$Force

# 后续要共享 node_modules 再打开：
# Copy-Item "$Main\node_modules" "$Target\node_modules" -Recurse
# 或者：npm ci
