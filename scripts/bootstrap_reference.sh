#!/usr/bin/env bash
# Step 3: clone the Roboflow Sports reference at the pinned commit, apply the headless patch,
# and download its demo weights/videos (Google Drive via gdown; needs normal internet access).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIN="$(cat "$ROOT/locks/roboflow-sports-commit.txt")"
DEST="$ROOT/third_party/roboflow-sports"
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/roboflow/sports.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$PIN"
if git -C "$DEST" apply --check "$ROOT/docs/patches/roboflow-sports-headless.patch" 2>/dev/null; then
  git -C "$DEST" apply "$ROOT/docs/patches/roboflow-sports-headless.patch"
  echo "Applied headless patch (--preview is now opt-in)."
fi
pip install -e "$DEST"
pip install gdown
bash "$DEST/examples/soccer/setup.sh"
sha256sum "$DEST"/examples/soccer/data/*.pt | tee "$ROOT/locks/roboflow-weights.sha256"
echo "Pinned roboflow/sports at $PIN"
