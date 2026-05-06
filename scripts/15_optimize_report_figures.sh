#!/usr/bin/env bash
set -euo pipefail

# Downscale report PNGs for faster Overleaf compiles.
# Usage:
#   bash scripts/15_optimize_report_figures.sh
#   bash scripts/15_optimize_report_figures.sh report/figures 1200

FIG_DIR="${1:-report/figures}"
MAX_SIDE="${2:-1200}"

if ! command -v sips >/dev/null 2>&1; then
  echo "ERROR: sips is required on macOS to optimize report figures."
  exit 1
fi

if [[ ! -d "$FIG_DIR" ]]; then
  echo "ERROR: figure directory not found: $FIG_DIR"
  exit 1
fi

echo "Optimizing PNGs in $FIG_DIR with max side ${MAX_SIDE}px"

find "$FIG_DIR" -maxdepth 1 -type f -name '*.png' | while IFS= read -r file; do
  width="$(sips -g pixelWidth "$file" 2>/dev/null | awk '/pixelWidth:/ {print $2}')"
  height="$(sips -g pixelHeight "$file" 2>/dev/null | awk '/pixelHeight:/ {print $2}')"

  if [[ -z "$width" || -z "$height" ]]; then
    echo "Skipping unreadable file: $file"
    continue
  fi

  if (( width > MAX_SIDE || height > MAX_SIDE )); then
    echo "  resizing $(basename "$file") from ${width}x${height}"
    sips -Z "$MAX_SIDE" "$file" >/dev/null
  fi
done

echo "Done."
