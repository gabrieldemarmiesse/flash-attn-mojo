#!/usr/bin/env bash
# Print headline metrics from a saved ncu .ncu-rep file.
#
# By default emits the `details` page (Speed Of Light, Memory, Compute,
# Occupancy, Scheduler, etc., for each captured kernel). Override with
# extra page names — common picks:
#   details             everything (default)
#   raw                 raw metric dump
#   source              SASS/PTX with metric overlay (only useful in ncu-ui)
#
# Usage:
#   scripts/profile_summary.sh <report.ncu-rep> [pages...]
#
# Examples:
#   scripts/profile_summary.sh /tmp/bwd_prof.ncu-rep
#   scripts/profile_summary.sh /tmp/bwd_prof.ncu-rep raw
set -euo pipefail

REPORT="${1:-}"
if [[ -z "$REPORT" ]]; then
    echo "usage: $(basename "$0") <report.ncu-rep> [page...]" >&2
    exit 2
fi
shift || true

PAGES=("$@")
if [[ ${#PAGES[@]} -eq 0 ]]; then
    PAGES=(details)
fi

PIXI="$(command -v pixi || true)"
if [[ -z "$PIXI" ]]; then
    echo "error: pixi not found on PATH" >&2
    exit 1
fi

for page in "${PAGES[@]}"; do
    echo "==== page: $page ===="
    "$PIXI" exec --spec nsight-compute=2024.3.2 -- ncu \
        --import "$REPORT" --page "$page"
done
