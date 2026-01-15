#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
SUBFOLDER="${1:-}"  # Optional: specify a subfolder to run configs from

cd "$PROJECT_ROOT"

if [[ -n "$SUBFOLDER" ]]; then
  # Use find for recursive search in specified subfolder
  mapfile -t configs < <(find "$SCRIPT_DIR/$SUBFOLDER" -name "*.yaml" -type f | sort)
else
  configs=("$SCRIPT_DIR"/*/*/*.yaml)
fi

echo "=== Found ${#configs[@]} configs to run ==="
for c in "${configs[@]}"; do
  echo "  ${c#$PROJECT_ROOT/}"
done
echo ""

count=0
for config in "${configs[@]}"; do
  if [[ -f "$config" ]]; then
    rel_config="${config#$PROJECT_ROOT/}"
    echo "=== Running: $rel_config ==="
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" python -m src.probe.main --config "$rel_config" --debug
    ((++count))   # <-- important change
  fi
done

echo "=== Completed $count configs ==="
