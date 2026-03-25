#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOURCES_DIR="${SCRIPT_DIR}/resources"
ACTIVESPLAT_ROOT="${ACTIVESPLAT_ROOT:-/opt/activesplat_ws/src/ActiveSplat}"

copy_activesplat_file() {
  local relative_path="$1"
  install -D -m 0644 \
    "${RESOURCES_DIR}/ActiveSplat/${relative_path}" \
    "${ACTIVESPLAT_ROOT}/${relative_path}"
  echo "Restored ActiveSplat file: ${relative_path}"
}

if [[ ! -d "${RESOURCES_DIR}" ]]; then
  echo "Resources directory not found: ${RESOURCES_DIR}" >&2
  exit 1
fi

if [[ ! -d "${ACTIVESPLAT_ROOT}" ]]; then
  echo "ActiveSplat root not found: ${ACTIVESPLAT_ROOT}" >&2
  echo "Set ACTIVESPLAT_ROOT to the restored project location and run again." >&2
  exit 1
fi

copy_activesplat_file "launch/habitat.launch"
copy_activesplat_file "scripts/nodes/mapper_node.py"
copy_activesplat_file "src/visualizer/visualizer.py"
copy_activesplat_file "src/dataloader/__init__.py"
copy_activesplat_file "src/dataloader/dataloader.py"
copy_activesplat_file "config/.templates/user_config.json"
copy_activesplat_file "config/user_config.json"
copy_activesplat_file "config/datasets/replica.json"
copy_activesplat_file "srv/GetDatasetConfig.srv"

# ROS launch expects type="*.py" nodes to be executable
chmod +x "${ACTIVESPLAT_ROOT}/scripts/nodes/mapper_node.py"
chmod +x "${ACTIVESPLAT_ROOT}/scripts/nodes/planner_node.py"

echo
echo "Restore complete."
echo "ActiveSplat root : ${ACTIVESPLAT_ROOT}"
echo
echo "Next steps:"
echo "1. Verify ${ACTIVESPLAT_ROOT}/config/user_config.json"
echo "2. Ensure Replica data exists at /workspace/datasets/replica"
echo "3. Launch roslaunch with config/datasets/replica.json"
