#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

echo "──────────────────────────────────────────"
echo "  FRE Task 4 — ROS 2 Humble Container"
echo "──────────────────────────────────────────"

# Auto-build if there are ROS packages in src/
if ls /ros2_ws/src/*/package.xml &>/dev/null; then
    echo "[+] Packages found in /ros2_ws/src — building workspace..."
    cd /ros2_ws
    colcon build --symlink-install
    source /ros2_ws/install/setup.bash
    echo "[+] Build complete."
else
    echo "[!] No packages in /ros2_ws/src yet — skipping build."
    echo "    Mount your code and run: colcon build --symlink-install"
fi

echo ""
echo "Bags folder : /bags"
echo "Workspace   : /ros2_ws/src"
echo ""

# Run whatever command was passed (default: bash)
exec "$@"