#!/usr/bin/env bash
# Regenerate the self-collision matrix in tiago_pro.srdf.
#
# The 802 disable_collisions entries in that file are computed, not authored. Editing them
# by hand is how a planner ends up quietly allowed to drive the arm through the torso, so
# change tiago_pro.base.srdf instead and run this.
#
# Needs the workspace on the search path: the tool resolves package://erc_description mesh
# URIs through ament, and without it aborts with PackageNotFoundError pointing only at
# /opt/ros/humble.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URDF="${URDF:-/opt/erc_ws/src/erc_description/urdf/tiago_pro.urdf}"
UPDATER=/opt/ros/humble/lib/moveit_setup_assistant/collisions_updater

source /opt/ros/humble/setup.bash
source /opt/erc_ws/install/setup.bash

"$UPDATER" \
  --urdf "$URDF" \
  --srdf "$HERE/tiago_pro.base.srdf" \
  --output "$HERE/tiago_pro.srdf" \
  --default --always --trials 10000

echo "wrote $HERE/tiago_pro.srdf with $(grep -c disable_collisions "$HERE/tiago_pro.srdf") pairs"
