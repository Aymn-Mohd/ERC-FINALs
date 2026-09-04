#!/usr/bin/env bash
# Launch the AVAA solution and always save a full console log.
#
# Usage (inside the container, after sourcing):
#   /opt/erc_ws/src/avaa_solution/scripts/run_solution.sh \
#       shelf_column_number:=3 book_colour:=blue
#
# Logs go under src/avaa_solution/logs/ (bind-mounted to the host), so they
# survive container restarts. /tmp/avaa_logs was easy to forget to mkdir and
# vanished with the container.
set -euo pipefail

WS="${WS:-/opt/erc_ws}"
# Prefer the sourced install; fall back to the workspace install.
if [ -f "$WS/install/setup.bash" ]; then
  # shellcheck disable=SC1091
  source "$WS/install/setup.bash"
fi

LOGDIR="${LOGDIR:-$WS/src/avaa_solution/logs}"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/solution_$(date +%Y%m%d_%H%M%S).txt"

echo "[AVAA] logging to $LOG"
echo "[AVAA] args: $*" | tee "$LOG"

# tee keeps the console live and writes every line (stdout + stderr).
exec ros2 launch avaa_solution solution.launch.py "$@" 2>&1 | tee -a "$LOG"
