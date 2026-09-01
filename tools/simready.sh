#!/usr/bin/env bash
# Make sure the simulation is actually usable, restarting whatever is not.
#
# Gazebo segfaults in its Ruby launcher every second or third run in this environment --
# nine crashes captured in one session -- and when it does, `sim start` still reports
# success while the model spawners sit waiting on a world that never appears. Every test
# after that fails for a reason unrelated to the code being tested, which is expensive
# when one grasp run takes five minutes.
#
# Checks the three things a grasp needs, and reports what it had to do about them:
#
#   Gazebo serving models        otherwise there is no ground truth and no physics
#   the depth camera publishing  otherwise perception cannot place a book in 3D
#   move_group planning          otherwise the arm has no collision checking
#
#   tools/simready.sh            check, and restart whatever is broken
#   tools/simready.sh --check    report only
set -uo pipefail

CHECK_ONLY=${1:-}

# Each probe must print exactly one integer. Piping a failing command into `|| echo 0`
# prints its output AND the zero, and the comparisons then fail with "integer expression
# expected" while looking exactly like a dead simulator.
number() { tr -dc '0-9\n' | tail -1 | grep -E '^[0-9]+$' || echo 0; }

books() {
    docker exec erc_sim /entrypoint.sh bash -c \
        "timeout 20 gz model --list 2>/dev/null | grep -c book_col" 2>/dev/null | number
}

depth_live() {
    docker exec erc_sim /entrypoint.sh bash -c \
        "source /opt/erc_ws/install/setup.bash && timeout 14 ros2 topic hz \
         /head_front_camera/head_front_camera/depth/image_rect_raw 2>&1 \
         | grep -c 'average rate'" 2>/dev/null | number
}

planner_up() {
    # Ask whether move_group is answering, not whether it once said it was ready. That
    # log line outlives the process, so grepping for it reported a planner that had been
    # dead for several restarts and let a grasp run start with nothing to plan against.
    docker exec erc_sim /entrypoint.sh bash -c \
        "source /opt/erc_ws/install/setup.bash && timeout 15 ros2 action list 2>/dev/null \
         | grep -c move_action" 2>/dev/null | number
}

start_planner() {
    docker exec erc_sim bash -c "pkill -f move_group; sleep 2" >/dev/null 2>&1
    docker exec -d erc_sim /entrypoint.sh bash -c \
        "source /opt/erc_ws/install/setup.bash && \
         ros2 launch avaa_solution moveit.launch.py > /tmp/movegroup.log 2>&1"
    sleep 26
}

restart_sim() {
    sim stop >/dev/null 2>&1
    sleep 4
    sim start --fast --headless >/dev/null 2>&1
    sleep 40
}

n=$(books)
echo "books visible: $n"
if [ "$n" -lt 20 ]; then
    if [ "$CHECK_ONLY" = "--check" ]; then
        echo "NOT READY: Gazebo is not serving models"
        exit 1
    fi
    echo "restarting Gazebo..."
    restart_sim
    n=$(books)
    echo "books visible after restart: $n"
    if [ "$n" -lt 20 ]; then
        echo "Gazebo will not come up. This one needs 'wsl --shutdown' from Windows;"
        echo "restarting the container alone does not clear it."
        exit 1
    fi
    # A fresh Gazebo means a fresh clock, so move_group has to come up with it.
    start_planner
fi

d=$(depth_live)
if [ "$d" -gt 0 ]; then
    echo "depth camera: publishing"
elif [ "$CHECK_ONLY" = "--check" ]; then
    echo "NOT READY: the depth camera is silent"
    exit 1
else
    echo "depth camera silent; restarting Gazebo for it..."
    restart_sim
    start_planner
    d=$(depth_live)
    [ "$d" -gt 0 ] && echo "depth camera: publishing" || echo "depth camera: STILL SILENT"
fi

p=$(planner_up)
if [ "$p" -eq 0 ]; then
    if [ "$CHECK_ONLY" = "--check" ]; then
        echo "NOT READY: move_group is down"
        exit 1
    fi
    echo "starting move_group..."
    start_planner
    p=$(planner_up)
fi
[ "$p" -gt 0 ] && echo "move_group: planning" || echo "move_group: DOWN"

if [ "$(books)" -ge 20 ] && [ "$p" -gt 0 ] && [ "$d" -gt 0 ]; then
    echo "READY"
else
    echo "NOT READY"
    exit 1
fi
