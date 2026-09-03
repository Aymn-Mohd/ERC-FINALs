# Team AVAA — project state

**Read this first when resuming.** Written 2026-08-27. Phase 1 deadline **2026-09-15**.

Nothing important lives in a chat transcript. Everything is in this folder, in the repo, or
in the git history — which carries the reasoning, not just the diffs.

---

## Repository

`https://github.com/AIsmail17/avaa-erc-2026` — branch **`avaa`**, private.
Local: `~/erc/erc_sim_2026` inside WSL (physically on D: via the VHDX).

The commit messages are the design record. `git log` is worth reading before changing
anything, because most non-obvious decisions have a measurement behind them.

## Documents in this folder

| File | What it holds |
|---|---|
| `NOTES.md` | Competition rules, scoring, deliverables |
| `SETUP.md` | Environment build, with every trap hit along the way |
| `PERCEPTION.md` | Colour and marker detection, measured accuracy, 3D localisation |
| `MANIPULATION.md` | Gripper curve, reach envelope, arm kinematics, tuck pose |
| `ORGANISER_QUESTIONS.md` | Four questions; all answered internally, none confirmed by the committee |
| `STATE.md` | This file |

---

## Where the work stands

| Piece | State |
|---|---|
| Environment | ✅ WSL 2.9.9 + Docker + GPU, RTF ~0.5 |
| Perception — column marker (1–5) | ✅ verified, 9 viewpoints, no misreads |
| Perception — book colour and row | ✅ verified, 16/16 books, 0 false positives |
| Scoring topics + annotated images | ✅ both written, timestamped |
| Nav2 gross navigation | ✅ works to ~0.35 m tolerance |
| Fine approach controller | ⚠️ **works sometimes** — not reliable |
| Arm kinematics + IK | ✅ exact to 0.7 mm; all four rows reachable |
| Grasp controller | ✅ **restored** into the tree and launch; offline geometry/IK tests green; live pick still to prove |
| Place in bin | ❌ not started |
| Video (D2) | ❌ not started |
| Report (D3) | ❌ not started |

Grasp was stripped on 2026-09-03 (`3e2e8cc`) for a movement-only phase and restored from
`3e2e8cc^`. `solution.launch.py` now starts `move_group`, perception, approach
(`return_to_bin:=false` so VERIFY hands off instead of driving to the bin), grasp, and
mission.

**Unit tests** (grasp geometry + arm chain + orientation). In the container:

```bash
sim shell
cd /opt/erc_ws/src/avaa_solution && python3 -m pytest \
    test/test_grasp_geometry.py test/test_arm_chain.py test/test_arm_orientation.py -q
```

---

## The next step

**Prove one clean pick at a healthy real-time factor**, then trust the full trial.

### 1. Isolated pick (skip flaky approach) — do this first

Same idea as ROSALYA / TIAGo Pro mobile manipulation: get the base in front of the object,
then let the arm + MoveIt do the pick. `tools/ideal_grasp.py` feeds Gazebo ground-truth
book pose into the real grasp controller so mechanics are tested without perception.

```bash
sim start --fast --headless
# confirm RTF is usable before believing any result
#   docker exec erc_sim gz topic -e -t /stats
tools/simready.sh                  # Gazebo models + depth + move_group
# park the base in front of a shelf column (existing place_robot / approach), then:
tools/in-sim ideal_grasp.py red
```

Success: book leaves the shelf; `/avaa/grasp/state` reaches `done`; RTF stays well above
the ~0.03 GUI death zone.

### 2. Full perception → approach → grasp

```bash
sim start --fast --headless
tools/simready.sh
# organisers' entry point already includes move_group + grasp:
ros2 launch avaa_solution solution.launch.py \
    shelf_column_number:=3 book_colour:=blue
```

### Approach still flaky at grasping range

What still fails on real perception: **the target book is not always in frame at grasping
range**, so the book point is never published and grasp sits in IDLE. Acquire-at-1.5 m,
depth-point final metre, and head pan remain the next approach fixes — only after an
ideal pick works.

Two or three clean picks before trusting it. After that: placing in the bin (+2 dropped,
**+4 gently placed** — the largest single scoring item in the task), then the video and
report.

---

## What will bite you

- **`use_sim_time:=true` on every node.** Gazebo stamps TF with `/clock`, hours behind wall
  time. Without it tf2 floods with `TF_OLD_DATA` and lookups silently return nothing.
- **Gazebo dies unpredictably.** WSL GPU device removal, roughly every 10–30 minutes under
  load. `wsl --shutdown`, then recreate the container with `./docker/up-wsl.sh` — a
  container *restart* is not enough, its mounts go stale.
- **`sim start --fast --headless` is much more stable** than with the GUI. Use the GUI only
  to watch or to record.
- **Recreating the container wipes `install/` and `build/`.** Expected; rebuild takes 15 s.
- **Re-publishing a `JointTrajectory` restarts it**, resetting `time_from_start`. A
  trajectory re-sent on a timer never completes.
- **The base will not stay where it is put.** The wheels are mecanum, modelled with
  `mu 0.8` along the roller axis and `mu2 0.0` across it, so nothing damps a sideways
  drift. Untouched with the arm folded: 8 mm and 2° every 30 s. With the arm extended
  into the shelf, far worse. Commanding zero `cmd_vel` does not help — that locks the
  wheels and the base slides across them.
- **`/odom` is wheel odometry and it lies.** A base sliding on locked wheels reports
  nothing; a base held still by driving its wheels reports travel that never happened
  (813 mm of it in one run held to 17 mm of true error). Never anchor a target in it.
- **Run headless.** With the Gazebo GUI the real time factor collapsed to 0.034 and
  nothing moved at all — trajectories "completed" while the arm was still folded. Check
  it with `gz topic -e -t /stats` before believing any result.
- **Polling Gazebo costs real time factor.** Each `gz model -p` spawns a process and hits
  Gazebo's service thread. Three tools polling at once took the simulation from 0.65 to
  0.008. Throttle every fixture.
- **Don't read gripper state from `/joint_states`** — the linkage joints are not published
  there. Use TF.
- **The stowed arm sits in the LiDAR plane.** Returns within 0.45 m of `base_footprint` are
  the robot seeing itself.

## Known-unresolved

1. **Depth has a systematic vertical bias of ~+0.108 m.** Occlusion, model origin and
   frame mismatch all ruled out. Worked around by taking height from the identified row
   and only x/y from depth. Worth revisiting if it shows up elsewhere.
2. **The fine approach is not reliable.** It has completed a full run (2.95 m → 0.82 m,
   squared up) and it has also crept and stalled. Suspect the target marker drifting to
   the frame edge and the yaw correction dominating forward drive.
3. **Nav2's DWB stops ~0.3 m short** of any goal and will not close further. Not
   CPU-related — measured with zero control-loop rate misses. Hence the separate approach
   controller.

## Decisions taken, and why

- **MoveIt, with an SRDF written by hand.** The image ships no SRDF for this robot
  despite the Phase 1 document saying MoveIt is "configured", so `src/avaa_solution/
  moveit/` carries one, generated collision matrix and all. Analytic IK from the URDF
  proposes postures; MoveIt validates and executes them. IK alone repeatedly reached
  correct points by paths that went through the shelf.
- **No lateral base motion.** Commanding pure `vy` yaws the base by roughly the magnitude
  it strafes. Rotate-then-drive avoids it.
- **Map-less Nav2 in the `odom` frame.** No prior map, start pose not guaranteed, a trial
  travels a few metres so drift stays well under the clearances.
- **Row numbering is a launch parameter** (`rows_top_down`), not a baked-in assumption,
  because the rules never state which end row 1 is.
