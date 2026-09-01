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
| Grasp controller | ⚠️ **written and unit-tested, never run end to end** |
| Place in bin | ❌ not started |
| Video (D2) | ❌ not started |
| Report (D3) | ❌ not started |

**80 unit tests**, no simulator required:

```bash
sim shell
cd /opt/erc_ws/src/avaa_solution && python3 -m pytest test/ -q
```

---

## The next step

**Get one clean end-to-end pick.** Everything upstream is measured; what has never happened
is the whole chain running through in a single trial.

### Exactly where it stops (as of 2026-08-27)

The approach now works: it tucks the arms, searches for the marker from the spawn pose,
centres, drives in to 0.80 m, and squares up to within about 2 degrees. That part is
repeatable.

What fails is the last step before grasping — **the target book is not reliably in frame at
grasping range**, so the book point is never published and the grasp controller sits in
IDLE waiting for a target it never gets.

Two things are in tension, and this is the crux:

- **The camera must see the book** to locate it, which wants distance and a level head.
- **The arm must reach the book**, which wants closeness — the reachability map loses a
  corner of the envelope beyond 0.85 m.

Tilting the head down keeps the books in frame but pushes the *markers* out, which is what
the robot had been steering by. Steering now switches to the book itself once it is seen,
but the handover is not yet reliable: if the robot drifts before the book is acquired, it
can arrive at the shelf's end upright with nothing recognisable in view.

Ideas not yet tried, roughly in order of promise:

1. **Acquire the book before closing in.** Stop at ~1.5 m, tilt the head, confirm the book
   is seen and centred, and only then drive the final stretch. Turns a handover mid-drive
   into a checkpoint.
2. **Use the depth point rather than the image bearing for the final metre.** The book's
   3D position is already published and is good to 15-35 mm in x and y; driving to a pose
   relative to that is stronger than centring pixels.
3. **Head pan.** `head_1_joint` has +/-75 degrees and is unused. The robot could keep the
   book in view while the base is not perfectly aligned.

```bash
sim start --fast --headless
# then, in the container, with use_sim_time on every node:
ros2 run avaa_solution perception --ros-args -p use_sim_time:=true \
    -p shelf_column_number:=3 -p book_colour:=blue
ros2 run avaa_solution approach --ros-args -p use_sim_time:=true
ros2 run avaa_solution grasp    --ros-args -p use_sim_time:=true
```

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
