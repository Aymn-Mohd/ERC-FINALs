"""Forward and inverse kinematics for the TIAGo Pro left arm.

Built directly from the URDF rather than through MoveIt. MoveIt 2 is installed in the
competition image, but there is **no SRDF for this robot** anywhere in it -- only MoveIt's
own test fixtures -- so there is no configured planning group to ask for IK, and authoring
a robot config is not a good use of the remaining time.

The chain from base_link to gripper_left_grasping_link has eight moving joints: the
prismatic torso lift and seven revolute arm joints, every one of them rotating about its
own local Z. That is simple enough to compose directly.

No ROS dependency, so it is unit-testable without a simulator.

    >>> chain = ArmChain.from_urdf(path)
    >>> chain.fk([0.0] * 8)[:3, 3]        # gripper position in base_link
    >>> chain.ik([0.75, 0.2, 1.2])        # joint values reaching that point
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

DEFAULT_URDF = "/opt/erc_ws/src/erc_description/urdf/tiago_pro.urdf"
ROOT_LINK = "base_link"
TIP_LINK = "gripper_left_grasping_link"


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF rpy is fixed-axis: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation about an arbitrary unit axis."""
    a = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


GRAVITY = np.array([0.0, 0.0, -9.81])


@dataclass
class Link:
    """Just enough of a link to work out what it weighs and where that weight is."""

    name: str
    mass: float
    com: np.ndarray          # centre of mass in the link frame


@dataclass
class Joint:
    name: str
    kind: str                 # "revolute", "prismatic" or "fixed"
    origin: np.ndarray        # 4x4, parent -> joint frame at zero
    axis: Optional[np.ndarray]
    lower: float
    upper: float
    child: str = ""
    effort: float = 0.0

    @property
    def moving(self) -> bool:
        return self.kind in ("revolute", "prismatic", "continuous")

    def transform_at(self, value: float) -> np.ndarray:
        if not self.moving or self.axis is None:
            return self.origin
        if self.kind == "prismatic":
            return self.origin @ transform(np.eye(3), self.axis * value)
        return self.origin @ transform(axis_rotation(self.axis, value), np.zeros(3))

    def clamp(self, value: float) -> float:
        return min(self.upper, max(self.lower, value))


class ArmChain:
    def __init__(self, joints: List[Joint], links: Optional[dict] = None):
        self.joints = joints
        self.moving = [j for j in joints if j.moving]
        self.links = links or {}

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_urdf(cls, path: str = DEFAULT_URDF,
                  root: str = ROOT_LINK, tip: str = TIP_LINK) -> "ArmChain":
        tree = ET.parse(path)
        raw = {}
        parent_of = {}
        for element in tree.getroot().findall("joint"):
            name = element.get("name")
            child = element.find("child").get("link")
            origin = element.find("origin")
            xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None
                                      else "0 0 0").split()]
            rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None
                                      else "0 0 0").split()]
            axis_el = element.find("axis")
            axis = ([float(v) for v in axis_el.get("xyz").split()]
                    if axis_el is not None and axis_el.get("xyz") else None)
            limit = element.find("limit")
            has = limit is not None
            lower = (float(limit.get("lower")) if has and limit.get("lower")
                     else -math.pi)
            upper = (float(limit.get("upper")) if has and limit.get("upper")
                     else math.pi)
            raw[name] = Joint(
                name=name,
                kind=element.get("type"),
                origin=transform(rpy_to_matrix(*rpy), np.array(xyz)),
                axis=np.array(axis) if axis else None,
                lower=lower,
                upper=upper,
                effort=(float(limit.get("effort"))
                        if limit is not None and limit.get("effort") else 0.0),
            )
            raw[name].child = child
            parent_of[child] = (name, element.find("parent").get("link"))

        chain: List[Joint] = []
        link = tip
        while link != root:
            entry = parent_of.get(link)
            if entry is None:
                raise ValueError(f"chain from {root} to {tip} is broken at {link}")
            jname, parent_link = entry
            chain.append(raw[jname])
            link = parent_link
        chain.reverse()

        # Link masses, for the static torque estimate. Without them a posture that the
        # arm physically cannot hold looks exactly like one it can: measured on this
        # robot, arm_left_3 and arm_left_4 sit at 25.8 and 25.0 Nm against a 26 Nm limit
        # while reaching into a shelf, and the arm then stops short of every commanded
        # position and sags onto its stops.
        links = {}
        for element in tree.getroot().findall("link"):
            inertial = element.find("inertial")
            if inertial is None:
                continue
            mass_el = inertial.find("mass")
            if mass_el is None or not mass_el.get("value"):
                continue
            origin = inertial.find("origin")
            com = [float(v) for v in (origin.get("xyz", "0 0 0")
                                      if origin is not None else "0 0 0").split()]
            links[element.get("name")] = Link(
                name=element.get("name"),
                mass=float(mass_el.get("value")),
                com=np.array(com))
        return cls(chain, links)

    # ------------------------------------------------------------------ kinematics

    @property
    def joint_names(self) -> List[str]:
        return [j.name for j in self.moving]

    @property
    def limits(self) -> List[tuple]:
        return [(j.lower, j.upper) for j in self.moving]

    def fk(self, values: Sequence[float]) -> np.ndarray:
        """4x4 pose of the grasping link in base_link, for the moving-joint values."""
        if len(values) != len(self.moving):
            raise ValueError(f"expected {len(self.moving)} values, got {len(values)}")
        result = np.eye(4)
        index = 0
        for joint in self.joints:
            if joint.moving:
                result = result @ joint.transform_at(float(values[index]))
                index += 1
            else:
                result = result @ joint.origin
        return result

    def position(self, values: Sequence[float]) -> np.ndarray:
        return self.fk(values)[:3, 3]

    def gravity_torque(self, values: Sequence[float]) -> List[float]:
        """Estimate the torque each moving joint holds against gravity, in Nm.

        A rigid-body sum, no dynamics: for every link out along the chain, the moment its
        weight exerts about each joint upstream of it. That is what matters here, because
        the arm fails while holding still rather than while accelerating.

        Why it exists: reaching into a shelf, arm_left_3 and arm_left_4 were measured at
        25.8 and 25.0 Nm against a 26 Nm limit. Saturated joints stop where they are, so
        the arm stopped 100 mm short of the book, sagged onto its stops, and drifted
        further away on every retry. Nothing in the planner or the kinematics can see
        that; it is not a geometry problem and no amount of replanning fixes it. Postures
        have to be chosen so it does not happen.

        Prismatic joints get the force along their axis instead, in newtons; the torso
        lift is rated 2000 and is nowhere near its limit, so the units being mixed in the
        returned list costs nothing in practice.
        """
        transform_at = np.eye(4)
        origins = []
        axes = []
        masses = []
        centres = []

        for joint in self.joints:
            index = len(origins)
            if joint.moving:
                value = float(values[index]) if index < len(values) else 0.0
                before = transform_at
                transform_at = transform_at @ joint.transform_at(value)
                origins.append(before[:3, 3].copy())
                axes.append((before[:3, :3] @ joint.axis)
                            if joint.axis is not None else np.zeros(3))
            else:
                transform_at = transform_at @ joint.origin
            link = self.links.get(joint.child)
            if link is not None and link.mass > 0.0:
                masses.append(link.mass)
                centres.append((transform_at @ np.append(link.com, 1.0))[:3])

        torques = []
        for position, axis, joint in zip(origins, axes, self.moving):
            total = 0.0
            for mass, centre in zip(masses, centres):
                weight = mass * GRAVITY
                if joint.kind == "prismatic":
                    total += float(np.dot(axis, weight))
                else:
                    total += float(np.dot(axis, np.cross(centre - position, weight)))
            torques.append(abs(total))
        return torques

    def effort_limits(self) -> List[float]:
        """Rated effort per moving joint, from the URDF."""
        return [j.effort for j in self.moving]

    def joint_origins(self, values: Sequence[float]) -> List[np.ndarray]:
        """Where every moving joint sits, in base_link, ending with the tip.

        Enough to tell whether a posture puts the elbow somewhere solid. The arm has four
        spare degrees of freedom, so the solver is free to choose an elbow that reaches
        the right point by going through the shelf, and it did: the same posture commanded
        at the shelf and in open floor came out 5.733 and 0.998 rad from its target.
        """
        if len(values) != len(self.moving):
            raise ValueError(f"expected {len(self.moving)} values, got {len(values)}")
        result = np.eye(4)
        points: List[np.ndarray] = []
        index = 0
        for joint in self.joints:
            if joint.moving:
                result = result @ joint.transform_at(float(values[index]))
                index += 1
                points.append(result[:3, 3].copy())
            else:
                result = result @ joint.origin
        points.append(result[:3, 3].copy())
        return points

    def clamp(self, values: Sequence[float]) -> List[float]:
        return [j.clamp(float(v)) for j, v in zip(self.moving, values)]

    def ik(self, target: Sequence[float], seed: Optional[Sequence[float]] = None,
           tolerance: float = 0.005,
           approach: Optional[Sequence[float]] = None,
           closing: Optional[Sequence[float]] = None,
           orientation_tolerance: float = 0.26,
           prefer=None, pin: Optional[dict] = None) -> Optional[List[float]]:
        """Joint values placing the gripper at target (x, y, z) in base_link.

        With approach and closing left out this solves position only, which is
        fine for waypoints in free space. It is emphatically not fine for a grasp: the
        arm has seven joints for three constraints, so the redundancy gets resolved
        arbitrarily and the wrist will happily arrive at exactly the right point having
        reached around from the far side. Measured on a real shelf target, a
        position-only solve put the approach axis 78 degrees off and the finger travel
        41 degrees off. The hand closed past the corner of the book without touching it
        and every downstream check reported success.

        For a grasp, pass both directions, in base_link:

            approach  where the gripper reaches, its local +x, so [1, 0, 0] to reach
                      straight into a shelf the robot has squared up to
            closing   where the fingers travel, its local +y, so [0, 1, 0] to close
                      across a book standing spine-out

        Those two axes come from the URDF: the fingers sit at y = +/-0.0288 offset
        +0.0756 along z in gripper_left_base_link, and the grasping frame adds a -pi/2
        pitch, which turns that into local x for the approach and local y for the
        finger travel.

        The closing axis is sign-agnostic -- the jaws are symmetric, so a solution with
        the fingers swapped is the same grasp.

        prefer picks between solutions that all satisfy the above. Reaching the
        right point with the right wrist still leaves four degrees of freedom, and the
        solver will spend them on an elbow inside the shelf as readily as beside it, so
        the caller gets to say which posture it wants. It is given the joint values and
        returns a cost to minimise.

        Returns None when nothing reaches within tolerance metres, or when an
        orientation was requested and no solution holds both axes within
        orientation_tolerance radians.
        """
        from scipy.optimize import least_squares

        target = np.asarray(target, dtype=float)
        want_orientation = approach is not None or closing is not None
        approach_v = _unit(approach) if approach is not None else None
        closing_v = _unit(closing) if closing is not None else None

        seed = list(seed) if seed is not None else [
            0.5 * (lo + hi) for lo, hi in self.limits
        ]
        seed = self.clamp(seed)
        lower = [lo for lo, _ in self.limits]
        upper = [hi for _, hi in self.limits]

        # Pinning narrows a joint's bounds instead of leaving the choice to the solver.
        # Expressing a preference through ``prefer`` was not reliable: asked for a book at
        # z=0.731 with the torso ideally at 0.054, the search returned 0.350 because no
        # low-torso solution happened to turn up among its restarts. Some joints are not
        # really free -- the torso is how this robot changes height, and the arm should
        # not be spending its 26 Nm doing it -- and for those, narrowing is honest.
        if pin:
            for index, name in enumerate(self.joint_names):
                if name in pin:
                    wanted, slack = pin[name]
                    lower[index] = max(lower[index], wanted - slack)
                    upper[index] = min(upper[index], wanted + slack)
                    if lower[index] > upper[index]:
                        return None
                    seed[index] = float(np.clip(seed[index],
                                                lower[index], upper[index]))

        # Orientation error is dimensionless where position error is in metres. This
        # scale makes a 10-degree axis error weigh about as much as 9 mm of position
        # error, so the solver squares the wrist up before chasing the last millimetre.
        weight = 0.05

        def residual(values):
            pose = self.fk(values)
            terms = [pose[:3, 3] - target]
            if approach_v is not None:
                terms.append(weight * (pose[:3, 0] - approach_v))
            if closing_v is not None:
                axis = pose[:3, 1]
                sign = 1.0 if float(axis @ closing_v) >= 0.0 else -1.0
                terms.append(weight * (axis - sign * closing_v))
            return np.concatenate(terms)

        def errors(values):
            pose = self.fk(values)
            position = float(np.linalg.norm(pose[:3, 3] - target))
            angle = 0.0
            if approach_v is not None:
                angle = max(angle, _angle_between(pose[:3, 0], approach_v))
            if closing_v is not None:
                axis = pose[:3, 1]
                sign = 1.0 if float(axis @ closing_v) >= 0.0 else -1.0
                angle = max(angle, _angle_between(axis, sign * closing_v))
            return position, angle

        best = None
        acceptable: List[List[float]] = []
        # A few restarts: the arm is redundant and the solver can settle in a local
        # minimum that does not reach, particularly near the limits. Pinning the wrist
        # narrows the basin, so allow more attempts when orientation matters. When the
        # caller wants to choose between postures, keep going to collect several rather
        # than stopping at the first that reaches.
        for attempt in range(40 if prefer is not None else (20 if want_orientation else 6)):
            start = seed if attempt == 0 else [
                np.random.uniform(lo, hi) for lo, hi in zip(lower, upper)
            ]
            try:
                result = least_squares(
                    residual, self.clamp(start), bounds=(lower, upper),
                    xtol=1e-10, ftol=1e-10, max_nfev=2000,
                )
            except Exception:  # noqa: BLE001 - a failed attempt is not fatal
                continue
            position_error, angle_error = errors(result.x)
            # Rank on position, but never prefer a pose whose wrist is out of spec over
            # one that is within it, however close the out-of-spec one reaches.
            key = (angle_error > orientation_tolerance, position_error)
            if best is None or key < best[0]:
                best = (key, list(result.x), position_error, angle_error)
            if position_error <= tolerance and angle_error <= orientation_tolerance:
                acceptable.append(list(result.x))
                if prefer is None:
                    break

        if prefer is not None and acceptable:
            return min(acceptable, key=prefer)
        if best is None:
            return None
        _, values, position_error, angle_error = best
        if position_error > tolerance:
            return None
        if want_orientation and angle_error > orientation_tolerance:
            return None
        return values


def _unit(vector: Sequence[float]) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError("direction vector must be non-zero")
    return v / norm


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in radians between two vectors, neither assumed to be normalised."""
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.acos(max(-1.0, min(1.0, cosine)))
