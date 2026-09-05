"""Mission node — the trial sequence, and the only place that talks to the judges.

Owns three things and nothing else:

1. **The scoring topics.** Exactly one node publishes an identification, so there is one
   place to look when a run scores nothing and one place to change when the topic names
   turn out to be different from the specification.
2. **The order of the phases.** Approach, then grasp, then delivery. The controllers do
   not know about each other and must not: each one is startable and debuggable on its
   own, and this decides when each is allowed to run.
3. **The clock.** The trial is timed from launch to the book touching the bin, and the
   tie-break is the fastest completion, so the elapsed time is logged at each transition
   rather than reconstructed afterwards from log timestamps.

Why the controllers are gated rather than left to start themselves
------------------------------------------------------------------
The grasp controller starts as soon as it has a row and a book point, and perception
publishes both the moment the target book is in frame -- which happens several metres
out, while the approach controller is still driving. Left ungated, the arm unfolds into
a shelf the robot has not reached yet. Gating is a `phase` topic rather than a service
call so that any controller can be run by hand against a phase published from the
command line, which is how every one of them has been debugged.

Why identifications are published as soon as they are known
-----------------------------------------------------------
They are worth a point each, plus two more for the annotated images, and none of that
depends on the arm working. Publishing them at the moment perception is confident, and
repeating them for the rest of the trial, means a run that fails at the grasp still
scores what the perception earned. They are republished rather than latched: the judges
read them live, and a latched value from a previous run is worse than no value at all.
"""

from enum import Enum
from typing import Optional

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from std_msgs.msg import Int32, String

try:
    from ros_gz_interfaces.msg import Contacts
except ImportError:  # pragma: no cover - the bridge is not needed to run the node
    Contacts = None

# Monitored live by the organising committee during evaluation.
TOPIC_COLUMN_ID = "/erc/shelf_column_identification"
TOPIC_ROW_ID = "/erc/shelf_row_identification"
TOPIC_BIN_CONTACTS = "/bin_contacts"

# The column's position on the shelf, 1-5, and NOT /avaa/perception/target_column.
# That one is an index among the columns currently in frame -- it is what the steering
# needs and it is meaningless outside the frame it was measured in, reading 0 or 1 while
# the robot drives along the shelf with two markers in view. Publishing it to the judges
# would have forfeited the point while looking like it worked.
TOPIC_PERCEIVED_COLUMN = "/avaa/perception/shelf_column"
TOPIC_PERCEIVED_ROW = "/avaa/perception/target_row"
TOPIC_APPROACH_STATE = "/avaa/approach/state"
TOPIC_GRASP_STATE = "/avaa/grasp/state"
TOPIC_DELIVER_STATE = "/avaa/deliver/state"
TOPIC_PHASE = "/avaa/mission/phase"

VALID_COLOURS = ("red", "blue", "green", "yellow")

SENSOR_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        durability=QoSDurabilityPolicy.VOLATILE,
                        history=QoSHistoryPolicy.KEEP_LAST, depth=1)


class Phase(Enum):
    STARTING = "starting"
    APPROACH = "approach"
    GRASP = "grasp"
    DELIVER = "deliver"
    DONE = "done"
    FAILED = "failed"


class MissionNode(Node):
    def __init__(self) -> None:
        super().__init__("avaa_mission")

        self.declare_parameter("shelf_column_number", 0)
        self.declare_parameter("book_colour", "")
        # How many times running perception has to agree before a digit is published.
        #
        # Perception only offers an answer at all when all five markers are in one frame
        # and their digits are a permutation of 1-5, and it latches the first such
        # reading, so this is a second line of defence rather than the only one. A
        # consistently wrong reading agrees with itself, and counting agreements would
        # not have caught the frame-relative index this topic used to carry.
        #
        # Perception is measured at 16 of 16 books and no false positives over nine
        # viewpoints, so this is not there to fix a reader that gets it wrong. It is
        # there because the first frames arrive while the head is still moving and the
        # robot is still turning, and a digit read off a marker at the edge of a moving
        # frame is a different proposition from the same digit read square on.
        self.declare_parameter("agree_before_publishing", 5)
        self.declare_parameter("republish_hz", 1.0)
        # Let the arm start before the base has finished. Off by default: the two have
        # never been run together and an arm unfolding into a moving base is the most
        # expensive way to find out they disagree.
        self.declare_parameter("grasp_needs_approach", True)
        self.declare_parameter("deliver_after_grasp", True)

        self.target_column = self.get_parameter("shelf_column_number").value
        self.target_colour = str(self.get_parameter("book_colour").value).lower()
        self.agree_needed = int(self.get_parameter("agree_before_publishing").value)
        self.grasp_needs_approach = bool(
            self.get_parameter("grasp_needs_approach").value)
        self.deliver_after_grasp = bool(
            self.get_parameter("deliver_after_grasp").value)

        self.pub_column = self.create_publisher(Int32, TOPIC_COLUMN_ID, 10)
        self.pub_row = self.create_publisher(Int32, TOPIC_ROW_ID, 10)
        self.pub_phase = self.create_publisher(String, TOPIC_PHASE, 10)

        self.column_seen: Optional[int] = None
        self.column_agreed = 0
        self.column_published: Optional[int] = None
        self.row_seen: Optional[int] = None
        self.row_agreed = 0
        self.row_published: Optional[int] = None

        self.approach_state = ""
        self.grasp_state = ""
        self.deliver_state = ""
        self.bin_touched_at: Optional[float] = None

        self.phase = Phase.STARTING
        # Not set here. With use_sim_time the node's clock reads zero until the first
        # /clock message arrives, and a baseline of zero makes the first elapsed time
        # whatever the simulator's uptime happens to be -- this reported "the bin
        # reports contact at 54.5 s" in its own constructor. Set on the first tick that
        # sees a running clock instead.
        self.started_at = None
        self.phase_at = None

        if not self._parameters_valid():
            raise SystemExit(2)

        self.create_subscription(
            Int32, TOPIC_PERCEIVED_COLUMN, self._on_column, 10)
        self.create_subscription(Int32, TOPIC_PERCEIVED_ROW, self._on_row, 10)
        self.create_subscription(
            String, TOPIC_APPROACH_STATE, self._on_approach, 10)
        self.create_subscription(String, TOPIC_GRASP_STATE, self._on_grasp, 10)
        self.create_subscription(String, TOPIC_DELIVER_STATE, self._on_deliver, 10)
        if Contacts is not None:
            self.create_subscription(
                Contacts, TOPIC_BIN_CONTACTS, self._on_bin_contact, SENSOR_QOS)
        else:
            self.get_logger().warn(
                "ros_gz_interfaces is not importable, so %s cannot be watched; the "
                "trial will not detect its own finish" % TOPIC_BIN_CONTACTS)

        period = 1.0 / max(float(self.get_parameter("republish_hz").value), 0.1)
        self.create_timer(period, self._report)
        self.create_timer(0.2, self._tick)

        self.get_logger().info(
            "AVAA mission ready — target column marker %s, book colour '%s'"
            % (self.target_column, self.target_colour))
        self._enter(Phase.APPROACH)

    # ------------------------------------------------------------------ inputs

    def _on_column(self, msg: Int32) -> None:
        value = int(msg.data)
        if value == self.column_seen:
            self.column_agreed += 1
        else:
            self.column_seen, self.column_agreed = value, 1
        if (self.column_published is None
                and self.column_agreed >= self.agree_needed
                and 1 <= value <= 5):
            self.column_published = value
            self.get_logger().info(
                "column identified as %d after %d agreeing readings (%.1f s in)"
                % (value, self.column_agreed, self._elapsed()))

    def _on_row(self, msg: Int32) -> None:
        value = int(msg.data)
        if value == self.row_seen:
            self.row_agreed += 1
        else:
            self.row_seen, self.row_agreed = value, 1
        if (self.row_published is None
                and self.row_agreed >= self.agree_needed
                and 1 <= value <= 4):
            self.row_published = value
            self.get_logger().info(
                "row identified as %d after %d agreeing readings (%.1f s in)"
                % (value, self.row_agreed, self._elapsed()))

    def _on_approach(self, msg: String) -> None:
        self.approach_state = msg.data

    def _on_grasp(self, msg: String) -> None:
        self.grasp_state = msg.data

    def _on_deliver(self, msg: String) -> None:
        self.deliver_state = msg.data

    def _on_bin_contact(self, msg) -> None:
        """Stop the trial clock when a BOOK touches the bin, not when anything does.

        The bin stands on a table and is therefore in contact with it permanently, so
        the sensor reports contacts from the first simulated instant. Taken at face
        value it stopped the trial clock inside this node's own constructor and
        declared the book delivered before the robot had moved. Only a contact whose
        other party is a book counts.
        """
        if self.bin_touched_at is not None:
            return
        for contact in msg.contacts:
            names = (getattr(contact.collision1, "name", ""),
                     getattr(contact.collision2, "name", ""))
            if not any("book" in name for name in names):
                continue
            self.bin_touched_at = self._elapsed()
            self.get_logger().info(
                "a book is in contact with the bin at %.1f s — that is the trial "
                "clock stopped (%s)"
                % (self.bin_touched_at,
                   " / ".join(n for n in names if n)))
            return

    # ------------------------------------------------------------------ helpers

    def _elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return (self.get_clock().now() - self.started_at).nanoseconds / 1e9

    def _clock_started(self) -> bool:
        """Latch the trial start on the first real clock reading."""
        if self.started_at is not None:
            return True
        now = self.get_clock().now()
        if now.nanoseconds == 0:
            return False
        self.started_at = now
        self.phase_at = now
        return True

    def _enter(self, phase: Phase) -> None:
        if phase is self.phase:
            return
        self.get_logger().info(
            "phase %s -> %s at %.1f s" % (self.phase.value, phase.value,
                                          self._elapsed()))
        self.phase = phase
        self.phase_at = self.get_clock().now()
        _ = self.phase_at
        self.pub_phase.publish(String(data=phase.value))

    def _report(self) -> None:
        """Republish whatever has been identified, and the current phase.

        Both go out every second for the whole trial. The judges read the
        identification topics live, so a single publish at the moment of decision can
        be missed by a subscriber that connected a moment later, and there is no cost to
        saying it again.
        """
        self.pub_phase.publish(String(data=self.phase.value))
        if self.column_published is not None:
            self.pub_column.publish(Int32(data=int(self.column_published)))
        if self.row_published is not None:
            self.pub_row.publish(Int32(data=int(self.row_published)))

    # ------------------------------------------------------------------ sequence

    def _tick(self) -> None:
        if not self._clock_started():
            return
        if self.phase is Phase.APPROACH:
            if self.approach_state == "done" or not self.grasp_needs_approach:
                self._enter(Phase.GRASP)
            elif self.approach_state == "failed":
                # Not the end of the trial. The identifications are already published
                # and keep being published, and they are worth points whatever the base
                # did. Only the arm phases are given up on.
                self.get_logger().error(
                    "the approach gave up; the identifications stand but there will be "
                    "no grasp")
                self._enter(Phase.FAILED)
            return

        if self.phase is Phase.GRASP:
            if self.grasp_state == "done":
                if self.deliver_after_grasp:
                    self._enter(Phase.DELIVER)
                else:
                    self._enter(Phase.DONE)
            elif self.grasp_state == "failed":
                self.get_logger().error(
                    "the grasp failed at %.1f s; not attempting a delivery with an "
                    "empty gripper" % self._elapsed())
                self._enter(Phase.FAILED)
            return

        if self.phase is Phase.DELIVER:
            if self.deliver_state == "done":
                self._enter(Phase.DONE)
            elif self.deliver_state == "failed":
                self._enter(Phase.FAILED)
            return

    # ------------------------------------------------------------- for testing

    def report_column(self, column: int) -> None:
        """Publish the identified column. Worth +1 when correct."""
        self.pub_column.publish(Int32(data=int(column)))
        self.get_logger().info("published column identification: %d" % column)

    def report_row(self, row: int) -> None:
        """Publish the identified row. Worth +1 when correct."""
        self.pub_row.publish(Int32(data=int(row)))
        self.get_logger().info("published row identification: %d" % row)

    def _parameters_valid(self) -> bool:
        ok = True
        if not 1 <= int(self.target_column) <= 5:
            self.get_logger().error(
                "shelf_column_number must be 1-5, got %r" % self.target_column)
            ok = False
        if self.target_colour not in VALID_COLOURS:
            self.get_logger().error(
                "book_colour must be one of %s, got %r"
                % (VALID_COLOURS, self.target_colour))
            ok = False
        return ok


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ordinary shutdown (Ctrl-C, or SIGTERM from `ros2 launch`), not a fault.
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
