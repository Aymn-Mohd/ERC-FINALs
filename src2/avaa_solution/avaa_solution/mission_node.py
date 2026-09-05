"""Mission node — skeleton.

Owns the top-level sequence for a challenge trial. Perception, navigation and
manipulation will live in their own nodes; this one holds the state machine and the
two scoring publishers, so there is exactly one place responsible for reporting
identifications to the judges.

Currently a stub: it validates its parameters, advertises the scoring topics, and
reports what it would do. It deliberately does NOT publish identification values yet,
because publishing a wrong column or row is worse than publishing nothing.
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Int32

# Monitored live by the organising committee during evaluation.
TOPIC_COLUMN_ID = "/erc/shelf_column_identification"
TOPIC_ROW_ID = "/erc/shelf_row_identification"

VALID_COLOURS = ("red", "blue", "green", "yellow")


class MissionNode(Node):
    def __init__(self) -> None:
        super().__init__("avaa_mission")

        self.declare_parameter("shelf_column_number", 0)
        self.declare_parameter("book_colour", "")

        self.target_column = self.get_parameter("shelf_column_number").value
        self.target_colour = str(self.get_parameter("book_colour").value).lower()

        # Latching-style QoS is not used: the judges read these live, and a stale
        # retained value could be scored against a later run.
        self.pub_column = self.create_publisher(Int32, TOPIC_COLUMN_ID, 10)
        self.pub_row = self.create_publisher(Int32, TOPIC_ROW_ID, 10)

        if not self._parameters_valid():
            return

        self.get_logger().info(
            f"AVAA mission ready — target column marker {self.target_column}, "
            f"book colour '{self.target_colour}'"
        )
        self.get_logger().warn(
            "Mission logic not implemented yet. This is the package skeleton; "
            "the trial sequence has not been built."
        )

    def _parameters_valid(self) -> bool:
        ok = True
        if not 1 <= int(self.target_column) <= 5:
            self.get_logger().error(
                f"shelf_column_number must be 1-5, got {self.target_column!r}"
            )
            ok = False
        if self.target_colour not in VALID_COLOURS:
            self.get_logger().error(
                f"book_colour must be one of {VALID_COLOURS}, got {self.target_colour!r}"
            )
            ok = False
        return ok

    def report_column(self, column: int) -> None:
        """Publish the identified column. Worth +1 when correct."""
        self.pub_column.publish(Int32(data=int(column)))
        self.get_logger().info(f"published column identification: {column}")

    def report_row(self, row: int) -> None:
        """Publish the identified row. Worth +1 when correct."""
        self.pub_row.publish(Int32(data=int(row)))
        self.get_logger().info(f"published row identification: {row}")


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
