#!/usr/bin/env python3
"""Ask MoveIt what is colliding, instead of guessing from joint numbers.

Planning fails with "Motion planning start tree could not be initialized", which means the
robot's current state is invalid. Reading joint angles and reasoning about geometry is how
several hours went missing earlier in this project; /check_state_validity answers directly
and names the pair of links involved.
"""
import sys
import time

import rclpy
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetPlanningScene, GetStateValidity
from moveit_msgs.msg import PlanningSceneComponents
from sensor_msgs.msg import JointState

GROUP = "arm_left_torso"


def main():
    rclpy.init()
    node = rclpy.create_node("validity_probe")
    node.set_parameters([rclpy.parameter.Parameter(
        "use_sim_time", rclpy.Parameter.Type.BOOL, True)])

    latest = {}
    node.create_subscription(JointState, "/joint_states",
                             lambda m: latest.__setitem__("js", m), 10)
    deadline = time.time() + 15
    while rclpy.ok() and time.time() < deadline and "js" not in latest:
        rclpy.spin_once(node, timeout_sec=0.2)
    if "js" not in latest:
        print("no joint states")
        return 1

    client = node.create_client(GetStateValidity, "/check_state_validity")
    if not client.wait_for_service(timeout_sec=15.0):
        print("no /check_state_validity")
        return 1

    request = GetStateValidity.Request()
    request.robot_state = RobotState()
    request.robot_state.joint_state = latest["js"]
    request.robot_state.is_diff = False
    request.group_name = GROUP

    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=20.0)
    result = future.result()
    if result is None:
        print("no reply from /check_state_validity")
        return 1

    print("current state valid: %s" % result.valid)
    if result.contacts:
        print("%d contact(s):" % len(result.contacts))
        seen = set()
        for contact in result.contacts:
            pair = (contact.contact_body_1, contact.contact_body_2)
            if pair in seen:
                continue
            seen.add(pair)
            print("   %-38s <-> %s" % pair)
    else:
        print("no contacts reported")

    # What the planner thinks is in the world, so a stale or misplaced box shows up.
    scene_client = node.create_client(GetPlanningScene, "/get_planning_scene")
    if scene_client.wait_for_service(timeout_sec=10.0):
        scene_request = GetPlanningScene.Request()
        scene_request.components.components = (
            PlanningSceneComponents.WORLD_OBJECT_NAMES
            | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY)
        future = scene_client.call_async(scene_request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=20.0)
        scene = future.result()
        if scene is not None:
            print()
            print("objects in the planning scene:")
            for obj in scene.scene.world.collision_objects:
                # MoveIt normalises an object into a pose plus primitives placed relative
                # to it, so the absolute position is obj.pose and primitive_poses are
                # offsets. Reading the wrong one shows every object sitting at the origin.
                where = obj.pose.position
                offset = (obj.primitive_poses[0].position
                          if obj.primitive_poses else None)
                size = (obj.primitives[0].dimensions if obj.primitives else None)
                print("   %-16s frame=%-16s at (%.2f, %.2f, %.2f)%s size %s"
                      % (obj.id, obj.header.frame_id, where.x, where.y, where.z,
                         "" if offset is None or (offset.x == 0 and offset.y == 0
                                                  and offset.z == 0)
                         else " + (%.2f, %.2f, %.2f)" % (offset.x, offset.y, offset.z),
                         [round(v, 2) for v in size] if size else "?"))

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
