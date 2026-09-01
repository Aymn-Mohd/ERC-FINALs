#!/usr/bin/env python3
"""Record the head camera through a run, with the grasp state burned in.

The Gazebo GUI cannot render in this container: Qt's OpenGL context creation
fails outright, and forcing software GL starved the server until it died. The
headless server still renders sensors through EGL, so the robot's own camera is
the reliable way to watch a run.
"""
import sys, time
import numpy as np, cv2, rclpy
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                       QoSHistoryPolicy, QoSDurabilityPolicy)
from sensor_msgs.msg import Image
from std_msgs.msg import String

TOPIC = '/head_front_camera/head_front_camera/color/image_raw'
OUT = '/tmp/run.mp4'
SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0

# Camera images are published best effort. A reliable subscription silently
# receives nothing at all, which is exactly what it did: zero frames in 20 s.
SENSOR_QOS = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        durability=QoSDurabilityPolicy.VOLATILE,
                        history=QoSHistoryPolicy.KEEP_LAST, depth=1)

rclpy.init()
n = rclpy.create_node('recorder')
n.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
state = {'s': 'waiting', 'n': 0, 'w': None}
n.create_subscription(String, '/avaa/grasp/state',
                      lambda m: state.__setitem__('s', m.data), 10)
began = time.time()
last = [0.0]

def on_image(msg):
    now = time.time()
    if now - last[0] < 0.10:
        return
    last[0] = now
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    try:
        img = buf.reshape(msg.height, msg.width, -1)
    except ValueError:
        return
    if msg.encoding in ('rgb8', 'rgba8'):
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
    else:
        img = np.ascontiguousarray(img[:, :, :3])
    label = '%-11s  t+%3.0fs' % (state['s'], now - began)
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    if state['w'] is None:
        state['w'] = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'),
                                     15.0, (img.shape[1], img.shape[0]))
        print('recording %dx%d -> %s' % (img.shape[1], img.shape[0], OUT), flush=True)
    state['w'].write(img)
    state['n'] += 1

n.create_subscription(Image, TOPIC, on_image, SENSOR_QOS)
while rclpy.ok() and time.time() - began < SECONDS:
    rclpy.spin_once(n, timeout_sec=0.1)
    if state['s'] in ('done', 'failed') and state['n'] > 40:
        t = time.time()
        while time.time() - t < 6:
            rclpy.spin_once(n, timeout_sec=0.1)
        break
if state['w'] is not None:
    state['w'].release()
print('wrote %d frames' % state['n'], flush=True)
n.destroy_node(); rclpy.shutdown()
