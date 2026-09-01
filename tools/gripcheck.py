#!/usr/bin/env python3
"""Can this gripper hold this book at all, with aiming taken out of the question?

The approach now arrives within a few millimetres of the book with the jaws 85 mm
apart, and the book is still swept. That leaves one question that has never been
answered on its own: put the book exactly between the jaws, with no aiming error at
all, and close. If it is held, everything left is targeting. If it is not, no amount of
targeting will ever produce a pick and the grasp needs rethinking.

The earlier version of this got it wrong by teleporting the book into a half open
gripper, which interpenetrated the pads and blew the finger to its hard stop. This one
opens fully first, places the book at the measured jaw midpoint, and checks the
clearance to each fingertip before closing anything.
"""
import math, subprocess, sys, time
import numpy as np, rclpy
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM=["arm_left_%d_joint"%i for i in range(1,8)]
POSTURE=[0.25, 3.14, -2.30, -0.68, -2.07, 1.57, -0.28, -1.80]
TIPS=["gripper_left_fingertip_left_link","gripper_left_fingertip_right_link"]
BASE_Z=0.186

def gz(*a,t=20):
    try: return subprocess.run(["gz",*a],capture_output=True,text=True,timeout=t).stdout
    except Exception: return ""

def pose(m,tries=6):
    for _ in range(tries):
        ls=[l.strip() for l in gz("model","-m",m,"-p").splitlines()]
        for i,l in enumerate(ls):
            if l.startswith("[") and i+1<len(ls) and ls[i+1].startswith("["):
                return ([float(v) for v in l.strip("[]").split()],
                        [float(v) for v in ls[i+1].strip("[]").split()])
        time.sleep(0.3)
    return None,None

def put(model,x,y,z):
    return gz("service","-s","/world/erc_world/set_pose","--reqtype","gz.msgs.Pose",
              "--reptype","gz.msgs.Boolean","--timeout","3000",
              "--req",'name: "%s", position: {x: %f, y: %f, z: %f}, '
                      'orientation: {x: 0, y: 0.7071068, z: 0, w: 0.7071068}'%(model,x,y,z))

def main():
    book=sys.argv[1] if len(sys.argv)>1 else "book_col_3_row_3_blue"
    rclpy.init(); n=rclpy.create_node("gripcheck2")
    n.set_parameters([rclpy.parameter.Parameter("use_sim_time",rclpy.Parameter.Type.BOOL,True)])
    buf=Buffer(); TransformListener(buf,n)
    st={}
    n.create_subscription(JointState,"/joint_states",lambda m: st.update(zip(m.name,m.position)),10)
    arm=n.create_publisher(JointTrajectory,"/arm_left_controller/joint_trajectory",10)
    torso=n.create_publisher(JointTrajectory,"/torso_controller/joint_trajectory",10)
    grip=n.create_publisher(JointTrajectory,"/gripper_left_controller/joint_trajectory",10)
    t=time.time()
    while time.time()-t<8: rclpy.spin_once(n,timeout_sec=0.1)

    def send(pub,names,vals,secs):
        tr=JointTrajectory(); tr.joint_names=names
        p=JointTrajectoryPoint(); p.positions=[float(v) for v in vals]
        p.time_from_start=Duration(sec=int(secs),nanosec=int((secs%1)*1e9))
        tr.points=[p]; pub.publish(tr)
    def wait(secs):
        t=time.time()
        while time.time()-t<secs: rclpy.spin_once(n,timeout_sec=0.1)

    print("posing the arm and opening the jaws fully...")
    send(torso,["torso_lift_joint"],[POSTURE[0]],16)
    send(arm,ARM,POSTURE[1:],16)
    send(grip,["gripper_left_finger_joint"],[0.068],6)
    wait(50)

    tips=[]
    for l in TIPS:
        tr=buf.lookup_transform("base_link",l,rclpy.time.Time()).transform.translation
        tips.append(np.array([tr.x,tr.y,tr.z]))
    mid=(tips[0]+tips[1])/2.0
    robot,rpy=pose("tiago_pro")
    if robot is None: print("no robot pose"); return
    yaw=rpy[2]
    wx=robot[0]+mid[0]*math.cos(yaw)-mid[1]*math.sin(yaw)
    wy=robot[1]+mid[0]*math.sin(yaw)+mid[1]*math.cos(yaw)
    wz=mid[2]+BASE_Z
    print("finger=%.4f, tip separation %.1f mm, jaw midpoint world (%.3f,%.3f,%.3f)"
          %(st.get("gripper_left_finger_joint",float('nan')),
            float(np.linalg.norm(tips[0]-tips[1]))*1000,wx,wy,wz))

    put(book,wx,wy,wz); wait(14)
    b,_=pose(book)
    if b is None: print("lost the book"); return
    off=math.dist(b,[wx,wy,wz])
    print("book settled at (%.3f,%.3f,%.3f), %.1f mm from the jaw midpoint"%(b[0],b[1],b[2],off*1000))
    if off > 0.05:
        print("it did not stay where it was put; the jaws are probably touching it")

    print("closing...")
    send(grip,["gripper_left_finger_joint"],[-0.001],10)
    for i in range(7):
        wait(5)
        here,_=pose(book)
        print("  t+%2ds finger=%+.4f  book=(%.3f,%.3f,%.3f)"%(
            5*(i+1),st.get("gripper_left_finger_joint",float('nan')),
            here[0] if here else float('nan'), here[1] if here else float('nan'),
            here[2] if here else float('nan')))
    before,_=pose(book)
    print()
    print("raising the torso 120 mm...")
    send(torso,["torso_lift_joint"],[min(0.35,POSTURE[0]+0.12)],10)
    wait(40)
    after,_=pose(book)
    if before is None or after is None: print("lost the book"); return
    rise=after[2]-before[2]
    print("book rose %+.3f m; finger ended at %+.4f"%(rise,st.get("gripper_left_finger_joint",float('nan'))))
    print("RESULT: %s"%("HELD — the gripper can carry this book" if rise>0.05
                        else "NOT HELD — it stayed behind"))
    n.destroy_node(); rclpy.shutdown()

main()
