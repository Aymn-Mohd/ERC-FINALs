"""Is the base drift constant, decaying, or caused by the arm moving?"""
import math, subprocess, time, rclpy
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM=["arm_left_%d_joint"%i for i in range(1,8)]
TUCK=[2.1521,0.3824,1.2785,-2.1517,0.8325,0.1926,1.3944]
REACH=[3.14,-2.30,-0.68,-2.07,1.57,-0.28,-1.80]

def pose():
    for _ in range(6):
        out=subprocess.run(["gz","model","-m","tiago_pro","-p"],capture_output=True,text=True,timeout=15).stdout
        ls=[l.strip() for l in out.splitlines()]
        for i,l in enumerate(ls):
            if l.startswith("[") and i+1<len(ls) and ls[i+1].startswith("["):
                return ([float(v) for v in l.strip("[]").split()],
                        [float(v) for v in ls[i+1].strip("[]").split()])
        time.sleep(0.3)
    return None,None
def wrap(a):
    while a>math.pi: a-=2*math.pi
    while a<-math.pi: a+=2*math.pi
    return a

rclpy.init(); n=rclpy.create_node("yawtrack")
pub=n.create_publisher(JointTrajectory,"/arm_left_controller/joint_trajectory",10)
t=time.time()
while time.time()-t<5: rclpy.spin_once(n,timeout_sec=0.1)
def send(vals,secs):
    tr=JointTrajectory(); tr.joint_names=ARM
    p=JointTrajectoryPoint(); p.positions=[float(v) for v in vals]
    p.time_from_start=Duration(sec=secs); tr.points=[p]; pub.publish(tr)

def watch(label, secs=60):
    p0,r0=pose(); t0=time.time(); last=(p0,r0)
    while time.time()-t0<secs:
        time.sleep(20)
        p,r=pose()
        if p is None: continue
        print("  %-14s +%2.0fs  moved %5.1f mm  yaw %+6.2f deg  (rate %+.2f deg/s)"%(
            label, time.time()-t0, math.dist(p[:2],p0[:2])*1000,
            math.degrees(wrap(r[2]-r0[2])),
            math.degrees(wrap(r[2]-last[1][2]))/20.0))
        last=(p,r)

print("arm tucked, nothing commanded:")
watch("idle")
print("moving the arm out to a reach posture and back, twice:")
t0=time.time()
for i in range(2):
    send(REACH,8); time.sleep(22)
    send(TUCK,8);  time.sleep(22)
p,r=pose()
print("  after two full arm swings: see next window")
watch("after arm")
n.destroy_node(); rclpy.shutdown()
