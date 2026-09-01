"""Jaw midpoint versus book, in world coordinates, through a whole grasp."""
import math, subprocess, sys, time
import numpy as np, rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

TIPS=["gripper_left_fingertip_left_link","gripper_left_fingertip_right_link"]
BASE_Z=0.186

def pose(m,attempts=4):
    for _ in range(attempts):
        try:
            out=subprocess.run(["gz","model","-m",m,"-p"],capture_output=True,
                               text=True,timeout=12).stdout
        except Exception: out=""
        ls=[l.strip() for l in out.splitlines()]
        for i,l in enumerate(ls):
            if l.startswith("[") and i+1<len(ls) and ls[i+1].startswith("["):
                return ([float(v) for v in l.strip("[]").split()],
                        [float(v) for v in ls[i+1].strip("[]").split()])
        time.sleep(0.25)
    return None,None

def main():
    book=sys.argv[1] if len(sys.argv)>1 else "book_col_3_row_3_red"
    rclpy.init(); n=rclpy.create_node("jaw_watch")
    n.set_parameters([rclpy.parameter.Parameter("use_sim_time",rclpy.Parameter.Type.BOOL,True)])
    buf=Buffer(); TransformListener(buf,n)
    st={"s":"?","f":float("nan"),"moving":0.0}
    n.create_subscription(String,"/avaa/grasp/state",lambda m: st.__setitem__("s",m.data),10)
    def js(m):
        if "gripper_left_finger_joint" in m.name:
            st["f"]=m.position[m.name.index("gripper_left_finger_joint")]
        # Whether the arm is moving, so rows taken mid-motion can be marked.
        #
        # A row is assembled from a TF lookup and two Gazebo model queries, and the
        # queries cost the better part of a second each. While the arm is swinging that
        # makes the gripper reading up to two seconds older than the book reading, which
        # is not a measurement of anything -- it showed the jaws 250 mm short of a book
        # that was visibly being pushed. Standing still the three agree.
        if m.velocity and len(m.velocity) == len(m.name):
            st["moving"] = max(
                (abs(v) for n, v in zip(m.name, m.velocity)
                 if n.startswith("arm_left") or n == "torso_lift_joint"), default=0.0)
    n.create_subscription(JointState,"/joint_states",js,10)
    t=time.time()
    while time.time()-t<8: rclpy.spin_once(n,timeout_sec=0.1)
    print("%-6s %-11s %-24s %-24s %8s %8s %8s %8s"%(
        "t","state","jaw mid (world)","book (world)","dx_mm","dy_mm","dz_mm","finger"))
    stop=time.time()+700; last=None
    while rclpy.ok() and time.time()<stop:
        rclpy.spin_once(n,timeout_sec=0.05)
        try:
            tips=[]
            for l in TIPS:
                tr=buf.lookup_transform("base_link",l,rclpy.time.Time()).transform.translation
                tips.append(np.array([tr.x,tr.y,tr.z]))
        except Exception:
            time.sleep(1.0); continue
        mid=(tips[0]+tips[1])/2.0
        robot,rpy=pose("tiago_pro"); bk,_=pose(book)
        if robot is None or bk is None: continue
        yaw=rpy[2]
        wx=robot[0]+mid[0]*math.cos(yaw)-mid[1]*math.sin(yaw)
        wy=robot[1]+mid[0]*math.sin(yaw)+mid[1]*math.cos(yaw)
        wz=mid[2]+BASE_Z
        row=(st["s"],round(wy,3))
        if row!=last:
            last=row
            mark = " ~moving" if st.get("moving", 0.0) > 0.02 else ""
            print("%-6.0f %-11s (%.3f,%+.3f,%.3f)   (%.3f,%+.3f,%.3f)   %+8.0f %+8.0f %+8.0f %8.4f%s"%(
                time.time()%10000,st["s"],wx,wy,wz,bk[0],bk[1],bk[2],
                (wx-bk[0])*1000,(wy-bk[1])*1000,(wz-bk[2])*1000,st["f"],mark))
            sys.stdout.flush()
        if st["s"] in ("done","failed"): break
        time.sleep(4.0)
    n.destroy_node(); rclpy.shutdown()
main()
