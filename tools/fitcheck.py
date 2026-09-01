"""Does a book-sized box at the grasp point actually fit between the pads?

Everything so far has reasoned about a "jaw gap" taken from fingertip link origins or
from the extremes of the pad meshes. Neither is the question. The question is whether
the volume the book occupies, placed where the IK aims, is free of pad geometry.
"""
import glob, struct, time
import numpy as np, rclpy
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

GRASP="gripper_left_grasping_link"
TIPS=["gripper_left_fingertip_left_link","gripper_left_fingertip_right_link"]
# book half extents along (approach into shelf, closing, up)
HALF=np.array([0.080, 0.015, 0.125])

def verts():
    p=glob.glob("/opt/erc_ws/install/**/meshes/fingertip.stl",recursive=True)[0]
    d=open(p,"rb").read(); cnt=struct.unpack("<I",d[80:84])[0]; off=84; out=[]
    for _ in range(cnt):
        v=struct.unpack("<12fH",d[off:off+50]); off+=50
        for k in range(3): out.append(v[3+3*k:6+3*k])
    return np.array(out)

def mat(tf):
    q=tf.transform.rotation; t=tf.transform.translation
    xx,yy,zz=q.x*q.x,q.y*q.y,q.z*q.z; xy,xz,yz=q.x*q.y,q.x*q.z,q.y*q.z
    wx,wy,wz=q.w*q.x,q.w*q.y,q.w*q.z
    return (np.array([[1-2*(yy+zz),2*(xy-wz),2*(xz+wy)],
                      [2*(xy+wz),1-2*(xx+zz),2*(yz-wx)],
                      [2*(xz-wy),2*(yz+wx),1-2*(xx+yy)]]),
            np.array([t.x,t.y,t.z]))

V=verts()
rclpy.init(); n=rclpy.create_node("fitcheck")
n.set_parameters([rclpy.parameter.Parameter("use_sim_time",rclpy.Parameter.Type.BOOL,True)])
buf=Buffer(); TransformListener(buf,n)
st={}
n.create_subscription(JointState,"/joint_states",lambda m: st.update(zip(m.name,m.position)),10)
pub=n.create_publisher(JointTrajectory,"/gripper_left_controller/joint_trajectory",10)
t=time.time()
while time.time()-t<8: rclpy.spin_once(n,timeout_sec=0.1)

def send(v,secs=6):
    tr=JointTrajectory(); tr.joint_names=["gripper_left_finger_joint"]
    p=JointTrajectoryPoint(); p.positions=[float(v)]; p.time_from_start=Duration(sec=secs)
    tr.points=[p]; pub.publish(tr)

def report(label, offset_along):
    g=buf.lookup_transform("base_link",GRASP,rclpy.time.Time())
    R,o=mat(g)
    axes=np.stack([R[:,0],R[:,1],R[:,2]],axis=1)   # approach, closing, up
    centre = o + R[:,0]*offset_along
    inside_total=0
    for l in TIPS:
        Rt,ot=mat(buf.lookup_transform("base_link",l,rclpy.time.Time()))
        w=(Rt@V.T).T+ot
        local=(w-centre)@axes                      # into book frame
        inside=np.all(np.abs(local)<=HALF,axis=1)
        inside_total+=int(inside.sum())
    # free gap across the closing axis, restricted to the book's own slab
    gaps=[]
    for l in TIPS:
        Rt,ot=mat(buf.lookup_transform("base_link",l,rclpy.time.Time()))
        w=(Rt@V.T).T+ot
        local=(w-centre)@axes
        slab=local[(np.abs(local[:,0])<=HALF[0])&(np.abs(local[:,2])<=HALF[2])]
        gaps.append(slab[:,1] if len(slab) else np.array([np.nan]))
    left,right=gaps
    if np.all(np.isnan(left)) or np.all(np.isnan(right)):
        span="no pad material beside the book at all"
    else:
        lo=np.nanmin(np.abs(left)); ro=np.nanmin(np.abs(right))
        span="nearest pad surfaces %.1f mm and %.1f mm from the book centre line"%(lo*1000,ro*1000)
    print("%-9s finger=%+.4f  depth offset %+4.0f mm | %5d pad vertices inside the book | %s"
          %(label, st.get("gripper_left_finger_joint",float('nan')),
            offset_along*1000, inside_total, span))

for value,label in ((0.068,"open"),(0.030,"half")):
    send(value)
    t=time.time()
    while time.time()-t<20: rclpy.spin_once(n,timeout_sec=0.1)
    for off in (0.0, -0.030, -0.060):
        report(label, off)
print()
print("Zero vertices inside means the book fits where the IK aims it.")
print("The book is 30 mm thick, so 15 mm from the centre line is its own surface.")
n.destroy_node(); rclpy.shutdown()
