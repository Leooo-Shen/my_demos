#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:43:06 2021

@author: yuanbi
"""
import rospy
from tf.transformations import euler_from_quaternion
from tf.transformations import quaternion_matrix
from iiwa_msgs.msg import CartesianPose
from geometry_msgs.msg import PoseStamped
#from iiwa_msgs.msg import WrenchStamped
from iiwa_msgs.msg import ControlMode
from iiwa_msgs.srv import ConfigureControlMode
from iiwa_msgs.msg import CartesianImpedanceControlMode
from image_processing.msg import Reward_Pose
#from geometry_msgs.msg import PointStamped
#from control_msgs.msg import JointTrajectoryControllerState
import math
import numpy as np
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import sys
import copy

def rotation_mat_to_quaternion(rotaion_matrix):
    tr=rotaion_matrix[0,0]+rotaion_matrix[1,1]+rotaion_matrix[2,2]
    
    if tr>0:
        s=math.sqrt(tr+1)*2
        qw=0.25*s
        qx=(rotaion_matrix[2,1]-rotaion_matrix[1,2])/s
        qy=(rotaion_matrix[0,2]-rotaion_matrix[2,0])/s
        qz=(rotaion_matrix[1,0]-rotaion_matrix[0,1])/s
    elif rotaion_matrix[0,0]>rotaion_matrix[1,1] and rotaion_matrix[0,0]>rotaion_matrix[2,2]:
        s=math.sqrt(1+rotaion_matrix[0,0]-rotaion_matrix[1,1]-rotaion_matrix[2,2])*2
        qw=(rotaion_matrix[2,1]-rotaion_matrix[1,2])/s
        qx=0.25*s
        qy=(rotaion_matrix[0,1]+rotaion_matrix[1,0])/s
        qz=(rotaion_matrix[0,2]+rotaion_matrix[2,0])/s
    elif rotaion_matrix[1,1]>rotaion_matrix[2,2]:
        s=math.sqrt(1-rotaion_matrix[0,0]+rotaion_matrix[1,1]-rotaion_matrix[2,2])*2
        qw=(rotaion_matrix[0,2]-rotaion_matrix[2,0])/s
        qx=(rotaion_matrix[0,1]+rotaion_matrix[1,0])/s
        qy=0.25*s
        qz=(rotaion_matrix[1,2]+rotaion_matrix[2,1])/s
    else:
        s=math.sqrt(1-rotaion_matrix[0,0]-rotaion_matrix[1,1]+rotaion_matrix[2,2])
        qw=(rotaion_matrix[1,0]-rotaion_matrix[0,1])/s
        qx=(rotaion_matrix[0,2]+rotaion_matrix[2,0])/s
        qy=(rotaion_matrix[1,2]+rotaion_matrix[2,1])/s
        qz=0.25*s
    return qx,qy,qz,qw

# def rot_z(rad):
#     rotation_matrix=[[math.cos(rad),-math.sin(rad),0],[math.sin(rad),math.cos(rad),0],[0,0,1]]
#     rotation_matrix=np.dot(np.array([[-1,0,0],[0,1,0],[0,0,-1]]),np.array(rotation_matrix))
# #     rotation_matrix=np.dot(np.delete(quaternion_matrix(self.start_pos[3:7]),-1,1),np.array(rotation_matrix))
#     return np.array(rotation_matrix)


class iiwa_control():
    def __init__(self,start_pos=np.array([0.4,0.3,0.12,1,0,0,0])):
        rospy.init_node("iiwa_control")
        self.command_pub=rospy.Publisher('/iiwa/command/CartesianPoseLin', PoseStamped, queue_size=10)
#        rospy.Subscriber('/iiwa/PositionJointInterface_trajectory_controller/state',
#                         JointTrajectoryControllerState, self.jointStateCallback)
        rospy.Subscriber('/iiwa/state/CartesianPose',CartesianPose,self.updatePose)
#        rospy.Subscriber('/iiwa/state/CartesianWrench',WrenchStamped,self.updateWrench)
        rospy.Subscriber("/imfusion/cephasonics",Image,self.updateImage)
        rospy.Subscriber("best_pose",Reward_Pose,self.updateRewardPose)
        self.robotStopped=False;
#        self.actualJS=np.array([0, 0, 0, 0, 0, 0, 0]);
#        self.desiredJS=np.array([0, 0, 0, 0, 0, 0, 0]);
        self.current_pos=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.current_rot_matrix=None
        self.desired_rot_matrix=None
        self.force=np.array([0.0, 0.0, 0.0])
        self.torque=np.array([0.0, 0.0, 0.0])
        self.start_pos=start_pos
        self.image = None
        self.bridge = CvBridge()
        self.client_config=rospy.ServiceProxy('/iiwa/configuration/ConfigureControlMode',ConfigureControlMode)
        self.best_reward=0
        self.current_reward=None
        self.best_pose=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        print('initialised')
    def move_to_cartesian_pose(self,desiredPose,z_needed,precision=2):
        posemsg=PoseStamped()
        posemsg.header.frame_id = "iiwa_link_0";
    
        posemsg.pose.position.x = desiredPose[0]
        posemsg.pose.position.y = desiredPose[1]
        posemsg.pose.position.z = desiredPose[2]
        posemsg.pose.orientation.x = desiredPose[3]
        posemsg.pose.orientation.y = desiredPose[4]
        posemsg.pose.orientation.z = desiredPose[5]
        posemsg.pose.orientation.w = desiredPose[6]
        
        self.desired_rot_matrix= quaternion_matrix(desiredPose[3:7])
        
        self.command_pub.publish(posemsg)
        start_time=time.time()
        self.robotStopped=False
        while not self.robotStopped:

            self.current_rot_matrix= quaternion_matrix(self.current_pos[3:7])
            if z_needed:
                pos_reached=bool(np.sum(np.abs(self.current_pos[0:3]-np.array(desiredPose)[0:3]))<0.005*precision)
            else:
                pos_reached=bool(np.sum(np.abs(self.current_pos[0:2]-np.array(desiredPose)[0:2]))<0.005*precision)
            rot_reached=bool(np.sum(np.abs(np.delete(np.array(self.current_rot_matrix),-1,1)-np.delete(np.array(self.desired_rot_matrix),-1,1)))<0.1*precision)
            
            self.robotStopped=(pos_reached and rot_reached) or time.time()-start_time>90
#             sys.stdout.write('pos: %s rot: %s \r' % (pos_reached,rot_reached))
#             sys.stdout.flush()
#             print('pos: ',np.sum(np.abs(self.current_pos[0:3]-np.array(desiredPose)[0:3])))
#             print('rot: ',rot_reached)
#             print('robot stopped: ',pos_reached and rot_reached)
#             print('rot_current: ',np.delete(np.array(self.current_rot_matrix),0,1))
#             print('rot_current: ',np.delete(np.array(self.desired_rot_matrix),0,1))
            self.command_pub.publish(posemsg)
#            print(np.sum(np.abs(self.current_pos-np.array(desiredPose))))
#            print(time.time()-start_time)
        time.sleep(1)
    def jointStateCallback(self,msg):
        self.actualJS=np.array(msg.actual.positions)
        self.desiredJS=np.array(msg.desired.positions)
        
    def move_to(self,pose,precision=2):
        rotaion_matrix=self.rot_z(pose[2])
        qx,qy,qz,qw=rotation_mat_to_quaternion(rotaion_matrix)
        desiredPose=[self.start_pos[0]+pose[0], self.start_pos[1]+pose[1], self.start_pos[2], qx,qy,qz,qw]
#         print(desiredPose)
        self.move_to_cartesian_pose(desiredPose,False,precision)
        
    def move_to_start(self):
        self.move_to_cartesian_pose(self.start_pos,True)
        print('move to start finished')
        
    def updatePose(self,msg):
        self.current_pos[0]=msg.poseStamped.pose.position.x
        self.current_pos[1]=msg.poseStamped.pose.position.y
        self.current_pos[2]=msg.poseStamped.pose.position.z
        self.current_pos[3]=msg.poseStamped.pose.orientation.x
        self.current_pos[4]=msg.poseStamped.pose.orientation.y
        self.current_pos[5]=msg.poseStamped.pose.orientation.z
        self.current_pos[6]=msg.poseStamped.pose.orientation.w
        
#     def updateWrench(self,msg):
#         self.force[0]=msg.wrench.force.x
#         self.force[1]=msg.wrench.force.y
#         self.force[2]=msg.wrench.force.z
#         self.torque[0]=msg.wrench.torque.x
#         self.torque[1]=msg.wrench.torque.x
#         self.torque[2]=msg.wrench.torque.x
        
    def updateImage(self,msg):
        msg.encoding = 'mono8'
        tmp = cv2.resize(self.bridge.imgmsg_to_cv2(msg),(256,256),interpolation=cv2.INTER_LANCZOS4)
        self.image=tmp.astype(np.float)/255
        
    def controller_init(self):
        msg_config=ControlMode()
        control_mode=msg_config.CARTESIAN_IMPEDANCE
        impedance_config=CartesianImpedanceControlMode()
        
        impedance_config.cartesian_stiffness.x=1200
        impedance_config.cartesian_stiffness.y=1200
        impedance_config.cartesian_stiffness.z=400
        impedance_config.cartesian_stiffness.a=50
        impedance_config.cartesian_stiffness.b=50
        impedance_config.cartesian_stiffness.c=150
        
        impedance_config.cartesian_damping.x=0.8
        impedance_config.cartesian_damping.y=0.8
        impedance_config.cartesian_damping.z=0.8
        impedance_config.cartesian_damping.a=0.8
        impedance_config.cartesian_damping.b=0.8
        impedance_config.cartesian_damping.c=0.8
        
        impedance_config.nullspace_stiffness=200
        impedance_config.nullspace_damping=1.0
        
        try:
            self.client_config(control_mode,None,impedance_config,None,None,None)
            print('cartesian impedance control mode activated')
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
            rospy.shutdown()
            
    def rot_z(self,rad):
        rotation_matrix=[[math.cos(rad),-math.sin(rad),0],[math.sin(rad),math.cos(rad),0],[0,0,1]]
    #     rotation_matrix=np.dot(np.array([[-1,0,0],[0,1,0],[0,0,-1]]),np.array(rotation_matrix))
        rotation_matrix=np.dot(np.delete(quaternion_matrix(self.start_pos[3:7]),-1,1),np.array(rotation_matrix))
        return np.array(rotation_matrix)
    
    def updateRewardPose(self,msg):
        self.current_reward=msg.reward
        if msg.reward>self.best_reward:
            self.best_reward=copy.deepcopy(msg.reward)
            best_pose=np.zeros(7)
            best_pose[0]=copy.deepcopy(msg.pose.poseStamped.pose.position.x)
            best_pose[1]=copy.deepcopy(msg.pose.poseStamped.pose.position.y)
            best_pose[2]=copy.deepcopy(msg.pose.poseStamped.pose.position.z)
            best_pose[3]=copy.deepcopy(msg.pose.poseStamped.pose.orientation.x)
            best_pose[4]=copy.deepcopy(msg.pose.poseStamped.pose.orientation.y)
            best_pose[5]=copy.deepcopy(msg.pose.poseStamped.pose.orientation.z)
            best_pose[6]=copy.deepcopy(msg.pose.poseStamped.pose.orientation.w)
            self.best_pose=best_pose
            
    def rot_x(self,rad):
        current_pos = copy.deepcopy(self.current_pos)
        
        rotation_matrix=[[1,0,0],[0,math.cos(rad),-math.sin(rad)],[0,math.sin(rad),math.cos(rad)]]
        
        rotation_matrix=np.dot(np.delete(quaternion_matrix(current_pos[3:7]),-1,1),np.array(rotation_matrix))
        
        qx,qy,qz,qw=rotation_mat_to_quaternion(rotation_matrix)
        desiredPose=[current_pos[0], current_pos[1], current_pos[2], qx,qy,qz,qw]
        self.move_to_cartesian_pose(desiredPose,False)

        
        
#iiwa=iiwa_control()
#iiwa.move_to_start()
        
        
        
        
# def quaternion_rotation_matrix(Q):
#     """
#     Covert a quaternion into a full three-dimensional rotation matrix.
 
#     Input
#     :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
#     Output
#     :return: A 3x3 element matrix representing the full 3D rotation matrix. 
#              This rotation matrix converts a point in the local reference 
#              frame to a point in the global reference frame.
#     """
#     # Extract the values from Q
#     q0 = Q[0]
#     q1 = Q[1]
#     q2 = Q[2]
#     q3 = Q[3]
     
#     # First row of the rotation matrix
#     r00 = 2 * (q0 * q0 + q1 * q1) - 1
#     r01 = 2 * (q1 * q2 - q0 * q3)
#     r02 = 2 * (q1 * q3 + q0 * q2)
     
#     # Second row of the rotation matrix
#     r10 = 2 * (q1 * q2 + q0 * q3)
#     r11 = 2 * (q0 * q0 + q2 * q2) - 1
#     r12 = 2 * (q2 * q3 - q0 * q1)
     
#     # Third row of the rotation matrix
#     r20 = 2 * (q1 * q3 - q0 * q2)
#     r21 = 2 * (q2 * q3 + q0 * q1)
#     r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
#     # 3x3 rotation matrix
#     rot_matrix = np.array([[r00, r01, r02],
#                            [r10, r11, r12],
#                            [r20, r21, r22]])
                            
#     return rot_matrix


# def rotation_matrix_x(angle):
#     theta = angle * np.pi/180  # convert to radians
#     return np.array([[1, 0, 0],
#                     [0, np.cos(theta), -np.sin(theta)],
#                     [0, np.sin(theta), np.cos(theta)]])
     
        
        
        
        
        
        
        
        
        
        
        
        