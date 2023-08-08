#!/usr/bin/env python3
import rospy
# import std_msgs.msg
from geometry_msgs.msg import PoseStamped #PoseStamped: Represents a point with reference coordinates
from sensor_msgs.msg import JointState, Image 
#JointState: Holds data to describe the state of a set of torque controlled joints
#Image: Contains uncompressed image 
from iiwa_msgs.msg import JointPosition, CartesianPose
#JointPosition: Holds data describing all joint positions
#CartesianPose: Contains the target pose including redundancy information

import message_filters
#message_filters: A set of message filters which take in messages and may output those messages at a later time
import sys
import os
from pynput import keyboard
sys.path.append(os.getcwd())


class RobotControlKeyboard():
    def __init__(self, step_size=0.01):
        self.pose_publisher = rospy.Publisher('/iiwa/command/CartesianPoseLin', PoseStamped, queue_size=1)
        self.image_pub_seg_stable = rospy.Publisher('/imfusion/cephasonics', Image, queue_size=1)
        self.init = True
        self.step_size = step_size
        
    def get_current_pose(self):
        '''
        read the published pose
        '''
        current_pose = rospy.wait_for_message('/iiwa/state/CartesianPose', CartesianPose, timeout=2)
        print('Current Pose is: ', current_pose.poseStamped)
        rospy.sleep(1) 
        return current_pose.poseStamped
    
    def move_x(self):
        current_pose = self.get_current_pose()
        current_pose.pose.position.x = current_pose.pose.position.x + self.step_size
        self.pose_publisher.publish(current_pose)
        print('Publish pose step ', current_pose)
    
    def move_x_negative(self):
        current_pose = self.get_current_pose()
        current_pose.pose.position.x = current_pose.pose.position.x - self.step_size
        self.pose_publisher.publish(current_pose)
        print('Publish pose step ', current_pose)
        
    def move_y(self):
        current_pose = self.get_current_pose()
        current_pose.pose.position.x = current_pose.pose.position.y + self.step_size
        self.pose_publisher.publish(current_pose)
        print('Publish pose step ', current_pose)
        
    def move_y_negative(self):
        current_pose = self.get_current_pose()
        current_pose.pose.position.x = current_pose.pose.position.y - self.step_size
        self.pose_publisher.publish(current_pose)
        print('Publish pose step ', current_pose)

    def move_z(self):
        current_pose = self.get_current_pose()
        current_pose.pose.position.x = current_pose.pose.position.z + self.step_size
        self.pose_publisher.publish(current_pose)
        print('Publish pose step ', current_pose)
        
    def move_z_negative(self):
        current_pose = self.get_current_pose()
        current_pose.pose.position.x = current_pose.pose.position.z + self.step_size
        self.pose_publisher.publish(current_pose)
        print('Publish pose step ', current_pose)


def on_press(key):
    global movement
    if key == keyboard.Key.left:
        movement['x'] = True
    elif key == keyboard.Key.right:
        movement['x_neg'] = True
    elif key == keyboard.Key.up:
        movement['y'] = True
    elif key == keyboard.Key.down:
        movement['y_neg'] = True
    elif key == keyboard.Key.enter:
        movement['z'] = True
    elif key == keyboard.Key.space:
        movement['z_neg'] = True
        
        
def on_release(key):
    global movement
    if key == keyboard.Key.left:
        movement['x'] = False
    elif key == keyboard.Key.right:
        movement['x_neg'] = False
    elif key == keyboard.Key.up:
        movement['y'] = False
    elif key == keyboard.Key.down:
        movement['y_neg'] = False
    elif key == keyboard.Key.enter:
        movement['z'] = False
    elif key == keyboard.Key.space:
        movement['z_neg'] = False
       
    
def main():
    robot = RobotControlKeyboard()
    rospy.init_node('robot',anonymous=True)
    print('init success')

    # for keyboard input
    listener = keyboard.Listener(on_press=on_press,on_release=on_release)
    listener.start()
    
    while not rospy.is_shutdown():
        if movement['x']:
            robot.move_x()
        elif movement['x_neg']:
            robot.move_x_negative()
        elif movement['y']:
            robot.move_y()
        elif movement['y_neg']:
            robot.move_y_negative()
        elif movement['z']:
            robot.move_z()
        elif movement['z_neg']:
            robot.move_z_negative()

if __name__ == '__main__':
    movement = {
        'x': False,
        'x_neg': False,
        'y': False,
        'y_neg': False,
        'z': False,
        'z_neg': False
    }
    
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    