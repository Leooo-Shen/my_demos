#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import (
    PoseStamped,
)  # PoseStamped: Represents a point with reference coordinates
from sensor_msgs.msg import JointState, Image

# JointState: Holds data to describe the state of a set of torque controlled joints
# Image: Contains uncompressed image
from iiwa_msgs.msg import JointPosition, CartesianPose

# JointPosition: Holds data describing all joint positions
# CartesianPose: Contains the target pose including redundancy information
from std_msgs.msg import Bool
from iiwa_msgs.msg import CartesianWrench
import logging

import sys
import os
import numpy as np
import utils
import time
from datetime import datetime
import csv
from RobotControl import RobotControl
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from pynput import keyboard

sys.path.append(os.getcwd())


class Pipeline:
    def __init__(self, robot, num_points=15):
        print("Initializing pipeline...")
        self.initial_point_publisher = rospy.Publisher(
            "initial_point_request", Bool, queue_size=1
        )
        self.initial_point_subscriber = rospy.Subscriber(
            "initial_point_publish", numpy_msg(Floats), self.inital_point_callback
        )
        
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press,on_release=self.on_release)
        self.keyboard_listener.start()
        
        rospy.sleep(1)

        self.started = False  # whether or not the robot is moving along the trajectory
        self.robot: RobotControl = robot  # the robot control
        self.trajectory_idx = 0  # the running trajectory index
        self.trajectory_z_fixed = False  # if the z coordinate was adjusted
        self.num_points = num_points
        self.trajectory = None
        self.safety_check = False
        
    def run(self):
        """
        Run the pipeline
        """
        rospy.sleep(1)
        print("Move to camera position, setting end effector perpendicular.")
        self.robot.move_camera_position()
        self.robot.set_tool_perpendicular()

        # request the initial point
        print("Request initial point")
        self.initial_point_publisher.publish(True)

    
    def on_press(self, key):
        if key == keyboard.Key.space:
           self.safety_check = True
           
    def on_release(self, key):
        pass
    
    def inital_point_callback(self, msg):
        """
        We receive a 3x9 matrix that contains 9 3D points.

        @param msg.data contains the matrix as string
        """
        if not self.started:  # only call once
            print("Received initial point, creating trajectory")
            self.started = True
            matrix = msg.data
            matrix = matrix.reshape(2, 4)

            # extract the two points
            p1 = matrix[0][:3]
            p2 = matrix[1][:3]
            grid_centers = utils.create_grid(p1, p2, num_cells=(3,3), shape='s')

            print('* The detected points are {} and {}. The depth is {}. Press Space to start moving the robot'.format(p1[:2], p2[:2], grid_centers[0][2]))
            while not self.safety_check:
                pass                
            # create the trajectory from our 9x9 grid
            self.trajectory = utils.interpolate_stepsize(grid_centers, step_size=1e-2)
            # utils.vis_trajectory(grid_centers, self.trajectory, projection='2d', diagonal_points=[p1, p2], savefig=True)

            # start the pipeline
            print("Starting the robotic movement pipeline")
            self._run()

    def _fix_trajectory_z(self, z):
        if not self.trajectory_z_fixed:
            print("Fix trajectory")
            self.trajectory = [(x, y, z) for (x, y, _) in self.trajectory]
            self.trajectory_z_fixed = True

    def _adjust_z_for_target_force(self, x, y, z):
        """
        Based on the force value, move the probe up or down until the
        force value is in the acceptable range.
        """
        print("Adjusting z for target force...")
        epsilon = 0.2
        step_size = 0.001
        oscillating = [False, False]

        # while the robot is not aournd the target force
        target_force = self.robot.target_force
        while abs(self.robot.get_z_force() - target_force) > epsilon:
            # print("Oscillating: ", oscillating)
            if all(oscillating):
                # print("Adjust step size!")
                step_size *= 0.5
                oscillating = [False, False]

            # print("Still adjusting....")
            # if the force is too small, move up
            if self.robot.get_z_force() < target_force:
                oscillating[0] = True
                # print("move up")
                z += step_size
            else:
                oscillating[1] = True
                # print("move down")
                z -= step_size
            self.robot.move_xyz(x, y, z, sleep=1, security_check=True)
            # rospy.sleep(0.5)

        self._fix_trajectory_z(z)
        # print("Done adjusting...")
        return z

    def _move_along_trajectory(self):
        x, y, z = self.trajectory[self.trajectory_idx]
        z = self._adjust_z_for_target_force(x, y, z)
        self.robot.move_xyz(x, y, z, sleep=0.5)
        self.trajectory_idx += 1

    def _normal_move(self):
        print("> Moving along trajectory...")
        print(self.trajectory)
        while self.trajectory_idx < len(self.trajectory):
            self.robot.prediction_request_publisher.publish(True)
            self._move_along_trajectory()
            # self.robot.wait_for_prediction()
            # if self.robot.is_shadow:
            #     print("  > shadow detected in normal_move")
            #     self._escape_shadow()
        print("Finished the trajectory")


    def _escape_shadow(self):
        print("Escaping shadow area")
        while self.trajectory_idx < len(self.trajectory):
            self.robot.prediction_request_publisher.publish(True)
            self._move_along_trajectory()
            if not self.robot.is_shadow:
                break

        if self.trajectory_idx < len(self.trajectory):
            print("shadow escaped, start sweeping")
            self._perform_sweep()

    def _perform_sweep(self):
        """Perform the sweeping motion"""
        print("Performing sweep...")
        self.robot.sweep()
        self._normal_move()

    def _run(self):
        """
        The main function to run the pipeline.
        """
        print("Running pipeline...")
        self._normal_move()
