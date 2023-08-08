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
from std_msgs.msg import Bool, String
from iiwa_msgs.msg import CartesianWrench
import logging

import sys
import os
import numpy as np
import utils
import time
from datetime import datetime
import csv

sys.path.append(os.getcwd())


# NOTE: the distance in code is in meters, but in robot is in mm!!
# Always * 1e-3 when using the robot

class RobotControl:
    def __init__(self, sweep_angle=20):
        self.pose_sub = rospy.Subscriber(
            "/iiwa/state/CartesianPose", CartesianPose, self.pose_callback
        )
        self.pose_publisher = rospy.Publisher(
            "/iiwa/command/CartesianPose", PoseStamped, queue_size=1
        )
        self.prediction_request_publisher = rospy.Publisher(
            "ultrasound_prediction_request", Bool, queue_size=1
        )
        self.prediction_subscriber = rospy.Subscriber(
            "ultrasound_prediction", Bool, queue_size=1, callback=self.shadow_pred_callback 
        )
        self.force_subscriber = rospy.Subscriber(
            "/iiwa/state/CartesianWrench", CartesianWrench, self.force_sensor_callback
        )

        self.sweep_angle = sweep_angle
        self.epsilon = 1e-3
        self.is_shadow = False

        self.MIN_Z_FORCE = 1
        self.MAX_Z_FORCE = 2
        self.target_force = self.MIN_Z_FORCE + (self.MAX_Z_FORCE - self.MIN_Z_FORCE) / 2
        self.force = None
        self.pose = None
        self.logfile = "history_" + datetime.now().strftime("%H:%M:%S") + ".csv"
        self._init_log()

        rospy.sleep(1)

    def _init_log(self):
        """
        Initialize the log file.
        """
        with open(self.logfile, "w") as f:
            f.write("timestamp,x,y,z,sweep\n")

    def _log_step(self, x, y, z):
        """
        Build a log file to visualize the movement later.
        """
        with open(self.logfile, "a") as f:
            ts = datetime.now().strftime("%H:%M:%S")
            f.write(f"{ts},{x},{y},{z},0\n")

    def _log_sweep(self):
        """
        Takes the previous last line and modifies the last column to 1 to indicate a sweep.
        """
        with open(self.logfile, "r") as file:
            lines = list(csv.reader(file))
            lines[-1][-1] = "1"

        with open(self.logfile, "w", newline="\n") as file:
            writer = csv.writer(file)
            writer.writerows(lines)

    def pose_callback(self, msg):
        """
        Callback to store the current robot pose.
        """
        self.pose = msg.poseStamped

    def wait_for_prediction(self, timeout=3):
        """
        Wait for the shadow prediction.
        Sets the self.is_shadow flag.
        """
        self.prediction_request_publisher.publish(True)
        pred_data = rospy.wait_for_message(
            "/ultrasound_prediction", Bool, timeout=timeout
        )
        self.is_shadow = pred_data.data

    def get_z_force(self):
        """
        Returns the z force.
        """
        while not self.force:
            time.sleep(0.1)

        return self.force.wrench.force.z

    def is_force_in_range(self):
        """
        Checks if the current force is within the pre-defined force range.
        """
        return (
            self.force.wrench.force.z > self.MIN_Z_FORCE
            and self.force.wrench.force.z < self.MAX_Z_FORCE
        )

    def force_sensor_callback(self, msg):
        """
        Callback to register the current force parameters from the robot.
        """
        self.force = msg

    def get_current_pose(self, data=None, sleep=2):
        """
        Waits for and returns the current pose parameters of the robot.
        """
        current_pose = rospy.wait_for_message(
            "/iiwa/state/CartesianPose", CartesianPose, timeout=2
        )
        return current_pose.poseStamped

    def move_xyz(self, x, y, z, sleep=2, security_check=True):
        """Move the end effector to the desired positon

        Args:
            x (float): target position x
            y (float): target position y
            z (float): target position z
        """
        # print(f"Moving to {x}, {y}, {z}")
        self._log_step(x, y, z)
        current_pose = self.get_current_pose()
        target_pose = current_pose

        # # TODO: add range check for safety
        # if x < 400 * 0.001:
        #     print("distance in x axis too small, abandon")
        #     exit(1)
        # elif z < 0:
        #     print("distance in x axis too small, abandon")
        #     exit(1)

        target_pose.pose.position.x = x
        target_pose.pose.position.y = y
        target_pose.pose.position.z = z
        self.pose_publisher.publish(target_pose)

        # make sure reaching the target
        self.reach_target(target_pose=target_pose, sleep=sleep, orientation=False)

    def reach_target(self, target_pose, sleep=0.1, orientation=True):
        """
        Waits until the robot end-effector has reached the target pose or position
        """
        # print("Reach target pose...")
        while True:
            if orientation:
                current_pose = self.get_current_pose()
                if utils.reach_target_orientation(
                    current_pose.pose.orientation,
                    target_pose.pose.orientation,
                    self.epsilon,
                ):
                    break
            else:
                current_pose = self.get_current_pose()
                if utils.reach_target_position(
                    current_pose.pose.position,
                    target_pose.pose.position,
                    self.epsilon,
                ):
                    break
            rospy.sleep(sleep)
        # print("Done reaching target pose")

    def sweep(self, sleep=0.1):
        """
        Performs sweeping motion.
        This is executed after the robot escaped a shadowy area.
        """
        self._log_sweep()
        current_pose = self.get_current_pose()
        target_pose = current_pose
        Q = np.array(
            [
                [current_pose.pose.orientation.x],
                [current_pose.pose.orientation.y],
                [current_pose.pose.orientation.z],
                [current_pose.pose.orientation.w],
            ]
        ).squeeze()
        movements = utils.rotate_along_axis(Q, self.sweep_angle)

        for move in movements:
            target_pose.pose.orientation.x = move[0]
            target_pose.pose.orientation.y = move[1]
            target_pose.pose.orientation.z = move[2]
            target_pose.pose.orientation.w = move[3]
            self.pose_publisher.publish(target_pose)

            # make sure reaching the target
            self.reach_target(target_pose=target_pose)

    def move_camera_position(self, sleep=1):
        """
        At very beginning, go to the high position for camera scan
        """
        print("move to camera position")
        x = 500 * 0.001
        y = 0.0 * 0.001
        z = 450 * 0.001
        self.move_xyz(x, y, z, sleep)

    def set_tool_perpendicular(self, sleep=1):
        """
        Sets the ultrasound probe to the initial neutral position.
        """
        current_pose = self.get_current_pose()
        target_pose = current_pose
        target_pose.pose.orientation.x = 0.0
        target_pose.pose.orientation.y = 1.0
        target_pose.pose.orientation.z = 0.0
        target_pose.pose.orientation.w = 0.0
        # print("Set end effector perpendicular ", target_pose)
        self.pose_publisher.publish(target_pose)
        self.reach_target(target_pose=target_pose, sleep=sleep)
        rospy.sleep(sleep)  # TODO: why this one?

    def shadow_pred_callback(self, prediction):
        """
        Callback to receive and store the most recent ultrasound prediction.
        """
        print(f"Received prediction {prediction}")
        self.is_shadow = prediction.data
