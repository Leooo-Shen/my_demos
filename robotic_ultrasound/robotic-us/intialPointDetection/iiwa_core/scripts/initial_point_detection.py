#! /usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image # String, Bool
import message_filters
from cv_bridge import CvBridge
import numpy as np
from itertools import combinations
import math


class PhatomCoordinate:
    def __init__(self, camera_info_topic=None, initial_position_topic=None):
        
        rospy.init_node('phatom_coordinate_node')

        # subscribe depth and color image from camera topic
        self.bridgeC = CvBridge()
        self.depth_image = message_filters.Subscriber("camera/depth/image_rect_raw", data_class=Image, queue_size=5)
        self.color_image = message_filters.Subscriber("camera/color/image_raw", data_class=Image, queue_size=5)
        self.timestamp = message_filters.TimeSynchronizer(fs=[self.color_image, self.depth_image], queue_size=1)
        self.timestamp.registerCallback(cb=self.subscribe_image_callback)
        rospy.wait_for_message(topic="/camera/depth/image_rect_raw", topic_type=Image)
        rospy.wait_for_message(topic="/camera/color/image_raw", topic_type=Image)
        
        self.camera_intrinsic = np.array([[615.7256469726562, 0.0, 323.13262939453125],
                            [0.0, 616.17236328125, 237.86715698242188], 
                            [0.0, 0.0, 1.0]])
        
        # # subscribe to the camera request from outside
        # self.camera_info = message_filters.Subscriber(camera_info_topic, data_class=Bool, queue_size=1)
        # # publish detected phantom coordinates to a topic
        # self.publish_initial_position = rospy.Publisher(name=initial_position_topic, data_class=String, queue_size=2)
        
    def subscribe_image_callback(self, ros_camera_img, ros_depth_img):
        self.cv2_color_img = self.bridgeC.imgmsg_to_cv2(ros_camera_img, "bgr8")
        self.cv2_depth_img = self.bridgeC.imgmsg_to_cv2(ros_depth_img, desired_encoding="passthrough")   


    def subscribe_camera_info_callback(self, msg):
        pass

    
    def detect_centers(self, visualize=False):
        
        # segment red color marker
        blur_img = cv2.blur(src=self.cv2_color_img, ksize=(5,5))
        hsv_img = cv2.cvtColor(src=blur_img, code=cv2.COLOR_BGR2HSV)
        lower = np.array([0, 110, 100])
        upper = np.array([5, 255, 235])
        
        self.mask = cv2.inRange(hsv_img, lower, upper)
        
        # remove noises and enlarge target markers
        circular_kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5,5))
        rect_kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5,5))
        opening = cv2.morphologyEx(src=self.mask, op=cv2.MORPH_OPEN, kernel=rect_kernel) # remove the finger
        closing = cv2.morphologyEx(src=opening, op=cv2.MORPH_CLOSE, kernel=circular_kernel) # remove noises in foreground
        dilation = cv2.dilate(src=closing, kernel=None, iterations=3) # expand objects
        
        # determine relevant points to construct a coordinate system
        centers, _ = cv2.findContours(image=dilation, mode=cv2.RECURS_FILTER, method=cv2.CHAIN_APPROX_SIMPLE)
        points = []
        
        vis_coordinate_res = self.cv2_color_img.copy()
        
        for idx, center in enumerate(centers[:4]):
            M = cv2.moments(center)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.drawContours(vis_coordinate_res, [center], -1, (0,255,0), 2)
                cv2.circle(vis_coordinate_res, (cx, cy), 7, (0,0,255), -1)
                cv2.putText(vis_coordinate_res, f'center_{idx}', (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                points.append([cx, cy])
        
        starting_point, direction_point = None, None
        if len(points) == 4:
            pt0, pt1, pt2, _ = self._calculate_point_relation(points)
            starting_point = self._decide_starting_point(pt0, pt1, pt2)
            direction_point = self._calculate_direction_point(starting_point)
        else:
            print('Not enough points.')
            
        if visualize:
            if starting_point:
                cv2.arrowedLine(img=vis_coordinate_res, pt1=pt2, pt2=pt1, color=(0,0,255), thickness=2)
                cv2.arrowedLine(img=vis_coordinate_res, pt1=pt2, pt2=pt0, color=(255,0,0), thickness=2)
                cv2.arrowedLine(img=vis_coordinate_res, pt1=pt2, pt2=starting_point, color=(255,255,0), thickness=2)
                cv2.arrowedLine(img=vis_coordinate_res, pt1=starting_point, pt2=direction_point, color=(255,0,255), thickness=2)
                cv2.imshow('opening.', vis_coordinate_res)
                cv2.waitKey(2)
        
        # return starting_point, direction_point
        return pt0, pt1, pt2
        
    def _calculate_point_relation(self, points):
        """ Determine the relation between red markers and return the (x,y)-coordinate of each point. """
        relation = {}
        for idx in list(combinations(list(range(len(points))), 2)):
            pt1, pt2 = points[idx[0]], points[idx[1]]
            # calculate euclidean distance
            relation[idx] = math.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1]) 
            
        relation = sorted(relation.items(), key=lambda x: x[1])
        longest = relation[-1][0]
        second_longest = relation[-2][0]
        shortest = relation[0][0]
        
        # try:
        pt1 = set(longest).intersection(second_longest).pop()
        pt0 = [i for i in list(longest) if i != pt1][0]
        pt2 = [i for i in list(second_longest) if i != pt1][0]        
        pt3 = [i for i in list(shortest) if i != pt1][0]
        # except KeyError:
        #     print(f'No point to pop.') 
        return points[pt0], points[pt1], points[pt2], points[pt3] 
    
    
    def _decide_starting_point(self, pt0, pt1, pt2, h=170, theta=2.7):
        pt1_x, pt1_y = pt2
        y = h * np.sin(theta)
        x = h * np.cos(theta)
        new_point = [int(pt1_x + x), int(pt1_y + y)]
        return new_point
    
    def _calculate_direction_point(self, pt, h=100, theta=1.5708*1.5):
        pt_x, pt_y = pt
        new_point = [int(pt_x + h * np.cos(theta)), int(pt_y + h * np.sin(theta))]
        return new_point
        
    def transform(self, pixel_x, pixel_y, camera_intrinsics):
        """  
        Convert the depth and image point information to metric coordinates
        Parameters:
        -----------
        depth: double
            The depth value of the image point
        pixel_x: double
            The x value of the image coordinate
        pixel_y: double
            The y value of the image coordinate
        camera_intrinsics: 
            The intrinsic values of the imager in whose coordinate system the depth_frame is computed
        Return:
        ----------
        X : double
            The x value in meters
        Y : double
            The y value in meters
        Z : double
            The z value in meters
            
        From https://github.com/IntelRealSense/realsense-ros/issues/551#issuecomment-489873418
        """
        depth = self.cv2_depth_img[pixel_x, pixel_y]
        x = (pixel_x - camera_intrinsics[0, 2])/camera_intrinsics[0, 0] * depth
        y = (pixel_y - camera_intrinsics[1, 2])/camera_intrinsics[1, 1] * depth
        return np.array([x, y, depth]) / 1e+3
    
    def get_rotation_matrix(self, direction_point, starting_point):
        
        direction = np.array([direction_point[0] - starting_point[0], 
                            direction_point[1] - starting_point[1], 0.0])
        direction = direction/np.linalg.norm(direction) # x-axis
        z_axis = np.array([0.0, 0.0, -1.0])
        y_axis = np.cross(z_axis, direction)
        
        return np.hstack((direction.reshape(3, 1),
                            y_axis.reshape(3, 1),
                            z_axis.reshape(3, 1),))
    
            
if __name__ == "__main__":
    pc = PhatomCoordinate(camera_info_topic='camera/camera_info', initial_position_topic='camera/initial_position')
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        # starting_point, direction_point = pc.detect_centers(visualize=True)
        pt0, pt1, pt2 = pc.detect_centers(visualize=True)
        if pt0 and pt1 and pt2:
            point0 = pc.transform(pixel_x=pt0[0], pixel_y=pt0[1], camera_intrinsics=pc.camera_intrinsic)
            point1 = pc.transform(pixel_x=pt1[0], pixel_y=pt1[1], camera_intrinsics=pc.camera_intrinsic)
            point2 = pc.transform(pixel_x=pt2[0], pixel_y=pt2[1], camera_intrinsics=pc.camera_intrinsic)
            # TODO: use for grid separation and find centers
            
            # TODO: publish it
            
        # if starting_point and direction_point:
            # starting_point_3d = pc.transform(pixel_x=starting_point[0],
            #                                  pixel_y=starting_point[1],
            #                                  camera_intrinsics=pc.camera_intrinsic)
            # final_mat = np.eye(N=4, M=4, dtype=np.float)
            # final_mat[:3, :3] = pc.get_rotation_matrix(direction_point, starting_point)
            # final_mat[:3, -1] = starting_point_3d
            # mat_info = f'final homogeneous matrix: \n{final_mat}\n\n'
            # rospy.loginfo(starting_point_3d)
            # print(starting_point_3d)
            # pc.publish_initial_position(starting_point_3d)
        rate.sleep()        
    # cv2.destroyAllWindows()