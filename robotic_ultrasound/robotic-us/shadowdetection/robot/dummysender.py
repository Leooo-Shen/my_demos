"""
Script that publishes webcam frames via ros.
"""


import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import time


class ImageSender:
    """
    @author Chris: fix this crap
    """

    def __init__(self):
        self.image_pub = rospy.Publisher("ultrasound_image", Image, queue_size=10)
        self.bridge = CvBridge()
        # self.rate = rospy.Rate(2)

        self.cam = cv2.VideoCapture(1)
        self.send()  # initial send

    def send(self):
        while True:
            # send new image
            ret, frame = self.cam.read()
            if not ret:
                print("could not read frame!")
                return

            # Get the height and width of the image
            height, width = frame.shape[:2]

            # Find the center of the image
            center_x, center_y = int(width / 2), int(height / 2)

            # Find the coordinates to crop the image
            x = center_x - 350
            y = center_y - 450
            w = 700
            h = 700

            # Crop the image
            crop_frame = frame[y : y + h, x : x + w]

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("img.png", crop_frame)

            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(crop_frame))
            except Exception as e:
                print(e)
                print("Could not publish image!")
            # rospy.sleep(0.1)
            # time.sleep(1)
            # self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node("ultrasound_sender")
    isndr = ImageSender()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        isndr.cam.release()
