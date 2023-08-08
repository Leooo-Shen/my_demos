"""
Script that publishes webcam frames via ros.
"""


import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
import time
from datetime import datetime


class ImageSender:
    def __init__(self):
        self.prediction_sub = rospy.Subscriber(
            "ultrasound_prediction", Bool, self.prediction_callback
        )
        self.image_request_sub = rospy.Subscriber(
            "ultrasound_image_request", String, self.send_callback
        )
        self.image_pub = rospy.Publisher("ultrasound_image", Image, queue_size=10)
        self.bridge = CvBridge()

    def send_callback(self, data=None):
        print(f"Receive new request - {data}")
        self.send()

    def send(self):
        print("Sending new frame")
        # send new image
        try:
            cam = cv2.VideoCapture(0)
            ret, frame = cam.read()
            if not ret:
                print("could not read frame!")
                return
            cam.release()
        except Exception as e:
            print("Could not open camera!")
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame))
            print("Image published!")
        except Exception as e:
            print(e)
            print("Could not publish image!")

    def prediction_callback(self, data):
        dt = datetime.now()
        t = dt.strftime("%H:%M:%S")
        print("=====================")
        print(f"Received prediction {data} at {t}")
        print("=====================")


if __name__ == "__main__":
    rospy.init_node("ultrasound_sender")
    sender = ImageSender()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
