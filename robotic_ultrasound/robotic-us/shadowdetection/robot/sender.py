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
    def __init__(self):
        self.prediction_sub = rospy.Subscriber("whatever", Bool, self.callback)
        self.image_pub = rospy.Publisher("ultrasound_image", Image, queue_size=10)
        self.bridge = CvBridge()

        self.cam = cv2.VideoCapture(1)
        self.send()  # initial send

    def send(self):
        # send new image
        ret, frame = self.cam.read()
        if not ret:
            print("could not read frame!")
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("img.png", frame)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame))
        except Exception as e:
            print(e)
            print("Could not publish image!")

    def callback(self, data):
        # t = time.time()
        # print(f"Received prediction {data} at {t}")
        self.send()


if __name__ == "__main__":
    rospy.init_node("ultrasound_sender")
    isndr = ImageSender()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    isndr.cam.release()
