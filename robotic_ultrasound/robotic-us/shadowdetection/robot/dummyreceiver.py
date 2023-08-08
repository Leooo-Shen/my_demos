import sys
import os
import torch
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
from encoder import Encoder
import numpy as np
from datetime import datetime
import cv2


class DummyImageReceiver:
    def __init__(self):
        print("init")
        self.img_batch = []
        self.window_size = 10  # maximum of 500 images saved
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_size = 256
        self.transforms = A.Compose(
            [
                A.CenterCrop(height=300, width=300),  # TODO: fix the cropping
                # A.Resize(290, 290),
                A.Resize(height=self.img_size, width=self.img_size),
                A.Normalize(mean=0, std=1),
                ToTensorV2(),
            ]
        )

        # clustering model
        with open("./kmeans_model.pkl", "rb") as f:
            self.kmeans: KMeans = pickle.load(f)
        with open("./pca_model.pkl", "rb") as f:
            self.pca: PCA = pickle.load(f)

        # encoder model
        self.ckpt_path = "./last.ckpt"
        self.model = Encoder()
        checkpoint = torch.load(self.ckpt_path)
        model_state_dict = self.model.state_dict()
        for item in checkpoint["state_dict"]:
            if item in model_state_dict:
                model_state_dict[item] = checkpoint["state_dict"][item]
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.request_sub = rospy.Subscriber(
            "ultrasound_prediction_request", Bool, self.request_callback
        )
        self.prediction_pub = rospy.Publisher(
            "ultrasound_prediction", Bool, queue_size=10
        )
        self.dummyshadow = [
            False,
            True,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
        self.idx = 0
        print("init!!!!!!")

    def get_prediction(self) -> bool:
        """
        Returns the prediction about whether or not an image is shadow.
        """
        batch = torch.cat(self.img_batch).to(self.device)
        latent_space = self.model(batch)
        latent_space = self.pca.transform(latent_space.detach().cpu())
        cluster_pred = self.kmeans.predict(latent_space)
        print(cluster_pred)
        num_shadows = len(cluster_pred[cluster_pred == 2])
        num_clear = abs(len(cluster_pred) - num_shadows)
        is_shadow = num_shadows > num_clear
        return is_shadow

    def request_callback(self, data):
        """
        Called if the sweep controller requests a prediction for the past batch.
        """
        print("Get new prediction request!")

        is_shadow = self.dummyshadow[self.idx]
        time.sleep(1)
        self.prediction_pub.publish(is_shadow)
        print("publish {} at index {}".format(is_shadow, self.idx))

        self.idx += 1

        # # get prediction
        # print("Batch size: ", len(self.img_batch))
        # is_shadow = self.get_prediction()

        # # reset image batch
        # self.img_batch = []

        # # publish the prediction
        # dt = datetime.now()
        # try:
        #     t = dt.strftime("%H:%M:%S")
        #     print(f"Publishing prediction {is_shadow} at {t}")
        #     self.prediction_pub.publish(is_shadow)
        # except Exception as e:
        #     print(e)
        #     print(f"Could not publish prediction at {t}")

    def img_callback(self, data):
        """
        Called whenever a new ultrasound image is received.
        Stores the image in the batch.
        """
        try:
            # img = self.bridge.imgmsg_to_cv2(data, "passthrough")
            img = np.frombuffer(data.data, dtype=np.uint8).reshape(
                data.height, data.width, -1
            )
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = self.transforms(image=img)["image"]

            img = img.reshape((1, 1, 256, 256))
            img = img.to(self.device)
            self.img_batch.append(img)

            # only store the last self.window_size images!
            if len(self.img_batch) > self.window_size:
                self.img_batch = self.img_batch[-self.window_size :]

            self.whatever_pub.publish(True)

        except Exception as e:
            print("Error")
            print(e)


if __name__ == "__main__":
    print("start dummy receiver")
    rospy.init_node("ultrasound_receiver", anonymous=True)
    ir = DummyImageReceiver()
    try:
        print("start spin")
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
