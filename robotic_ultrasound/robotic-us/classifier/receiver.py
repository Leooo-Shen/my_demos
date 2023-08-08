import torch
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datetime import datetime
from classifier import Classifier
import warnings

warnings.filterwarnings("ignore")


class ImageReceiver:
    def __init__(self):
        self.batch_size = 11  # choose an uneven number
        self.img_batch = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_size = 256
        self.transforms = A.Compose(
            [
                A.Resize(290, 290),
                A.CenterCrop(height=self.img_size, width=self.img_size),
                A.RandomBrightnessContrast(p=0.1),
                A.Normalize(mean=0, std=1),
                ToTensorV2(),
            ]
        )
        self.clean_idx = 0
        self.shadow_idx = 1

        # classifier model
        self.ckpt_path = "./classifier_weights.ckpt"
        self.model = Classifier()
        checkpoint = torch.load(self.ckpt_path)
        model_state_dict = self.model.state_dict()
        for item in checkpoint["state_dict"]:
            if item in model_state_dict:
                model_state_dict[item] = checkpoint["state_dict"][item]
        self.model.load_state_dict(model_state_dict)
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # ros stuff
        self.bridge = CvBridge()

        self.prediction_pub = rospy.Publisher(
            "ultrasound_prediction", Bool, queue_size=10
        )
        self.image_request_pub = rospy.Publisher(
            "ultrasound_image_request", String, queue_size=10
        )
        self.image_sub = rospy.Subscriber(
            "ultrasound_image", Image, self.callback, queue_size=10
        )

    def request_frame(self):
        """
        Only used for debugging
        """
        try:
            msg = String()
            msg.data = "New frame please"
            self.image_request_pub.publish(msg)
            rospy.Rate.sleep(rospy.Rate(1))
        except Exception as e:
            print(e)
            print("Could not publish image request!")

    def get_prediction(self) -> bool:
        """
        Returns the prediction about whether or not an image is shadow
        """
        print(f"Calculating prediction for batch of size {len(self.img_batch)}")
        batch = torch.cat(self.img_batch).to(self.device)
        prediction = self.model(batch)
        num_shadows = len(prediction[prediction == self.shadow_idx])
        num_clear = abs(len(prediction) - num_shadows)
        is_shadow = num_shadows > num_clear
        return is_shadow

    def callback(self, data):
        print("Received image")
        try:
            img = self.bridge.imgmsg_to_cv2(data, "passthrough")
            img = self.transforms(image=img)["image"]
            img = img.reshape((1, 1, 256, 256))
            img = img.to(self.device)
            self.img_batch.append(img)

            if len(self.img_batch) == self.batch_size:
                # calculate probability
                is_shadow = self.get_prediction()

                # reset image batch
                self.img_batch = []

                # publish the prediction
                dt = datetime.now()
                try:
                    t = dt.strftime("%H:%M:%S")
                    print(f"Publishing prediction {is_shadow} at {t}")
                    self.prediction_pub.publish(is_shadow)

                except Exception as e:
                    print(e)
                    print(f"Could not publish prediction at {t}")

        except Exception as e:
            print("Error")
            print(e)

        # DEBUG: request new frame
        # self.request_frame()


if __name__ == "__main__":
    rospy.init_node("ultrasound_receiver")
    receiver = ImageReceiver()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
