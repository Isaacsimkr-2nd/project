# subscribe /image and apply YOLOv11 model to the image

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from ultralytics import YOLO
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self._subscription = self.create_subscription(
            Image,
            '/camera_color_frame/image_raw',
            self.image_callback,
            10
        )
        self.model_path = '~/yolo/yolo11n.pt'  # YOLO 모델 경로
        self.model = YOLO(self.model_path)

    def image_callback(self, msg):
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(image)
        annotated_frame = results[0].plot()
        cv2.imshow("Yolo Inference", annotated_frame)
        cv2.waitKey(1)
    
def main():
    rclpy.init()
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()


if __name__=="__main__":
    main()