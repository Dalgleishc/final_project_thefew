import os
import logging
import subprocess
import urllib.request
import rospy
import cv2
import cv_bridge
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Int32  # Import the standard message type for x-offset
from yolov5 import YOLOv5

# ANSI escape codes
red_color_code = "\033[91m"
reset_color_code = "\033[0m"
green_color_code = "\033[92m"
yellow_color_code = "\033[93m"

class model_run_v5(object):
    def __init__(self):
        # Get the current file's directory
        self.current_dir = os.path.dirname(__file__)
        self.src_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))
        self.model_dir = os.path.abspath(os.path.join(self.src_dir, "model"))
        self.export_dir = os.path.join(self.model_dir, "yolov5-export")
        self.data_yaml = os.path.join(self.export_dir, "dataset.yaml")
        self.top_dir = os.path.abspath(os.path.join(self.src_dir, os.pardir))
        self.yolo_dir = os.path.join(self.top_dir, "yolov5")

        logging.getLogger().setLevel(logging.CRITICAL)

        # Set up ROS / OpenCV bridge
        self.bridge = cv_bridge.CvBridge()
        self.model = None  # Placeholder for the YOLOv5 model instance

        # Subscribe to the robot's RGB camera data stream
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)

        # Publisher for object position
        self.position_pub = rospy.Publisher('object_position', Int32, queue_size=10)

    def get_model_weight(self):
        weights_dir = os.path.join(self.yolo_dir, 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, 'yolov5l.pt')
        urllib.request.urlretrieve("https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5l.pt", weights_path)
        return weights_path

    def image_callback(self, msg):
        try:
            # Convert the ROS message to OpenCV format
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Perform inference
            results = self.model(image)
            x_offset = self.process_results(results, image.shape[1])

            # Publish the result
            if x_offset is not None:
                self.position_pub.publish(x_offset)

            # Optionally, render the results and display the image
            cv_image = results.render()[0]
            cv2.imshow("YOLOv5 Detection", cv_image)
            cv2.waitKey(3)
        except Exception as e:
            print(f"Error processing image: {e}")

    def process_results(self, results, image_width):
        # Process the model's results to find "can" or "bottle" and calculate x_offset
        center_x = image_width // 2
        for detection in results.xyxy[0]:
            label = results.names[int(detection[5])]
            confidence = detection[4]
            if label in ['can', 'bottle'] and confidence > 0.5:
                # Calculate the center of the bounding box
                bbox_center_x = (detection[0] + detection[2]) / 2
                # Calculate the offset from the image center
                x_offset = int(bbox_center_x - center_x)
                return x_offset
        return None

    def run(self):
        print(f"Current directory: {self.current_dir}")
        print(f"Src directory: {self.src_dir}")
        print(f"Model directory: {self.model_dir}")
        print(f"Export directory: {self.export_dir}")
        print(f"Yolo directory: {self.yolo_dir}")

        # Download and load the model weights
        model_weight = self.get_model_weight()
        self.model = YOLOv5(model_weight)

        rospy.init_node('model_run_v5', anonymous=True)

        # Spin to keep the script running and processing callbacks
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print(f'\n\n{red_color_code}{"-" * 100}\n\n\tSession terminated by user\n\n{"-" * 100}{reset_color_code}\n')

if __name__ == '__main__':
    try:
        model_run_v5().run()
    except KeyboardInterrupt:
        print(f'\n\n{red_color_code}{"-" * 100}\n\n\tSession terminated by user\n\n{"-" * 100}{reset_color_code}\n')
