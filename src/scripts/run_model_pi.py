#!/usr/bin/env python3

import os, logging, subprocess

import urllib.request

import rospy, cv2, cv_bridge, numpy, time, sys

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from yolov5 import YOLOv5
from std_msgs.msg import Bool
from PIL import Image as PILImage

from ultralytics import YOLO

from std_msgs.msg import Float32

class run_model(object):
    def __init__(self):

        rospy.init_node("run_model")

        # ANSI escape codes
        self.red_color_code = "\033[91m"
        self.reset_color_code = "\033[0m"
        self.green_color_code = "\033[92m"
        self.yellow_color_code = "\033[93m"

        # Loading for the model
        self.loading = False

         # Get the current file's directory
        self.current_dir = os.path.dirname(__file__)

        # Get the scripts directory
        self.src_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))

        # Get the model directory
        self.model_dir = os.path.abspath(os.path.join(self.src_dir, "model")) 

        # Set the export directory
        self.export_dir = os.path.join(self.model_dir, "yolov5-export")

        # Set the YAML file path
        self.data_yaml = os.path.join(self.export_dir, "dataset.yaml")

        # Set the project directory
        self.top_dir = os.path.abspath(os.path.join(self.src_dir, os.pardir))

        # Set the yolo submodule directory
        self.yolo_dir = os.path.join(self.top_dir, "yolov5")

        logging.getLogger().setLevel(logging.CRITICAL)

        # make image run in main thread
        self.latest_image = None
        self.new_image_flag = False

        # set up ROS / OpenCV bridge
        self.bridge = cv_bridge.CvBridge()

        # initalize the debugging window
        #cv2.namedWindow("window", 1)

        # subscribe to the robot's RGB camera data stream
        self.image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.image_callback)

        # set up model
        self.model = None

        self.model_loaded_pub = rospy.Publisher('model_loaded', Bool, queue_size=10)
        self.model_loaded_pub.publish(Bool(data=False))

        self.px_error_pub = rospy.Publisher('px_error', Float32, queue_size=10)

        print(f"\n\tDone Initializing\n")
    
    def get_model_weight(self):
        weights_dir = os.path.join(self.yolo_dir, 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, 'yolov5l.pt')

        urllib.request.urlretrieve("https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5l.pt", weights_path)

        return os.path.join(self.yolo_dir, 'weights', 'yolov5l.pt')

    # only if in yolo file
    def get_best_weights(self, exp_num):
        # Path to the specific experiment directory
        exp_dir = f'exp{exp_num}'
        weights_path = os.path.join(self.yolo_dir, 'runs', 'train', exp_dir, 'weights', 'best.pt')
        
        # Check if the best.pt file exists in the specified experiment
        if os.path.exists(weights_path):
            return weights_path
        else:
            return None
        
    # return the path of the best.pt file in the directory
    def get_trained_model(self):
        return os.path.join(self.top_dir, 'best.pt')

    def run_pi(self, source_image):

        results = self.model(source_image, stream=True)

        # Process results generator
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs
            result.show()  # display to screen
            result.save(filename='result.jpg')  # save to disk

    def run_live_inference(self, weights_path, img_size=640):
        # Path to the YOLOv5 detection script
        detect_script = os.path.join(self.yolo_dir, 'detect.py')
        
        # Construct the command to use the live camera feed
        command = [
            'python3', detect_script,
            '--weights', weights_path,
            '--source', "0",
            '--img', str(img_size)
        ]
        
        # Run the inference process
        subprocess.run(command)

    def image_callback(self, msg):
        # converts the incoming ROS message to OpenCV format and HSV (hue, saturation, value)
        
        if self.loading == True:
            pass
        else:
            # converts the incoming ROS message to OpenCV format and HSV (hue, saturation, value)
            image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            self.latest_image = image
            self.new_image_flag = True

    def send_px(self, boxes_array):
        if len(boxes_array):

            x_min, y_min, x_max, y_max = boxes_array[0]
            # print(f"x_min:\t{x_min}\tx_max:\t{x_max}")
            box_width = x_max - x_min
            x_error = (x_max - (box_width)/2) - 0.5
            # print(f"\n\nPX ERROR: {x_error}")

            #publish this px error
            px_msg = Float32()
            px_msg.data = x_error
            self.px_error_pub.publish(px_msg)

        else:
            pass

    def publish_model_state(self, value):
        # Create a Bool message
        bool_msg = Bool()
        bool_msg.data = value

        # Publish the bool value
        self.model_loaded_pub.publish(bool_msg)
        rospy.loginfo(f"Published bool value: {value}")


    def run(self):

        print(f'\n\n{self.green_color_code}{"-" * 100}\n\n\tGetting Path Vaiables:\n{self.reset_color_code}\n')
        print(f"\t\tCurrent directory: {self.current_dir}")
        print(f"\t\tSrc directory: {self.src_dir}")
        print(f"\t\tModel directory: {self.model_dir}")
        print(f"\t\tExport directory: {self.export_dir}")
        print(f"\t\tYolo directory: {self.yolo_dir}")
        print(f"\t\tTop directory: {self.top_dir}")
        print(f'\n\n{self.green_color_code}{"-" * 100}\n{self.reset_color_code}\n')

        self.loading = True

        self.publish_model_state(False)

        try:
            # Model weight
            model_weight = self.get_model_weight()
            self.model = YOLO('yolov8n.pt')
            self.loading = False
        except:
            self.loading = False
            time.sleep(0.5)
            print(f"\n\t{self.red_color_code}Unable to Load Model{self.reset_color_code}\n")
            sys.exit()

        # Stop the loading dots
        self.loading = False
    
        source_path =  os.path.abspath(os.path.join((os.path.abspath(os.path.join(self.src_dir, os.pardir))), "can.jpg"))

        self.publish_model_state(True)

        try:
            print(f"\n\t{self.yellow_color_code}Running Model with PI Camera:\n{self.reset_color_code}")
            #self.run_pi(source_path)
            rate = rospy.Rate(30)  # Set an appropriate rate (e.g., 30Hz)
            while not rospy.is_shutdown():
                if self.new_image_flag:
                    # cv2.imshow("window", self.latest_image)
                    # cv2.waitKey(3)
                    results = self.model.predict(
                        self.latest_image,
                        verbose = False,
                        conf = 0.1,
                        save=False,
                        stream_buffer = True,
                        # visualize=True,
                        show=True,
                        max_det=1,
                        vid_stride = 10,
                        classes = [39,40,41]
                        )
                    for r in results:
                        #print(f"\n\nBox: {r.boxes.xyxyn}")
            
                        # Convert tensor to NumPy array and extract values
                        boxes_array = r.boxes.xyxyn.cpu().numpy()
                        self.send_px(boxes_array)
                        
                    self.new_image_flag = False
                rate.sleep()
            # while True:
            #     pass
        except KeyboardInterrupt:
            print(f'\n\n{self.red_color_code}{"-" * 100}\n\n\tSession terminated by user\n\n{"-" * 100}{self.reset_color_code}\n')
            

        # try:
        #     if device == '1':
        #         print(f"\n\t{self.yellow_color_code}Running Model with Mackbook Camera:\n{self.reset_color_code}")
        #         self.run_live_inference(model_weight)
        #     else:
        #         print(f"\n\t{self.yellow_color_code}Running Model with PI Camera:\n{self.reset_color_code}")
        #         self.run_inference_image(model_weight, source_path)
        #     while True:
        #         pass
        # except KeyboardInterrupt:
        #     print(f'\n\n{self.red_color_code}{"-" * 100}\n\n\tSession terminated by user\n\n{"-" * 100}{self.reset_color_code}\n')

        # try:
        #     session = fo.launch_app(validation_dataset)
        #     print('\n')
        #     # Keep the session running until you manually close it
        #     while True:
        #         pass
        # except KeyboardInterrupt:
        #     # Handle keyboard interrupt (Ctrl+C)
        #     print(f'\n\n{self.red_color_code}{"-" * 100}\n\n\tSession terminated by user\n\n{"-" * 100}{self.reset_color_code}\n')

if __name__ == '__main__':
    try:
        run_model().run()
    except KeyboardInterrupt:
        red_color_code = "\033[91m"
        reset_color_code = "\033[0m"
        
        print(f'\n\n{red_color_code}{"-" * 100}\n\n\tSession terminated by user\n\n{"-" * 100}{reset_color_code}\n')