import os, logging, subprocess

import urllib.request

import rospy, cv2, cv_bridge, numpy, threading, time, sys
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from yolov5 import YOLOv5
from std_msgs.msg import Bool

# ANSI escape codes
red_color_code = "\033[91m"
reset_color_code = "\033[0m"
green_color_code = "\033[92m"
yellow_color_code = "\033[93m"

run_thread = True

class model_run_v5(object):
    def __init__(self):

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

        # set up ROS / OpenCV bridge
        self.bridge = cv_bridge.CvBridge()

        # set up model
        self.model = []

        # initalize the debugging window
        #cv2.namedWindow("window", 1)

        # subscribe to the robot's RGB camera data stream
        #self.image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.image_callback)


        ###### for movement.py ######s
        self.trash_pub = rospy.Publisher('is_trash', Bool, queue_size=10)
        # self.trash_pub.publish(Bool(data=True))
    
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

    # def image_callback(self, msg):
    #     # converts the incoming ROS message to OpenCV format and HSV (hue, saturation, value)
        
    #     try:
    #         image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
    #         results = self.model(image)
    #         is_trash = self.process_results(results)
    #         self.trash_pub.publish(is_trash)
    #         # Render detections
    #         cv_image = results.render()[0]

    #         # Display Image
    #         # cv2.imshow("YOLOv5 Detection", cv_image)
    #         # cv2.waitKey(3)
    #     except:
    #         print(f"Waiting for image")

    def loading_dots(self):
        while self.loading and run_thread:
            for dot in range(1, 4):
                print(f"{green_color_code}\033[F\tLoading Model{'.' * dot}{' ' * (3 - dot)}{reset_color_code}\n", end="")
                time.sleep(0.5)
        #print("\t\rModel loaded successfully!    ")

    def run(self):

        print(f'\n\n{green_color_code}{"-" * 100}\n\n\tGetting Path Vaiables:\n{reset_color_code}\n')

        print(f"\t\tCurrent directory: {self.current_dir}")
        print(f"\t\tSrc directory: {self.src_dir}")
        print(f"\t\tModel directory: {self.model_dir}")
        print(f"\t\tExport directory: {self.export_dir}")
        print(f"\t\tYolo directory: {self.yolo_dir}")
        print(f"\t\tTop directory: {self.top_dir}")

        print(f'\n\n{green_color_code}{"-" * 100}\n{reset_color_code}\n')

        self.loading = True

        try:
            # Start the loading dots in a separate thread
            loading_thread = threading.Thread(target=self.loading_dots)
            loading_thread.start()

            # Model weight
            model_weight = self.get_model_weight()

            self.model = YOLOv5(self.get_model_weight())

            loading_thread.join()
        except:
            self.loading = False
            loading_thread.join()
            time.sleep(0.5)
            print(f"\n\t{red_color_code}Unable to Load Model{reset_color_code}\n")
            sys.exit()
        
        # Stop the loading dots
        self.loading = False
    
        source_path =  os.path.abspath(os.path.join((os.path.abspath(os.path.join(self.src_dir, os.pardir))), "can.jpg"))

        device = input(f"\n\t{green_color_code}Type 1 for Macbook Camera\tType 2 for PI Camera{reset_color_code}\n\n\t")

        try:
            if device == '1':
                print(f"\n\t{yellow_color_code}Running Model with Mackbook Camera:\n{reset_color_code}")
                self.run_live_inference(model_weight)
            else:
                print(f"\n\t{yellow_color_code}Running Model with PI Camera:\n{reset_color_code}")
                self.run_pi(source_path)
            while True:
                pass
        except KeyboardInterrupt:
            print(f'\n\n{red_color_code}{"-" * 100}\n\n\tSession terminated by user\n\n{"-" * 100}{reset_color_code}\n')


        # try:
        #     if device == '1':
        #         print(f"\n\t{yellow_color_code}Running Model with Mackbook Camera:\n{reset_color_code}")
        #         self.run_live_inference(model_weight)
        #     else:
        #         print(f"\n\t{yellow_color_code}Running Model with PI Camera:\n{reset_color_code}")
        #         self.run_inference_image(model_weight, source_path)
        #     while True:
        #         pass
        # except KeyboardInterrupt:
        #     print(f'\n\n{red_color_code}{"-" * 100}\n\n\tSession terminated by user\n\n{"-" * 100}{reset_color_code}\n')

        # try:
        #     session = fo.launch_app(validation_dataset)
        #     print('\n')
        #     # Keep the session running until you manually close it
        #     while True:
        #         pass
        # except KeyboardInterrupt:
        #     # Handle keyboard interrupt (Ctrl+C)
        #     print(f'\n\n{red_color_code}{"-" * 100}\n\n\tSession terminated by user\n\n{"-" * 100}{reset_color_code}\n')

if __name__ == '__main__':
    try:
        model_run_v5().run()
    except KeyboardInterrupt:
        run_thread = False
        print(f'\n\n{red_color_code}{"-" * 100}\n\n\tSession terminated by user\n\n{"-" * 100}{reset_color_code}\n')
        sys.exit(0)