import os, random, shutil, logging, zipfile, yaml, PIL, torch, random, time, sys, warnings, subprocess

import urllib.request

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib as plt
import tensorflow as tf

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.yolo as foy

from IPython.display import Image 
from sklearn.model_selection import train_test_split
from pylabel import importer
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
from pylabel import importer
from requests.exceptions import ConnectionError

# ANSI escape codes
red_color_code = "\033[91m"
reset_color_code = "\033[0m"
green_color_code = "\033[92m"
yellow_color_code = "\033[93m"

class model_run_v5(object):
    def __init__(self):
         # Get the current file's directory
        self.current_dir = os.path.dirname(__file__)

        # Get the parent directory
        self.parent_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir)) 

        # Set the export directory
        self.export_dir = os.path.join(self.parent_dir, "yolov5-export")

        # Set the YAML file path
        self.data_yaml = os.path.join(self.export_dir, "dataset.yaml")

        # Set the yolo submodule directory
        self.yolo_dir = os.path.join(self.parent_dir, "yolov5")

        logging.getLogger().setLevel(logging.CRITICAL)
    

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

        return os.path.join(self.parent_dir,"best.pt")

    def run_inference_image(self, weights_path, source_path, img_size=640, conf_thres=0.25, iou_thres=0.45):
        # Path to the YOLOv5 detection script
        detect_script = os.path.join(self.yolo_dir, 'detect.py')
        
        # Construct the command
        command = [
            'python3', detect_script,
            '--weights', weights_path,
            '--source', source_path,
            '--img', str(img_size),
            '--conf-thres', str(conf_thres),
            '--iou-thres', str(iou_thres)
        ]
        
        # Run the inference process
        subprocess.run(command)

    def run_live_inference(self, weights_path, img_size=640):
        # Path to the YOLOv5 detection script
        detect_script = os.path.join(self.yolo_dir, 'detect.py')
        
        # Construct the command to use the live camera feed
        command = [
            'python3', detect_script,
            '--weights', weights_path,
            '--source', '0',
            '--img', str(img_size)
        ]
        
        # Run the inference process
        subprocess.run(command)
    
    def run(self):

        # Model weight
        model_weight = self.get_trained_model()

        model_weight_train = self.get_model_weight()

        # # Test image
        # source_path = os.path.join(parent_dir, "can_dataset", "archive", "test_image.jpg")

        # self.run_inference_image(model_weight, source_path)

        try:
            self.run_live_inference(model_weight_train)
            while True:
                pass
        except KeyboardInterrupt:
            print(f'\n\n{red_color_code}{"-" * 100}\n\n\tSession terminated by user\n\n{"-" * 100}{reset_color_code}\n')

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
        print(f'\n\n{red_color_code}{"-" * 100}\n\n\tSession terminated by user\n\n{"-" * 100}{reset_color_code}\n')