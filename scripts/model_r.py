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

# Get the current file's directory
current_dir = os.path.dirname(__file__)

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) 

# Set the export directory
export_dir = os.path.join(parent_dir, "yolov5-export")

# Set the YAML file path
data_yaml = os.path.join(export_dir, "dataset.yaml")

# Set the yolo submodule directory
yolo_dir = os.path.join(parent_dir, "yolov5")

logging.getLogger().setLevel(logging.CRITICAL)

# Set random seed for NumPy
np.random.seed(22)

# Set random seed for TensorFlow
tf.random.set_seed(22)

# Set random seed for PyTorch
torch.manual_seed(22)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(22)

# Set random seed for Python's built-in random module
random.seed(22)

# Get the specific COCO subset https://docs.voxel51.com/user_guide/dataset_zoo/datasets.html#dataset-zoo-coco-2017
def load_dataset_with_retry(split_type, max_retries, retry_delay):
    retry_count = 0
    print("\n")
    while retry_count < max_retries:
        try:
            print(f'{"-" * 100}\n\n\tLoading dataset: {yellow_color_code}{split_type}{reset_color_code}\n')
            # Set max_samples conditionally based on split_type to avoid the 40gb test dataset
            max_samples = 25000 if split_type == 'test' else None
            
            # Load dataset with appropriate parameters
            dataset = foz.load_zoo_dataset(
                "coco-2017",
                split=split_type,
                label_types=["detections", "segmentations"],
                classes=["bottle", "cup"],
                max_samples=max_samples
            )
            print(f'\n\tFinished loading: {green_color_code}{split_type}{reset_color_code}')
            return dataset
        
        except Exception as e:
            print(f'\n{red_color_code}\n{"-" * 100}\n\n\tError: {reset_color_code}{e}')
            retry_count += 1
            time.sleep(retry_delay)
            retry_delay *= 2
            print(f'\n{red_color_code}{"-" * 100}{reset_color_code}\n')
            print(f"\nNow retrying to load: {split_type}\n")
    print(f"\n\n{red_color_code}Error: Maximum number of retries ({max_retries}) has been reached. Failed to load {split_type} dataset.{reset_color_code}\n")
    sys.exit()

def export_dataset(export_dir, dataset, split_type):

    print(f'{"-" * 100}\n\n\tExporting dataset: {yellow_color_code}{split_type}{reset_color_code}\n')

    os.makedirs(export_dir, exist_ok=True)
    
    # Suppress UserWarning from fiftyone.utils.yolo module
    warnings.filterwarnings("ignore", category=UserWarning, module="fiftyone.utils.yolo")

    if split_type == "test":
        pass

    dataset.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        split=split_type,
        classes = ["bottle", "cup"]
        )
    
    warnings.resetwarnings()

    print(f'\n\tFinished exporting: {green_color_code}{split_type}{reset_color_code}\n')

def update_yaml(data_yaml):

    print(f'{"-" * 100}\n\n\t{yellow_color_code}Updating Yaml{reset_color_code}')

    with open(data_yaml, 'r') as f:
        dataset = yaml.safe_load(f)
    
    # Replace 'validation' key with 'val'
    if 'validation' in dataset:
        dataset['val'] = dataset.pop('validation')

    # Update the 'nc' field
    if 'names' in dataset:
        dataset['nc'] = len(dataset['names'])

    # Write updated dataset to file
    with open(data_yaml, 'w') as f:
        yaml.safe_dump(dataset, f, default_flow_style=False)

    print(f'\n\t{green_color_code}Finished updating Yaml{reset_color_code}\n')

# increase epochs later
def train_yolov5(yolo_dir, data_yaml_path, model_cfg='yolov5s.yaml', weights='yolov5s.pt', img_size=640, batch_size=16, epochs=1):
    
    print(f'{"-" * 100}\n\n\t{yellow_color_code}Training model{reset_color_code}\n')

    # Path to the YOLOv5 training script
    train_script = os.path.join(yolo_dir, 'train.py')

    # Construct the command
    command = [
        'python3', train_script,
        '--img', str(img_size),
        '--batch', str(batch_size),
        '--epochs', str(epochs),
        '--data', data_yaml_path,
        '--cfg', model_cfg,
        '--weights', weights
    ]

    # Run the training process
    subprocess.run(command)

    print(f'\n\t{green_color_code}Finished training model{reset_color_code}\n')

def get_model_weight(yolo_dir):
    weights_dir = os.path.join(yolo_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, 'yolov5m.pt')

    urllib.request.urlretrieve("https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m.pt", weights_path)

    return os.path.join(yolo_dir, 'weights', 'yolov5m.pt')

def get_best_weights(yolo_dir, exp_num):
    # Path to the specific experiment directory
    exp_dir = f'exp{exp_num}'
    weights_path = os.path.join(yolo_dir, 'runs', 'train', exp_dir, 'weights', 'best.pt')
    
    # Check if the best.pt file exists in the specified experiment
    if os.path.exists(weights_path):
        return weights_path
    else:
        return None

def run_inference_image(yolo_dir, weights_path, source_path, img_size=640, conf_thres=0.25, iou_thres=0.45):
    # Path to the YOLOv5 detection script
    detect_script = os.path.join(yolo_dir, 'detect.py')
    
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

def run_live_inference(yolo_dir, weights_path, img_size=640):
    # Path to the YOLOv5 detection script
    detect_script = os.path.join(yolo_dir, 'detect.py')
    
    # Construct the command to use the live camera feed
    command = [
        'python3', detect_script,
        '--weights', weights_path,
        '--source', '0',
        '--img', str(img_size)
    ]
    
    # Run the inference process
    subprocess.run(command)

validation_dataset = load_dataset_with_retry("validation",5,2)
train_dataset = load_dataset_with_retry("train",5,2)
#test_dataset = load_dataset_with_retry("test",5,2)

print("\n")

export_dataset(export_dir, validation_dataset, "validation")
export_dataset(export_dir, train_dataset, "train")
#export_dataset(export_dir, test_dataset, "test")

update_yaml(data_yaml)

train_yolov5(yolo_dir, data_yaml)

# Best weights on given model
weights_path = get_best_weights(yolo_dir,1)

# Model weight
model_weight = get_model_weight(yolo_dir)

# Test image
source_path = os.path.join(parent_dir, "can_dataset", "archive", "test_image.jpg")

run_inference_image(yolo_dir, model_weight, source_path)

try:
    run_live_inference(yolo_dir, model_weight)
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
