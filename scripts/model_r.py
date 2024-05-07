import os, random, shutil, logging, zipfile, yaml, PIL, torch, random, time, sys, warnings

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

export_path_prefix = os.path.dirname(__file__) + "/yolov5-dataset/"

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

def export_dataset(testlol, split_type):

    print(f'{"-" * 100}\n\n\tExporting dataset: {yellow_color_code}{split_type}{reset_color_code}\n')

    export_dir = os.path.dirname(__file__) + "/yolov5-dataset/"
    os.makedirs(export_dir, exist_ok=True)
    
    # Suppress UserWarning from fiftyone.utils.yolo module
    warnings.filterwarnings("ignore", category=UserWarning, module="fiftyone.utils.yolo")

    classes = None if split_type == "test" else ["bottle", "cup"]

    testlol.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        split=split_type,
        classes=classes
        )
    
    warnings.resetwarnings()

    print(f'\n\tFinished exporting: {green_color_code}{split_type}{reset_color_code}\n')

validation_dataset = load_dataset_with_retry("validation",5,2)
train_dataset = load_dataset_with_retry("train",5,2)
#test_dataset = load_dataset_with_retry("test",5,2)

export_dataset(validation_dataset, "validation")
export_dataset(train_dataset, "train")
#export_dataset(test_dataset, "test")


try:
    session = fo.launch_app(validation_dataset)
    print('\n')
    # Keep the session running until you manually close it
    while True:
        pass
except KeyboardInterrupt:
    # Handle keyboard interrupt (Ctrl+C)
    print(f'\n\n{red_color_code}{"-" * 100}\n\n\tSession terminated by user\n\n{"-" * 100}{reset_color_code}\n')
