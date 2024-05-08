import os, logging, yaml, time, sys, warnings, subprocess, torch

import urllib.request

import fiftyone as fo
import fiftyone.zoo as foz

from ultralytics import YOLO

# ANSI escape codes
red_color_code = "\033[91m"
reset_color_code = "\033[0m"
green_color_code = "\033[92m"
yellow_color_code = "\033[93m"

class train(object):

    def __init__(self):

         # Get the current file's directory
        self.current_dir = os.path.dirname(__file__)

        # Get the scripts directory
        self.src_dir = os.path.abspath(os.path.join(self.current_dir, os.pardir))

        # Get the model directory
        self.model_dir = os.path.abspath(os.path.join(self.src_dir, "model")) 

        # Set the export directory
        self.export_dir = os.path.join(self.model_dir, "yolov8-export")

        # Set the YAML file path
        self.data_yaml = os.path.join(self.export_dir, "dataset.yaml")

        # Set the yolo submodule directory
        self.yolo_dir = os.path.join(self.model_dir, "yolov5")

        logging.getLogger().setLevel(logging.CRITICAL)

        # Build a new model from YAML 
        self.model = YOLO('yolov8n.yaml')

        # Pre-trained model
        self.pre_trained_model = YOLO('yolov8n.pt')

        logging.getLogger().setLevel(logging.CRITICAL)

    # Get the specific COCO subset https://docs.voxel51.com/user_guide/dataset_zoo/datasets.html#dataset-zoo-coco-2017
    def load_dataset_with_retry(self, split_type, max_retries, retry_delay):
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

    def export_dataset(self, dataset, split_type):

        print(f'{"-" * 100}\n\n\tExporting dataset: {yellow_color_code}{split_type}{reset_color_code}\n')

        os.makedirs(self.export_dir, exist_ok=True)
        
        # Suppress UserWarning from fiftyone.utils.yolo module
        warnings.filterwarnings("ignore", category=UserWarning, module="fiftyone.utils.yolo")

        if split_type == "test":
            pass

        dataset.export(
            export_dir=self.export_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            split=split_type,
            classes = ["bottle", "cup"]
            )
        
        warnings.resetwarnings()

        print(f'\n\tFinished exporting: {green_color_code}{split_type}{reset_color_code}\n')

    def update_yaml(self):

        print(f'{"-" * 100}\n\n\t{yellow_color_code}Updating Yaml{reset_color_code}')

        with open(self.data_yaml, 'r') as f:
            dataset = yaml.safe_load(f)
        
        # Replace 'validation' key with 'val'
        if 'validation' in dataset:
            dataset['val'] = dataset.pop('validation')

        # Update the 'nc' field
        if 'names' in dataset:
            dataset['nc'] = len(dataset['names'])

        # Write updated dataset to file
        with open(self.data_yaml, 'w') as f:
            yaml.safe_dump(dataset, f, default_flow_style=False)

        print(f'\n\t{green_color_code}Finished updating Yaml{reset_color_code}\n')
    
    # might have to change this
    def get_trained_model(self):
        
        # Path to the model file ** need to change this for v8
        # model_path = YOLO('path/to/last.pt')
        model_path = ''

        return model_path

    # increase epochs later
    def train_yolov8(self, model, data_yaml):
        
        print(f'{"-" * 100}\n\n\t{yellow_color_code}Training model{reset_color_code}\n')
        
        results = model.train(
            data=data_yaml,
            patience=50,
            save=True,
            name="v8_trained",
            epochs=100,
            imgsz=640,
            device='mps'
        )

        print(f'\n\t{green_color_code}Finished training model on CPU{reset_color_code}\n')

        return results
    
    def val_yolov8(self, model, data_yaml):

        print(f'{"-" * 100}\n\n\t{yellow_color_code}Validating model{reset_color_code}\n')

        validation_results = model.val(
            data=data_yaml,
            save_hybrid=True,
            conf=0.1,
            iou=0.6,
            device='mps',
            split='val'
        )

        return validation_results

    def run(self):
        # validation_dataset = self.load_dataset_with_retry("validation",5,2)
        # train_dataset = self.load_dataset_with_retry("train",5,2)
        # #test_dataset = self.load_dataset_with_retry("test",5,2)

        # print("\n")

        # self.export_dataset(validation_dataset, "validation")
        # self.export_dataset(train_dataset, "train")
        # #export_dataset(export_dir, test_dataset, "test")

        # self.update_yaml()

        # # Best weights on given model
        # weights_path = get_best_weights(yolo_dir,9)

        # # Pre-trained weight
        # pre_trained_weight = self.get_model_weight()

        # Trained weight
        trained_weight = self.get_trained_model()

        model = YOLO('yolov8n.pt')

        trained_model = self.train_yolov8(self.model,self.data_yaml)
        
if __name__ == "__main__":
    try:
        train().run()
    except KeyboardInterrupt:
        print(f'\n\n{red_color_code}{"-" * 100}\n\n\tSession terminated by user\n\n{"-" * 100}{reset_color_code}\n')