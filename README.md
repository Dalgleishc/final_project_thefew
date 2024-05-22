# final_project_thefew
Final Project for UChicago CMSC Intro to Robotics

# Training 
https://pytorch.org/hub/ultralytics_yolov5/

tutorial: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

# Autonomous Trash-Collecting Robot

## Project Description

### Goal of the Project

The primary goal of this project is to develop an autonomous trash-collecting robot using a TurtleBot 3 equipped with an Open Manipulator arm. By leveraging computer vision and machine learning, the robot is designed to identify, approach, and collect trash autonomously.

### Project Highlights

- **Autonomous Trash Detection and Collection**: The robot is capable of detecting trash using a YOLOv5 model trained on various types of litter. Once detected, the robot moves towards the identified trash, picks it up using its manipulator arm, and deposits it into a makeshift trashcan attached to its back.
- **Integration of Computer Vision and Robotics**: The project integrates advanced computer vision techniques with robotic manipulation, demonstrating a seamless interaction between software and hardware components to achieve a practical task.
- **Continuous Operation**: The robot is designed to operate continuously until no more trash is detected within its field of view, making it efficient for long-term deployments in controlled environments.

### Main Components

1. **TurtleBot 3**
   - **Function**: Serves as the mobile base of the robot, providing locomotion and hosting the core computational unit.
   - **Components**: Includes wheel motors, battery, control board, and various sensors (IMU, encoders).

2. **Open Manipulator Arm**
   - **Function**: Enables the robot to physically interact with the environment, specifically to pick up and manipulate objects.
   - **Components**: Includes multiple servo motors for articulation, gripper for grasping objects.

3. **LIDAR and Camera**
   - **Function**: Provides the robot with environmental perception, crucial for navigation and object detection.
   - **Components**: A 2D LIDAR sensor for mapping and obstacle detection, and a camera for visual recognition of trash.

4. **YOLOv5 Model**
   - **Function**: Performs real-time object detection, identifying trash within the camera's field of view.
   - **Components**: Pre-trained neural network model, customized for trash detection through transfer learning.

5. **Control Software**
   - **Function**: Integrates all hardware components, processes sensor data, executes movement commands, and performs object manipulation.
   - **Components**: ROS (Robot Operating System) for communication and control, Python scripts for specific tasks (detection, movement, grasping).

### System Workflow

1. **Trash Detection**: The camera captures images, which are processed by the YOLOv5 model to identify trash. The detected trash is highlighted with bounding boxes.
2. **Navigation**: The robot calculates the optimal path to the identified trash using LIDAR data for obstacle avoidance and localization.
3. **Object Manipulation**: Upon reaching the trash, the manipulator arm picks it up and drops it into the attached trash can.
4. **Repetition**: The robot continuously scans for more trash, repeating the process until the environment is clear.

## System Architecture

### TurtleBot 3
- **Function**: Provides mobility for the robot, allowing it to navigate the environment.
- **Code**:
  - `cmd_vel_pub` (Publisher): Sends movement commands to the robot.
  - `lidar_sub` (Subscriber): Receives LIDAR data for obstacle detection and navigation.

### Open Manipulator Arm
- **Function**: Performs the physical task of picking up and manipulating objects.
- **Code**:
  - `move_group_arm` (MoveGroupCommander): Controls the arm's movements.
  - `move_group_gripper` (MoveGroupCommander): Controls the gripper's movements.

### YOLOv5 Model
- **Function**: Detects trash in the environment using computer vision.
- **Code**:
  - `YOLOv5` (Model): Loaded and used for real-time trash detection.
  - `image_sub` (Subscriber): Receives images from the robot's camera for processing.

### Control Software
- **Function**: Integrates all hardware and software components, processes sensor data, and executes movement and manipulation commands.
- **Code**:
  - `Movement` class: Main control class that orchestrates the robot's actions.
  - `run()`, `initialize_robot()`, `approach_closest_object()`, `pick_up_object()`: Key methods for robot initialization, navigation, and manipulation.

## ROS Node Diagram
<img width="568" alt="Screenshot 2024-05-22 at 4 45 28 PM" src="https://github.com/Dalgleishc/final_project_thefew/assets/109634431/39203014-b9dd-4db5-a10e-8e21059e9201">


## Execution

### Prerequisites
- Ensure you have ROS installed on your system.
- Install the required ROS packages and dependencies, including `cv_bridge`, `moveit_commander`, and `yolov5`.
- Clone your project repository to your workspace.

### Step-by-Step Instructions



# Final Demo

https://github.com/Dalgleishc/final_project_thefew/assets/114620452/24c61115-2f4a-4f2e-9ab2-a61d19d916ba

Note: Some of the bottles are too wide for the gripper. 
