#!/usr/bin/env python3

import rospy
import sys
import logging
import os
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
# import cv2
# from cv_bridge import CvBridge
import moveit_commander
import math
from std_msgs.msg import Float32
# from yolov5 import YOLOv5
# import os

class Movement:
    def __init__(self):
        rospy.init_node('movement_controller')

        # ANSI escape codes
        self.red_color_code = "\033[91m"
        self.reset_color_code = "\033[0m"
        self.green_color_code = "\033[92m"
        self.yellow_color_code = "\033[93m"
        self.bold_start = "\033[1m"
        self.bold_end = "\033[0m"

        # for hand
        self.something_in_hand = False
        
        # For loading the model
        self.loaded = False
        
        # initalize pixel error subscriber
        self.px_error_sub = rospy.Subscriber('px_error', Float32, self.get_px)

        # initalize the loading model subscriber
        self.model_loaded_sub = rospy.Subscriber('model_loaded', Bool, self.model_loading)
        
        # initalize velocity publisher
        self.velocity_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)

        #initialize the subscriber to read the LiDAR messages
        self.scan_subscriber = rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        self.front_distance = 3.5

        self.stop_distance = 0.2
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")

        self.move_group_arm.go([0, 0, 0, 0], wait=False)
        self.move_group_gripper.go([0.01, 0.01], wait=False)
        self.move_group_arm.stop()
        self.move_group_gripper.stop()


        self.initialize_robot()
        self.px_error = 2

        rospy.on_shutdown(self.stop_robot)

    def send_movement(self, velocity, angular):
        """
        Sends a movement to the bot with a given forward (x-axis) velocity and angular velocity (z-axis) for convinience.
        """
        #set velocity arguments
        move_cmd = Twist()
        move_cmd.linear.x = velocity
        move_cmd.angular.z = angular
        # send velocity command
        self.velocity_publisher.publish(move_cmd)
        rospy.sleep(0.05)

    def stop_robot(self):
        os.system('clear')
        print(f'\n{self.red_color_code}{"-" * 100}\n\n\tShutting down: Stopping the robot:\n{self.reset_color_code}')
        self.send_movement(0,0)
        print(f'\n{self.green_color_code}\tDone\n\n{"-" * 100}{self.reset_color_code}\n')
        self.reset_arms()

    def initialize_robot(self):
        print(f'\n{self.green_color_code}{"-" * 100}\n\n\tInitializing robot arm and gripper to default states:\n{self.reset_color_code}')        
        self.move_group_arm.go([0, 0, 0, 0], wait=True)
        self.move_group_gripper.go([0.01, 0.01], wait=True)
        self.move_group_arm.stop()
        self.move_group_gripper.stop()
        print(f'\n{self.green_color_code}\tDone\n\n{"-" * 100}{self.reset_color_code}\n')

    def reset_arms(self):
        print(f'\n\n{self.red_color_code}{"-" * 100}\n\n\tResetting arm angles:\n{self.reset_color_code}')
        self.move_group_arm.go([0, 0, 0, 0], wait=True)
        self.move_group_gripper.go([0.01, 0.01], wait=True)
        self.move_group_arm.stop()
        self.move_group_gripper.stop()
        print(f'\n{self.green_color_code}\tDone\n\n{"-" * 100}{self.reset_color_code}\n')
        os.system('clear')

    def scan_callback(self, msg):
        """
        Callback for the LiDAR measurements. We collect and update the only stat we care about, which
        is the closest distance in the narrow cone in front of us, so we know how close we are to our goal.
        """
        if len(msg.ranges) < 400:
            #LDS-01 processing
            max_range = 3.5
            processed_ranges = [r if r != 0.0 else max_range for r in msg.ranges]
            #angle_offset = math.pi
            # Get the first 10 and last 10 range values
            front_ranges = processed_ranges[:10] + processed_ranges[-10:]

        else:
            #the other one processing
            max_range = 12
            processed_ranges = [r if r != math.inf else max_range for r in msg.ranges]
            # Get the first 10 and last 10 range values
            front_index = (len(processed_ranges)-1) // 2
            front_ranges = processed_ranges[front_index-5:front_index+5]

        # Calculate the minimum distance in the front range
        self.front_distance = min(front_ranges)

    def get_px(self, msg):
        self.px_error = msg.data
        if self.px_error == None:
            print(f"Do not see something")

    def screen_print(self):

        os.system('clear')

        # Set the logger level to ERROR
        rospy.set_param('/rosconsole/logger_level', 'error')
        logging.getLogger('rosout').setLevel(logging.ERROR)

        print(f"\n\n{self.green_color_code}{self.bold_start}TRASH BOT:{self.bold_end}{self.reset_color_code}\n\n{'-' * 100}\n\n{self.yellow_color_code}\tPX Error: {self.reset_color_code}{self.px_error}\n\n{self.yellow_color_code}\tFront Distance: {self.reset_color_code}{self.front_distance}\n\n{'-' * 100}\n \033[{20}F", end='')

    def go_to(self):
        """
        Makes the robot go to a certain location based on horizontal linear difference from image. 
        We use the front_distance attribute which is updated by LiDAR to know when we are close distance wise,
        while continuously correcting our angle based on where horizontal error.
        """

        self.screen_print()

        #cx is the x position of where the object is
        #the robot needs to turn toward the object (have cx be in the middle of the screen)
        #then the robot needs to aproach the robot till until it is 0.1m away (scan lidar)
        # convert pixal error to angular error
        angular = -1 * (self.px_error*2)

        # does not see anything
        if self.px_error == 2:
            self.send_movement(0,0.1)
        
        # if not aligned
        # elif abs(angular) > 0.05 and self.px_error <= 1:
        #     self.send_movement(0, angular)

        # is aligned
        # elif self.px_error <= 1 and abs(angular) <= 0.05:
        else:
            # far from object, getting color
            if self.front_distance > 0.252:
                self.send_movement(min(0.1, 0.1 * self.front_distance), angular / 5)

            # close to object
            elif self.front_distance <= 0.23 and abs(self.px_error) <= 0.08:
                print('stop')
                self.send_movement(0, 0)
                self.pick_up_object()
                rospy.sleep(1)
                self.reset_arms()

    def pick_up_object(self):
        print("Approaching the object...")
        # Extend arm forward and as low as possible while respecting joint limits 
        move_downward_position = [0, math.radians(73), math.radians(-27), math.radians(-12)]  # Extend arm downwards
        self.move_group_arm.go(move_downward_position, wait=True)
        rospy.sleep(4)  # Wait for the arm to reach the extended position
        self.move_group_arm.stop()
        print("downward position reached.")
        self.move_group_gripper.go([-0.01, -0.01], wait=True)  # maximum closure within limits
        rospy.sleep(2)  # Allow time for the gripper to close
        self.move_group_gripper.stop()
        print("Gripper clenched. Lifting the object...")
        self.something_in_hand = True
        lift_position = [0, math.radians(-48), math.radians(-45), math.radians(-101)]  # Retract and lift the arm
        self.move_group_arm.go(lift_position, wait=True)
        rospy.sleep(6)  # Wait for the arm to lift to the safe position
        self.move_group_arm.stop()
        print("Object lifted.")
        # Open the gripper to throw away
        self.move_group_gripper.go([0.01, 0.01], wait=True)
        rospy.sleep(2)
        self.something_in_hand = False
        self.reset_arms()

    def model_loading(self, msg):
        self.loaded = msg.data
        # print(f"Is the model loaded?: {self.loaded}\n")

    def run(self):
        
        self.send_movement(0,0)
        
        try:    
            while not self.loaded:
                # print(f"\033[F Waiting for model to load...")
                pass
        
            rospy.sleep(2)  # Wait for 2 seconds

            os.system('clear')

            while not rospy.is_shutdown():
                self.go_to()
                rospy.sleep(0.1)
        except KeyboardInterrupt:
            print(f'\n\n{self.red_color_code}{"-" * 100}\n\n\tSession terminated by user\n\n{"-" * 100}{self.reset_color_code}\n')
            sys.exit(0)

if __name__ == "__main__":
    try:
        executor = Movement()
        executor.run()
    except rospy.ROSInterruptException:
        print("Movement node interrupted.")
