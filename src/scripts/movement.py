#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import cv2
from cv_bridge import CvBridge
import moveit_commander
import math
from std_msgs.msg import Float32
# from yolov5 import YOLOv5
# import os

class Movement:
    def __init__(self):
        rospy.init_node('movement_controller')
        self.loaded = False
        ###### Publisher ########
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        # self.align = rospy.Subscriber('/custom_message_Reece', CustomMessage, self.go_to) TODO see line 44
        # self.is_trash = rospy.Subscriber('/custom_message_Reece', BOOL, self.is_trash) TODO see line 86
        self.px_error_sub = rospy.Subscriber('px_error', Float32, self.get_px)
        self.model_loaded_sub = rospy.Subscriber('model_loaded', Bool, self.model_loading)
        
        self.cmd_vel_pub.publish
        self.bridge = CvBridge()
        self.twist = Twist()
        self.current_scan = None
        self.stop_distance = 0.2
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")
        self.is_trash = False
        self.front_distance = 3.5
        self.initialize_robot()
        self.px_error = None
        self.somethinginhand = False
        rospy.on_shutdown(self.stop_robot)
    


    ###### custom message from model node for alignment of robot ########
    # def go_to(self, px_error):
    #     """
    #     TODO Makes the robot go to a certain location based on horizontal linear difference from image. 
    #     We use the front_distance attribute which is updated by LiDAR to know when we are close distance wise,
    #     while continuously correcting our angle based on where horizontal error.
    #     """

    #     #cx is the x position of where the object is
    #     #the robot needs to turn toward the object (have cx be in the middle of the screen)
    #     #then the robot needs to aproach the robot till until it is 0.1m away (scan lidar)
    #     # convert pixal error to angular error
    #     angular = -1 * (px_error / 100)

    #     # if not aligned
    #     if abs(angular) > 1.0:
    #         self.cmd_vel_pub.publish(0, angular)

    #     # is aligned
    #     else:
    #         # far from object, getting color
    #         if self.front_distance > 0.25 and not self.something_in_hand:
    #             self.cmd_vel_pub.publish(min(0.1, 0.1 * self.front_distance), angular / 5)

    #         # far from object, getting AR
    #         elif self.front_distance > 0.5 and self.something_in_hand:
    #             self.cmd_vel_pub.publish(min(0.1, 0.1 * self.front_distance), angular / 5)

    #         # close to object
    #         else:
    #             self.cmd_vel_pub.publish(0, 0)
    #             if not self.something_in_hand:
    #                 self.pick_object()
    #             else:
    #                 self.drop_object()
    #                 self.something_in_hand = False
    #                 self.state = self.next_state

    #                 self.cmd_vel_pub.publish(-0.5, 0)
    #                 rospy.sleep(2)

    #                 self.get_next_move(self.state)

    ###### custom boolean message from model node ########
    # def is_trash(self, msg):
    #     self.is_trash = msg.data
    #     rospy.loginfo("Trash detection updated: {}".format(self.is_trash))

    def stop_robot(self):
        rospy.loginfo("Shutting down: Stopping the robot...")
        self.twist.linear.x = 0
        self.twist.angular.z = 0
        self.cmd_vel_pub.publish(self.twist)
        self.reset_arms()

    def initialize_robot(self):
        print("Initializing robot arm and gripper to default states...")
        self.move_group_arm.go([0, 0, 0, 0], wait=True)
        self.move_group_gripper.go([0.01, 0.01], wait=True)
        self.move_group_arm.stop()
        self.move_group_gripper.stop()
        print("Robot initialized to home position with gripper open.")

    ###### Subscriber ########
    def lidar_callback(self, msg):
        self.current_scan = msg
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
            front_ranges = processed_ranges[front_index-10:front_index+10]

        # Calculate the minimum distance in the front range
        self.front_distance = min(front_ranges)

    def find_closest_object(self):
        if self.current_scan is None:
            print("No LIDAR data")
            return None
        ranges = self.current_scan.ranges
        min_distance = float('inf')
        min_angle = None
        for i, distance in enumerate(ranges):
            if 0.05 < distance < min_distance:
                min_distance = distance
                min_angle = i
        if min_angle is None:
            return None
        return min_distance, min_angle

    def get_px(self, msg):
        self.px_error = msg.data
        print(f"px error: {px_error}")

    def approach_closest_object(self):
        '''
        Approaches the closest object based on px_error for alignment and distance checking.
        '''
        print("Approaching the closest object.")
        while not rospy.is_shutdown():
            closest_object = self.find_closest_object()
            if closest_object:
                # Convert pixel error to angular error
                angular = -1 * (self.px_error / 100)
                
                # Check if the robot is aligned
                if abs(angular) > 1.0:
                    # Not aligned
                    self.send_movement(0, angular)
                else:
                    # Aligned, move forward
                    if self.front_distance > 0.25 and not self.somethinginhand:
                        self.send_movement(min(0.1, 0.1 * self.front_distance), angular / 5)
                    else:
                        self.send_movement(0,0)                   
                        self.pick_up_object()
                                      
                

        ##### model logic ########
        # if self.is_trash:
        print("Trash detected, executing pick up.")
        if not self.somethinginhand:
            self.pick_up_object()
        # else:
            # rospy.loginfo("Object detected is not trash, skipping.")

    def send_movement(self, velocity, angular):
        """
        Sends a movement command to the robot with given forward (x-axis) velocity and angular velocity (z-axis) for convenience.
        """
        move_cmd = Twist()
        move_cmd.linear.x = velocity
        move_cmd.angular.z = angular
        self.cmd_vel_pub.publish(move_cmd)
        rospy.sleep(0.05)
        

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
        self.somethinginhand = True
        lift_position = [0, math.radians(-48), math.radians(-30), math.radians(-101)]  # Retract and lift the arm
        self.move_group_arm.go(lift_position, wait=True)
        rospy.sleep(6)  # Wait for the arm to lift to the safe position
        self.move_group_arm.stop()
        print("Object lifted.")
        # Open the gripper to throw away
        self.move_group_gripper.go([0.01, 0.01], wait=True)
        rospy.sleep(2)
        self.somethinginhand = False
        self.reset_arms()
        self.reset()

    def reset(self):
        print("Resetting to search for another object...")
        # Spin the robot to search for another object
        for _ in range(12):  # Spin for a fixed number of iterations
            self.twist.angular.z = 0.5  # Set a moderate spinning speed
            self.cmd_vel_pub.publish(self.twist)
            rospy.sleep(0.5)  # Spin for half a second per iteration
        self.twist.angular.z = 0  # Stop spinning
        self.cmd_vel_pub.publish(self.twist)
        print("Search reset complete, looking for new objects.")
        # After spinning, attempt to approach the closest object again
        self.run()

    def reset_arms(self):
        print("resetting arm angles")
        self.move_group_arm.go([0, 0, 0, 0], wait=True)
        self.move_group_gripper.go([0.01, 0.01], wait=True)
        self.move_group_arm.stop()
        self.move_group_gripper.stop()
        print("Done resetting arm angles")

    def model_loading(self, msg):
        self.loaded = msg.data
        print(f"Is the model loaded?: {self.loaded}\n")

    def run(self):
        while not self.loaded:
            print(f"Waiting for model to load\n")
        
        print("Starting the robot...")
        self.reset_arms()
        rospy.sleep(2)  # Wait for 2 seconds
        self.approach_closest_object()

if __name__ == "__main__":
    try:
        executor = Movement()
        executor.run()
    except rospy.ROSInterruptException:
        print("Movement node interrupted.")
