#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import moveit_commander
import math

class Movement:
    def __init__(self):
        rospy.init_node('movement_controller')
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.twist = Twist()
        self.current_scan = None
        self.stop_distance = 0.2  # Stop about 0.2 meters away
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")
        self.initialize_robot()

    def initialize_robot(self):
        rospy.loginfo("Initializing robot arm and gripper to default states...")
        self.move_group_arm.go([0, 0, 0, 0], wait=True)
        self.move_group_gripper.go([0.01, 0.01], wait=True)
        self.move_group_arm.stop()
        self.move_group_gripper.stop()
        rospy.loginfo("Robot initialized to home position with gripper open.")

    def lidar_callback(self, msg):
        self.current_scan = msg

    def find_closest_object(self):
        if self.current_scan is None:
            rospy.loginfo("No LIDAR data")
            return None
        ranges = self.current_scan.ranges
        min_distance = float('inf')
        min_angle = None
        for i, distance in enumerate(ranges):
            if 0.05 < distance < min_distance:  # Ignore zero and very close readings
                min_distance = distance
                min_angle = i
        if min_angle is None:
            return None
        return min_distance, min_angle

    def approach_closest_object(self):
        rospy.loginfo("Approaching the closest object.")
        while not rospy.is_shutdown():
            closest_object = self.find_closest_object()
            if closest_object:
                min_distance, min_angle = closest_object
                if min_distance < self.stop_distance:
                    # Check the next few readings to confirm consistency within stop distance
                    consistency_count = 0
                    for i in range(1, 5):  # Check the next 4 readings around the minimum angle
                        index = (min_angle + i) % len(self.current_scan.ranges)
                        if self.current_scan.ranges[index] <= self.stop_distance:
                            consistency_count += 1
                        index = (min_angle - i) % len(self.current_scan.ranges)
                        if self.current_scan.ranges[index] <= self.stop_distance:
                            consistency_count += 1
                    if consistency_count >= 6:  # If at least 6 out of 8 surrounding points are within the distance
                        self.twist.linear.x = 0
                        self.cmd_vel_pub.publish(self.twist)
                        rospy.loginfo(f"Stopped close to the object at distance {min_distance} meters with consistent readings.")
                        break
                    else:
                        rospy.loginfo("Inconsistent object distance readings, continuing approach.")
                else:
                    self.twist.linear.x = 0.1  # Move forward at a constant speed
                    self.twist.angular.z = 0
                    self.cmd_vel_pub.publish(self.twist)
                    rospy.loginfo(f"Current distance to object: {min_distance} meters")
            else:
                rospy.loginfo("No objects detected.")
            rospy.sleep(0.1)
        self.twist.linear.x = 0
        self.cmd_vel_pub.publish(self.twist)
        rospy.loginfo("Final stop command issued.")
        self.pick_up_object()


    def pick_up_object(self):
        rospy.loginfo("Approaching the object...")
        # Extend arm forward and as low as possible while respecting joint limits
        move_downward_position = [0, math.radians(82), math.radians(-43), math.radians(-11)]  # Extend arm downwards
        # Move arm to the extended forward position
        self.move_group_arm.go(move_downward_position, wait=True)
        rospy.sleep(3)  # Wait for the arm to reach the extended position
        self.move_group_arm.stop()
        rospy.loginfo("downward position reached.")
        self.move_group_gripper.go([-0.01, -0.01], wait=True)  # Use maximum closure within limits
        rospy.sleep(3)  # Allow time for the gripper to close
        self.move_group_gripper.stop()
        rospy.loginfo("Gripper clenched. Lifting the object...")
        # Lift the arm to clear any obstacles
        lift_position = [0, math.radians(-40), math.radians(-54), math.radians(-101)]  # Retract and lift the arm
        self.move_group_arm.go(lift_position, wait=True)
        rospy.sleep(3)  # Wait for the arm to lift to the safe position
        self.move_group_arm.stop()
        rospy.loginfo("Object lifted.")
        # Open the gripper to throw away
        # self.move_group_gripper.go([0.01, 0.01], wait=True)


    def run(self):
        rospy.loginfo("Starting the robot...")
        rospy.sleep(2)  # Wait for 2 seconds to stabilize
        self.approach_closest_object()

if __name__ == "__main__":
    executor = Movement()
    executor.run()
    rospy.spin()
