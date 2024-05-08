#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import moveit_commander
import math

class Movement:
    def __init__(self):
        rospy.init_node('robot_policy_executor')
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.twist = Twist()
        self.current_scan = None
        self.safe_distance = 0.15
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
            return None
        ranges = self.current_scan.ranges
        min_distance = float('inf')
        min_angle = None
        for i, distance in enumerate(ranges):
            if 0.05 < distance < min_distance:  # Ignore zero and very close readings
                min_distance = distance
                min_angle = i
        return min_distance, min_angle

    def approach_closest_object(self, min_distance, min_angle):
        angular_speed = 0.3
        linear_speed = 0.1
        target_angle = min_angle - 180 if min_angle > 180 else min_angle
        rospy.loginfo(f"Approaching object at angle {min_angle} with distance {min_distance}")
        while not rospy.is_shutdown():
            if min_distance < self.safe_distance:
                self.twist.linear.x = 0
                self.twist.angular.z = 0
                self.cmd_vel_pub.publish(self.twist)
                rospy.loginfo("Object is within safe distance, stopping.")
                break
            self.twist.linear.x = linear_speed
            self.twist.angular.z = -angular_speed if target_angle > 0 else angular_speed
            self.cmd_vel_pub.publish(self.twist)
            rospy.sleep(0.1)
            min_distance, min_angle = self.find_closest_object()

    def pick_up_object(self):
        rospy.loginfo("Picking up the object...")
        self.move_group_arm.go([0, math.radians(18), math.radians(2), math.radians(0)], wait=True)
        rospy.sleep(2)
        self.move_group_gripper.go([-0.01, -0.01], wait=True)
        rospy.sleep(2)
        self.move_group_arm.go([0, -0.5, 0, 0], wait=True)
        rospy.sleep(2)
        self.move_group_arm.stop()
        self.move_group_gripper.stop()
        rospy.loginfo("Object picked up successfully.")

    def throw_away(self):
        rospy.loginfo("Dumping the object into the bin at the back...")
        dump_position = [0, -math.radians(30), -math.radians(10), math.radians(0)]
        self.move_group_arm.go(dump_position, wait=True)
        rospy.sleep(2)
        self.move_group_gripper.go([0.01, 0.01], wait=True)
        rospy.sleep(2)
        self.move_group_arm.go([0, 0, 0, 0], wait=True)
        self.move_group_gripper.go([0.01, 0.01], wait=True)
        self.move_group_arm.stop()
        self.move_group_gripper.stop()
        rospy.loginfo("Object dumped successfully into the bin.")

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            closest_object = self.find_closest_object()
            if closest_object:
                min_distance, min_angle = closest_object
                rospy.loginfo(f"Closest object at distance: {min_distance}, angle: {min_angle}")
                self.approach_closest_object(min_distance, min_angle)
                self.pick_up_object()
                self.throw_away()
            else:
                rospy.loginfo("No objects detected.")
            rate.sleep()

if __name__ == "__main__":
    executor = Movement()
    executor.run()
    rospy.spin()
