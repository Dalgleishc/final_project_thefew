#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import moveit_commander
import math
from sensor_msgs.msg import Image, LaserScan

class Movement:
    '''
    Class for the movement of the robot
    '''
    def __init__(self):
        rospy.init_node('movement_controller')
        rospy.loginfo("Initializing Movement Node")
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.trash_detect = rospy.Subscriber('camera/rgb/image_raw', Image, self.detect_trash)
        self.scan_subscriber = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.twist = Twist()
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")
        self.current_scan = None
        self.initialize_robot()

    def initialize_robot(self):
        rospy.loginfo("Initializing robot arm and gripper to default states...")
        self.move_group_arm.go([0, 0, 0, 0], wait=True)
        self.move_group_gripper.go([0.01, 0.01], wait=True)
        self.move_group_arm.stop()
        self.move_group_gripper.stop()
        rospy.loginfo("Robot initialized to home position with gripper open.")

    def pick_up_object(self):
        rospy.loginfo("Picking up the object...")
        self.move_group_arm.go([0, math.radians(18), math.radians(2), math.radians(0)], wait=True)
        rospy.sleep(5)
        self.move_group_gripper.go([-0.01, -0.01], wait=True)
        rospy.sleep(5)
        self.move_group_arm.go([0, -0.5, 0, 0], wait=True)
        rospy.sleep(5)
        self.move_group_arm.stop()
        self.move_group_gripper.stop()
        rospy.loginfo("Object picked up successfully.")

    def spin_around(self):
        rospy.loginfo("Spinning the robot...")
        self.twist.angular.z = 0.5  # Spin speed
        self.cmd_vel_pub.publish(self.twist)
        rospy.sleep(2)  # Spin for 2 seconds
        self.twist.angular.z = 0  # Stop the spin
        self.cmd_vel_pub.publish(self.twist)
        rospy.loginfo("Finished spinning.")

    def move_towards_object(self, distance):
        rospy.loginfo(f"Moving towards the object by {distance} meters...")
        self.twist.linear.x = 0.2  # Set forward speed
        self.cmd_vel_pub.publish(self.twist)
        rospy.sleep(distance / 0.2)
        self.twist.linear.x = 0  # Stop moving forward
        self.cmd_vel_pub.publish(self.twist)
        rospy.loginfo("Reached the object.")

    def throw_away(self):
        rospy.loginfo("Dumping the object into the bin at the back...")
        
        # Adjust the arm to a position behind the robot, suitable for dumping into the bin
        dump_position = [0, -math.radians(30), -math.radians(10), math.radians(0)]  # Example: adjust angles as necessary
        self.move_group_arm.go(dump_position, wait=True)
        rospy.sleep(5)
        
        # Open the gripper to release the object into the bin
        self.move_group_gripper.go([0.01, 0.01], wait=True)
        rospy.sleep(5)
        
        # Return the arm to a default position
        self.move_group_arm.go([0, 0, 0, 0], wait=True)
        self.move_group_gripper.go([0.01, 0.01], wait=True)
        self.move_group_arm.stop()
        self.move_group_gripper.stop()
        
        rospy.loginfo("Object dumped successfully into the bin.")

    def detect_trash(self, data):
        '''
        This is a placeholder for the actual trash detection logic
        '''
        rospy.loginfo("Detecting trash...")

    def scan_callback(self, data):
        '''
        This callback processes LIDAR data to find the closest object
        '''
        self.current_scan = data

    def find_closest_object(self):
        if self.current_scan is None:
            return None
        ranges = self.current_scan.ranges
        min_distance = float('inf')
        min_angle = None
        for i, distance in enumerate(ranges):
            if distance < min_distance:
                min_distance = distance
                min_angle = i
        return min_distance, min_angle

    def run(self):
        '''
        Main logic:
        - Spin around to scan the environment
        - Move towards the closest object
        - Pick up the object
        - Throw away the object
        '''
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # Spin around to find objects
            self.spin_around()
            closest_object = self.find_closest_object()
            if closest_object is not None:
                min_distance, min_angle = closest_object
                rospy.loginfo(f"Closest object at distance: {min_distance}, angle: {min_angle}")
                self.move_towards_object(min_distance)
                self.pick_up_object()
                self.throw_away()
            else:
                rospy.loginfo("No objects detected.")
            rate.sleep()

if __name__ == "__main__":
    robot_system = Movement()
    robot_system.run()
    rospy.spin()
