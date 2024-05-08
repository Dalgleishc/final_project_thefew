#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
import moveit_commander
import math
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
# from model import ..

class Movement:
    '''
    Class for the movement of the robot
    '''
    def __init__(self):
        rospy.init_node('movement_controller')
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.trash_detect = rospy.Subscriber('camera/rgb/image_raw', Image, self.detect_trash) # see bottom
        self.scan_subscriber = rospy.Subscriber('/scan', LaserScan, self.scan_callback) # see bottom
        self.twist = Twist()
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
        self.twist.angular.z = 0.5 # spin speed
        self.cmd_vel_pub.publish(self.twist)
        rospy.sleep(2)  # Spin for 2 seconds
        self.twist.angular.z = 0  # Stop the spin
        self.cmd_vel_pub.publish(self.twist)
        rospy.loginfo("Finished spinning.")

    def move_towards_object(self, distance):
        rospy.loginfo(f"Moving towards the object by {distance} meters...")
        self.twist.linear.x = 0.2  # Set forward speed
        rospy.sleep(distance / 0.2)
        self.twist.linear.x = 0  # Stop moving forward
        self.cmd_vel_pub.publish(self.twist)
        rospy.loginfo("Reached the object.")

    def throw_away(self):
        rospy.loginfo("Dumping the object into the bin at the back...")
        
        # Adjust the arm to a position behind the robot, suitable for dumping into the bin (with testing)
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

    def detect_trash(self):
        '''
        this is a subscriber to the robot's camera. import the model here to detect the trash 
        '''
        pass

    def scan_callback(self):
        '''
        this is a subscriber to the robot's lidar. 
        '''
        pass

    def run(self):
        '''
        Main logic:
        - Spin around to scan the environment
        - Move towards the closest object
        - Pick up the object
        '''
        while not rospy.is_shutdown():
            self.spin_around()
            # Implement logic to find the closest object and move towards it
            self.move_towards_object(1.0)  # Move 1 meter towards the object
            self.pick_up_object()


if __name__ == "__main__":
    robot_system = Movement()
    rospy.spin()