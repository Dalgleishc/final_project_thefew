#!/usr/bin/env python3

import rospy
import numpy as np
import time
import sys, time, cv2, cv_bridge, math, os, moveit_commander, moveit_msgs.msg
import cv2.aruco as aruco
from q_learning_project.msg import QLearningReward, RobotMoveObjectToTag
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

# Path of directory on where this file is located
path_prefix = os.path.dirname(__file__) + "/action_states/"

# Action # -> goal mapping
ACTION_MAP = [
    ('pink', 1),
    ('pink', 2),
    ('pink', 3),
    ('green', 1),
    ('green', 2),
    ('green', 3),
    ('blue', 1),
    ('blue', 2),
    ('blue', 3),
]

class perception(object):
    def __init__(self):
        # Initialize this node
        rospy.init_node("perception")
        self.terminal = False


        # color range dictionary
        self.color_range = {
            "blue": (np.array([85, 50, 130]), np.array([95, 255, 170])),
            "green": (np.array([30, 50, 130]), np.array([70, 255, 170])),
            "pink": (np.array([160,50,130]), np.array([170,255,170]))
        }
        self.last_im = time.time()
        self.since_last = 0.

        # set up ROS / OpenCV bridge
        self.bridge = cv_bridge.CvBridge()
        # initalize the debugging window
        cv2.namedWindow("window", cv2.WINDOW_NORMAL)

        # # load DICT_4X4_50
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

        # subscribe to the robot's RGB camera data stream
        self.last_image = None
        self.image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.image_callback)

        # initalize velocity publisher
        self.velocity_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)

        #initialize the subscriber to read the LiDAR messages
        self.scan_subscriber = rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        # load the q-matrix
        self.q_matrix = np.loadtxt(path_prefix + "q_matrix.csv", delimiter=',')
        self.action_matrix = np.loadtxt(path_prefix + "action_matrix.txt", dtype=int)

        # # inialize a publisher for actions
        # self.action_pub = rospy.Publisher('/q_learning/robot_action', RobotMoveObjectToTag, queue_size=10)

        # current color turple
        self.current_color = ("blue", "green", "pink")

        self.current_color_range = None

        self.cx = 0

        self.front_distance = 3.5


        ## First initialize `moveit_commander`:
        moveit_commander.roscpp_initialize(sys.argv)

        ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        scene = moveit_commander.PlanningSceneInterface()

        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to a planning group (group of joints).  In this tutorial the group is the primary
        ## arm joints in the Panda robot, so we set the group's name to "panda_arm".
        ## If you are using a different robot, change this value to the name of your robot
        ## arm planning group.

        ## This interface can be used to plan and execute motions:
        # the interface to the group of joints making up the turtlebot3
        # openmanipulator arm
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")


        self.something_in_hand = False
        self.goal_color = None
        self.goal_id = None
        self.state = 0
        self.next_state = 0
        self.get_next_move(self.state)


        # the interface to the group of joints making up the turtlebot3
        # openmanipulator gripper
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")

        # Reset arm position
        self.move_group_arm.go([0, -1, 0.9, 0], wait=True)

        rospy.sleep(2)
        # Reset gripper position
        self.move_group_gripper.go([0.01, 0], wait=True)
        self.want_tag = 2

        rospy.sleep(5)

        print(f"Set Up Done")

    def get_next_move(self, state):
        """
        Given the state number, it returns what the next move should be based on the Q matrix policy.
        This is acquired by the getting the best action as the argmax of the Q matrix row, then translating
        the action number based on the table given in the assignment. Then we update the robot's internal goal
        system. We also check if we are in a terminal state, and if so we mark that we are terminal.
        """
        if (self.action_matrix[state] == -1).sum() == 64:
            print('terminal state, stop')
            self.terminal = True
            return

        print(f'state row[{state}]: {self.q_matrix[state]}')
        next_action = np.argmax(self.q_matrix[state])
        print(f'next action: {next_action}')
        next_state = np.argwhere(self.action_matrix[state] == next_action).item()
        self.next_state = next_state
        print(f'next action: {next_action} next state: {next_state}')

        col, arid = ACTION_MAP[next_action]
        print(f'setting next goal: {col} -> {arid}')

        self.current_color_range = self.color_range[col]
        self.goal_color = col
        self.goal_id = arid


    def move_arm(self, pos_name, wait=7):
        """
        This is a wrapper around arm movement. The pos_name should be either "up", "down", which are the positions
        the arm should be in after and before picking up the toilet roll, resp. Anything else for pos_name results
        in the arm being set into the "default" position.
        """
        print(f'Moving arm: {pos_name}')
        # default pos
        move = [0, -1, 0.9, 0]

        if pos_name == 'down':
            move = [0, 0.5, -0.5, 0]
        elif pos_name == 'up':
            move = [0, -0.5, -0.5, 0]

        self.move_group_arm.go(move, wait=True)
        rospy.sleep(wait)

    def grip(self, wait=7):
        """
        Activates vice grip
        """
        print('GRIPPER CLOSE')
        self.move_group_gripper.go([-0.01, -0.01])
        rospy.sleep(wait)

    def ungrip(self, wait=7):
        """
        Opens gripper
        """
        print('GRIPPER OPEN')
        self.move_group_gripper.go([0.01, 0])
        rospy.sleep(wait)

    def pick_object(self):
        """
        Sequence of moves to pick up an object. Ungrip -> arm down -> grip -> arm up.
        """
        if self.something_in_hand:
            return
        self.ungrip()
        self.move_arm('down')
        self.grip()

        self.move_arm('up')
        self.something_in_hand = True
        rospy.sleep(1)
        print("okay")

    def drop_object(self):
        """
        Sequence of moves to drop object. Arm down -> ungrip -> arm reset
        """
        print('Dropping object')
        self.move_arm('down')
        self.ungrip()
        self.move_arm('reset')

    def spin_around(self):
        """
        Makes robot spin around. Default move when we don't see a goal, so we can search for it.
        """
        self.send_movement(0, -0.1)

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
            front_ranges = processed_ranges[front_index-10:front_index+10]

        # Calculate the minimum distance in the front range
        self.front_distance = min(front_ranges)

    def find_ar_tag(self, image):
        """
        Helper function that takes in an image and returns the corners and ids of the AR tags found.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rej = aruco.detectMarkers(gray, self.aruco_dict)
        if ids is None:
            return corners, ids
        return corners, ids
    
    def put_debug_overlay(self):
        """"
        Helper function which overlays the cv2 image window output with useful current stats and states.
        """
        h, w, _ = self.last_image.shape
        ox, oy = 3*w // 5, h//8
        color = (0, 255, 0)
        cv2.putText(self.last_image, f'{self.since_last:.2f}', (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,  1)
        cv2.putText(self.last_image, f'goal id: {self.goal_id}', (ox, oy+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,  1)
        cv2.putText(self.last_image, f'goal color: {self.goal_color}', (ox, oy+36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,  1)
        cv2.putText(self.last_image, f'holding: {self.something_in_hand}', (ox, oy+54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,  1)

    def image_callback(self, msg):
        """
        Callback for camera. It will just make robot do a dance if we are terminal. Otherwise, it will, depending on state, either
        1). Spin around looking for a goal. 
        2). See a color it needs to pickup, and go towards it. If it is close enough and aligned, it will attempt to pick up the color.
        3). See a AR tag it needs to drop off to, and go towards it. if it is close enough, it will drop the roll.
        """
        if self.terminal:
            while True:
                self.send_movement(0, 0.1)
                self.move_group_arm.go([1, -1, 0.9, 0], wait=True)
                for _ in range(12):
                    self.grip(wait=0.3)
                    self.ungrip(wait=0.3)
                self.move_group_arm.go([-1, -1, 0.9, 0], wait=True)
                rospy.sleep(2)
        self.since_last = time.time() - self.last_im
        self.last_im = time.time()

        # converts the incoming ROS message to OpenCV format and HSV (hue, saturation, value)
        image = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        h, w, d = image.shape
        self.last_image = image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        current_range = self.current_color_range

        if current_range is None:
            self.spin_around()
            print('setting up, or done')
            return

        self.put_debug_overlay() 
        cv2.waitKey(3)
        # checking to see if there is something in the range
        if not self.something_in_hand:
            lower = current_range[0]
            upper = current_range[1]

            # this erases all pixels that aren't the current color
            mask = cv2.inRange(hsv, lower, upper)

            # this limits our search scope to only view a slice of the image (just in front))
            # search_left = int(1*w/4)
            # search_right = int(3*w/4)
            # mask[0:h, :search_left] = 0
            # mask[0:h, search_right:] = 0

            # using moments() function, the center of the current color pixels is determined
            M = cv2.moments(mask)
            # if there are any current color pixels found
            if M['m00'] > 0:
                    # center of the current color pixels in the image
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    #print(f"cx: {cx}\t cy:{cy}\n")

                    # a red circle is visualized in the debugging window to indicate
                    cv2.circle(self.last_image, (cx, cy), 20, (0,0,255), -1)
                    cx_error = cx - w / 2
                    self.go_to(cx_error)

            else:
                self.spin_around()
        else:
            corners, ids = self.find_ar_tag(image)
            if ids is None:
                self.spin_around()
                cv2.imshow('window', self.last_image)
                print('no tag')
                return
            ids = [x[0] for x in ids]

            middles = [x.mean(axis=1)[0] for x in corners]
            ar_locs = {k:v for k,v in zip(ids, middles)}

            if self.goal_id not in ar_locs:
                self.spin_around()
                print('goal AR tag not found')
                cv2.imshow('window', self.last_image)
                return

            tx, ty= ar_locs[self.goal_id] # x value
            cv2.circle(self.last_image, (tx, ty), 10, (0, 255, 0), -1)
            cx_error = tx - w / 2

            self.go_to(cx_error/10)
        cv2.imshow('window', self.last_image)

    def go_to(self, px_error):
        """
        Makes the robot go to a certain location based on horizontal linear difference from image. 
        We use the front_distance attribute which is updated by LiDAR to know when we are close distance wise,
        while continuously correcting our angle based on where horizontal error.
        """
        #cx is the x position of where the object is
        #the robot needs to turn toward the object (have cx be in the middle of the screen)
        #then the robot needs to aproach the robot till until it is 0.1m away (scan lidar)
        # convert pixal error to angular error
        angular = -1 * (px_error / 100)

        # if not aligned
        if abs(angular) > 1.0:
            self.send_movement(0, angular)

        # is aligned
        else:
            # far from object, getting color
            if self.front_distance > 0.25 and not self.something_in_hand:
                self.send_movement(min(0.1, 0.1 * self.front_distance), angular / 5)

            # far from object, getting AR
            elif self.front_distance > 0.5 and self.something_in_hand:
                self.send_movement(min(0.1, 0.1 * self.front_distance), angular / 5)

            # close to object
            else:
                self.send_movement(0, 0)
                if not self.something_in_hand:
                    self.pick_object()
                else:
                    self.drop_object()
                    self.something_in_hand = False
                    self.state = self.next_state

                    self.send_movement(-0.5, 0)
                    rospy.sleep(2)

                    self.get_next_move(self.state)

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


    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        node = perception()
        rospy.sleep(1)
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        print(f"Done")
