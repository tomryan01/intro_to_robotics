#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pdb

import numpy as np
import rclpy
import os
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# Laser scan message:
# http://docs.ros.org/api/sensor_msgs/html/msg/LaserScan.html
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

# For groundtruth information.
from scipy.spatial.transform import Rotation as R

import math

def braitenberg(front, front_left, front_right, left, right):
    u = 0.  # [m/s]
    w = 0.  # [rad/s] going counter-clockwise.

    # MISSING: Implement a braitenberg controller that takes the range
    # measurements given in argument to steer the robot.
    max_u = .2
    max_w = np.pi / 4.
    safety_dist = .5

    I = [front, front_left, front_right, left, right]

    for i in range(5):
        if I[i] > 4.0:
            I[i] = 0
    
    print(I)

    w_1 = [2,0.5,0.5,0,0]
    w_2 = [0,0.5,-0.5,0.4,-0.4]
    c_1 = -2
    c_2 = 0

    x_1 = sum([w_1[i]*I[i] for i in range(5)])
    x_2 = sum([w_2[i]*I[i] for i in range(5)])
    
    u = max_u * math.tanh(x_1 + c_1)
    w = max_w * math.tanh(x_2 + c_2)

    # Solution:
    return u, w


def rule_based(front, front_left, front_right, left, right):
    u = 0.  # [m/s]
    w = 0.  # [rad/s] going counter-clockwise.

    # MISSING: Implement a rule-based controller that avoids obstacles.
    max_u = .2
    max_w = np.pi / 4.
    safety_dist = .5

    # Solution:
    if front < 9999 or front_left < 9999 or front_right < 9999 or left < 9999 or right < 9999:
        if front < 0.5:
            u = max_u / 3
            if right >= left:
                w = -max_w
            else:
                w = max_w
        elif front_left < 0.5 or front_right < 0.5:
            u = max_u / 3
            if front_right >= front_left:
                w = -max_w
            else:
                w = max_w
        else:
            u = max_u
    return u, w


class SimpleLaser(Node):
    def __init__(self):
        super().__init__('simple_laser')#try subnode
        self._angles = [0., np.pi / 4., -np.pi / 4., np.pi / 2., -np.pi / 2.]
        self._width = np.pi / 180. * 10.  # 10 degrees cone of view.
        self._measurements = [float('inf')] * len(self._angles)
        self._indices = None

    def callback(self, msg):
        # Helper for angles.
        def _within(x, a, b):
            pi2 = np.pi * 2.
            x %= pi2
            a %= pi2
            b %= pi2
            if a < b:
                return a <= x <= b
            return a <= x or x <= b

        # Compute indices the first time.
        if self._indices is None:
            self._indices = [[] for _ in range(len(self._angles))]
            for i, d in enumerate(msg.ranges):
                angle = msg.angle_min + i * msg.angle_increment
                for j, center_angle in enumerate(self._angles):
                    if _within(angle, center_angle - self._width / 2., center_angle + self._width / 2.):
                        self._indices[j].append(i)

        ranges = np.array(msg.ranges)
        for i, idx in enumerate(self._indices):
            # np.percentile outputs NaN if there are less than 2 actual numbers in the input
            if np.isinf(ranges[idx]).sum() <= len(ranges[idx]) - 2:
                # We do not take the minimum range of the cone but the 10-th percentile for robustness.
                self._measurements[i] = np.nanpercentile(ranges[idx], 10)
            else:
                self._measurements[i] = np.inf

    @property
    def measurements(self):
        return self._measurements


class GroundtruthPose(Node):
    def __init__(self):
        super().__init__('groundtruth_pose')
        self._pose = [np.nan, np.nan, np.nan]

    def callback(self, msg):
        self._pose[0] = msg.pose.pose.position.x
        self._pose[1] = msg.pose.pose.position.y
        _, _, yaw = R.from_quat([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z + 1,
            msg.pose.pose.orientation.w]).as_rotvec()
        self._pose[2] = yaw

    @property
    def pose(self):
        return self._pose


class ObstacleAvoidance(Node):

    def __init__(self, args):
        super().__init__('obstacle_avoidance')
        self._avoidance_method = globals()[args.mode]
        self._laser = SimpleLaser()
        self._laser_subscriber = self.create_subscription(LaserScan, 'TurtleBot3Burger/scan', self._laser.callback, 5)
        self._publisher = self.create_publisher(Twist, 'cmd_vel', 5)
        self._groundtruth = GroundtruthPose()
        self._groundtruth_subscriber = self.create_subscription(Odometry, 'diffdrive_controller/odom',
                                                                self._groundtruth.callback, 5)
        share_tmp_dir = os.path.join(get_package_share_directory('part1'), 'tmp')
        os.makedirs(share_tmp_dir, exist_ok=True)
        file_path = os.path.join(share_tmp_dir, 'webots_exercise.txt')
        self._temp_file = file_path
        self._pose_history = []
        self._rate_limiter = self.create_timer(timer_period_sec=0.1, callback=self.timer_callback)

        # Keep track of groundtruth position for plotting purposes.
        with open(self._temp_file, 'w+'):
            pass

    def timer_callback(self):
        u, w = self._avoidance_method(*self._laser.measurements)
        vel_msg = Twist()
        vel_msg.linear.x = u
        vel_msg.angular.z = w
        self._publisher.publish(vel_msg)

        # Log groundtruth positions in: tmp/webots_exercise.txt
        self._pose_history.append(self._groundtruth.pose)
        if len(self._pose_history) % 10:
            with open(self._temp_file, 'a') as fp:
                fp.write('\n'.join(','.join(str(v) for v in p) for p in self._pose_history) + '\n')
                self._pose_history = []


def run(args):
    rclpy.init()
    obstacle_avoidance = ObstacleAvoidance(args)

    rclpy.spin(obstacle_avoidance)

    obstacle_avoidance.destroy_node()
    rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Runs obstacle avoidance')
    parser.add_argument('--mode', action='store', default='braitenberg', help='Method.',
                        choices=['braitenberg', 'rule_based'])
    args, unknown = parser.parse_known_args()
    run(args)



if __name__ == '__main__':
    main()
