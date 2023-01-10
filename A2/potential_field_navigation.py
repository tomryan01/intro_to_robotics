#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
import sys
import matplotlib.pylab as plt

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# For pose information.
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry

# Import the potential_field.py code rather than copy-pasting.
directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../_python_exercises')
sys.path.insert(0, directory)
try:
    import potential_field as potential_field
except ImportError:
    raise ImportError('Unable to import potential_field.py. Make sure this file is in "{}"'.format(directory))

WALL_OFFSET = 2.
ROBOT_RADIUS = 0.105 / 2.
CYLINDER_POSITION = np.array([.5, .6], dtype=np.float32)
CYLINDER_RADIUS = .3
GOAL_POSITION = np.array([1.5, 1.5], dtype=np.float32)
MAX_SPEED = .25
EPSILON = .2

USE_RELATIVE_POSITIONS = True

X = 0
Y = 1
YAW = 2


def feedback_linearized(pose, velocity, epsilon):
    u = 0.  # [m/s]
    w = 0.  # [rad/s] going counter-clockwise.

    # MISSING: Implement feedback-linearization to follow the velocity
    # vector given as argument. Epsilon corresponds to the distance of
    # linearized point in front of the robot.

    # Solution:
    u = velocity[X]*np.cos(pose[YAW]) + velocity[Y]*np.sin(pose[YAW])
    w = (1/epsilon)*(velocity[Y]*np.cos(pose[YAW]) - velocity[X]*np.sin(pose[YAW]))
    return u, w

def get_relative_position(absolute_pose, absolute_position):
    relative_position = absolute_position.copy()

    # MISSING: Compute the relative position of absolute_position in the
    # coordinate frame defined by absolute_pose.

    # Solution:
    r = np.array([absolute_position[X]-absolute_pose[X],absolute_position[Y]-absolute_pose[Y]])
    alpha = np.arctan2(r[1],r[0])
    relative_position[X] = np.linalg.norm(r)*np.cos(absolute_pose[YAW]-alpha)
    relative_position[Y] = -np.linalg.norm(r)*np.sin(absolute_pose[YAW]-alpha)
    return np.array(relative_position.tolist(), dtype=np.float64)


class GroundtruthPose(object):
    def __init__(self):
        self._pose = [np.nan, np.nan, np.nan]

    def callback(self, msg):
        self._pose[0] = msg.pose.pose.position.x
        self._pose[1] = msg.pose.pose.position.y
        _, _, self._pose[2] = R.from_quat([
                                msg.pose.pose.orientation.x,
                                msg.pose.pose.orientation.y,
                                msg.pose.pose.orientation.z,
                                msg.pose.pose.orientation.w]).as_euler('XYZ')

    @property
    def pose(self):
        return self._pose

    @property
    def ready(self):
        return not np.isnan(self._pose[0])


def get_velocity(point_position, goal_position, obstacle_position):
    v_goal = potential_field.get_velocity_to_reach_goal(point_position, goal_position)
    v_avoid = potential_field.get_velocity_to_avoid_obstacles(point_position, [obstacle_position],
                                                              [CYLINDER_RADIUS + ROBOT_RADIUS])
    v = v_goal + v_avoid
    return potential_field.cap(v, max_speed=MAX_SPEED)


class PotentialFieldNavigation(Node):
    def __init__(self, args):
        super().__init__('potential_field_navigation')

        self._publisher = self.create_publisher(Twist, 'cmd_vel', 5)
        # Keep track of groundtruth position for plotting purposes.
        self._groundtruth = GroundtruthPose()
        self._groundtruth_subscriber = self.create_subscription(Odometry, 'odom',
                                                                self._groundtruth.callback, 5)

        share_tmp_dir = os.path.join(get_package_share_directory('part2'), 'tmp')
        os.makedirs(share_tmp_dir, exist_ok=True)
        file_path = os.path.join(share_tmp_dir, 'potential_field_logging.txt')
        self._temp_file = file_path
        self._pose_history = []
        self._vis = False;
        self._rate_limiter = self.create_timer(timer_period_sec=0.1, callback=self.timer_callback)

        with open(self._temp_file, 'w+'):
          pass

    def timer_callback(self):
        # Make sure all measurements are ready.
        if not self._groundtruth.ready:
            return

        if not self._vis:
            self._vis = True;
            # Plot field only once
            pX, pY = np.meshgrid(np.linspace(-WALL_OFFSET, WALL_OFFSET, 30),
                     np.linspace(-WALL_OFFSET, WALL_OFFSET, 30))
            pU = np.zeros_like(pX)
            pV = np.zeros_like(pX)
            for i in range(len(pX)):
                for j in range(len(pX[0])):
                    velocity = get_velocity(np.array([pX[i, j], pY[i, j]]), GOAL_POSITION, CYLINDER_POSITION)
                    pU[i, j] = velocity[0]
                    pV[i, j] = velocity[1]
            plt.quiver(pX, pY, pU, pV, units='width')
            plt.show()

        absolute_point_position = np.array([
            self._groundtruth.pose[X] + EPSILON * np.cos(self._groundtruth.pose[YAW]),
            self._groundtruth.pose[Y] + EPSILON * np.sin(self._groundtruth.pose[YAW])], dtype=np.float32)

        if USE_RELATIVE_POSITIONS:
            point_position = get_relative_position(self._groundtruth.pose, absolute_point_position)
            goal_position = get_relative_position(self._groundtruth.pose, GOAL_POSITION)
            obstacle_position = get_relative_position(self._groundtruth.pose, CYLINDER_POSITION)
            pose = np.array([0., 0., 0.], dtype=np.float32)
        else:
            point_position = absolute_point_position
            goal_position = GOAL_POSITION
            obstacle_position = CYLINDER_POSITION
            pose = self._groundtruth.pose

        # Get velocity.
        v = get_velocity(point_position, goal_position, obstacle_position)

        u, w = feedback_linearized(pose, v, epsilon=EPSILON)
        vel_msg = Twist()
        vel_msg.linear.x = u
        vel_msg.angular.z = w
        self._publisher.publish(vel_msg)

        # Log groundtruth positions in tmp/potential_field_logging.txt
        self._pose_history.append(self._groundtruth.pose)
        if len(self._pose_history) % 10:
          with open(self._temp_file, 'a') as fp:
            fp.write('\n'.join(','.join(str(v) for v in p) for p in self._pose_history) + '\n')
            self._pose_history = []


def run(args):
    rclpy.init()

    potential_field_navigation_node = PotentialFieldNavigation(args)

    rclpy.spin(potential_field_navigation_node)

    potential_field_navigation_node.destroy_node()
    rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Runs potential field navigation')
    args, unknown = parser.parse_known_args()
    run(args)


if __name__ == '__main__':
    main()
