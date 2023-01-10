#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import rclpy
from rclpy.node import Node
import sys
import matplotlib.pylab as plt

# Robot motion commands:
# http://docs.ros.org/api/geometry_msgs/html/msg/Twist.html
from geometry_msgs.msg import Twist
# Occupancy grid.
from nav_msgs.msg import OccupancyGrid
# Position.
import tf2_ros
# Goal.
from geometry_msgs.msg import PoseStamped
# Path.
from nav_msgs.msg import Path
# For pose information.
from scipy.spatial.transform import Rotation as R

# Import the potential_field.py code rather than copy-pasting.
directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../_python_exercises')
sys.path.insert(0, directory)
try:
    import rrt as rrt
except ImportError:
    raise ImportError('Unable to import rrt.py. Make sure this file is in "{}"'.format(directory))

SPEED = .2
EPSILON = .1

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

def get_velocity(position, path_points):
    v = np.zeros_like(position)
    if len(path_points) == 0:
        return v
    # Stop moving if the goal is reached.
    if np.linalg.norm(position - path_points[-1]) < .2:
        return v

    # MISSING: Return the velocity needed to follow the
    # path defined by path_points. Assume holonomicity of the
    # point located at position.

    # Solution:
    if len(path_points) > 1:
        distances = list(map(lambda x: np.linalg.norm(x-position), path_points))
        i = distances.index(np.max(distances))
        if i == 0:
            j = i+1
        elif i == len(distances)-1:
            j = i-1
        elif distances[i+1] < distances[i-1]:
            j = i+1
        else:
            j = i-1
        v = 0.75 * SPEED * (path_points[i] - path_points[j])*np.sign(i-j)/np.linalg.norm(path_points[i] - path_points[j])
    else:
        v = 0.75 * SPEED * (path_points[0] - position) / np.linalg.norm(path_points[0] - position)
    return v


class SLAM(object):
    def __init__(self, node, use_webots):
        self._node = node
        self._buffer = tf2_ros.Buffer()
        self._tf = tf2_ros.TransformListener(self._buffer, node)
        self._occupancy_grid = None
        self._pose = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        self._use_webots = use_webots

    def callback(self, msg):
        values = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        processed = np.empty_like(values)
        processed[:] = rrt.FREE
        processed[values < 0] = rrt.UNKNOWN
        processed[values > 50] = rrt.OCCUPIED
        if not self._use_webots:
            processed = np.transpose(processed)
        origin = [msg.info.origin.position.x, msg.info.origin.position.y, 0.]
        resolution = msg.info.resolution
        self._occupancy_grid = rrt.OccupancyGrid(processed, origin, resolution)

    def update(self, draw_map=False):
        # Display the current occupancy grid for debugging
        if draw_map and self._occupancy_grid is not None:
            plt.clf()
            self._occupancy_grid.draw()
            plt.scatter(self._pose[X], self._pose[Y], s=10, marker='o', color='green', zorder=1000)
            plt.ion()
            plt.show()
            plt.pause(0.001)
        # Get pose w.r.t. map.
        a = 'occupancy_grid'
        b = 'base_link'
        try:
            t = self._buffer.lookup_transform(
                a,
                b,
                rclpy.time.Time())


            self._pose[X] = t.transform.translation.x
            self._pose[Y] = t.transform.translation.y
            _, _, self._pose[YAW] = R.from_quat([
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w
            ]).as_euler("XYZ")

        except tf2_ros.TransformException as ex:
            self._node.get_logger().info(
                f'Could not transform {a} to {b}: {ex}')
            return

    @property
    def ready(self):
        return self._occupancy_grid is not None and not np.isnan(self._pose[0])

    @property
    def pose(self):
        return self._pose

    @property
    def occupancy_grid(self):
        return self._occupancy_grid


class GoalPose(object):
    def __init__(self):
        self._position = np.array([np.nan, np.nan], dtype=np.float32)

    def callback(self, msg):
        # The pose from RViz is with respect to the "map".
        self._position[X] = msg.pose.position.x
        self._position[Y] = msg.pose.position.y
        print('Received new goal position:', self._position)

    @property
    def ready(self):
        return not np.isnan(self._position[0])

    @property
    def position(self):
        return self._position


def get_path(final_node):
    # Construct path from RRT solution.
    if final_node is None:
        return np.array([])
    path_reversed = []
    path_reversed.append(final_node)
    while path_reversed[-1].parent is not None:
        path_reversed.append(path_reversed[-1].parent)
    path = list(reversed(path_reversed))
    # Put a point every 5 cm.
    distance = 0.05
    offset = 0.
    points_x = []
    points_y = []
    for u, v in zip(path, path[1:]):
        center, radius = rrt.find_circle(u, v)
        du = u.position - center
        theta1 = np.arctan2(du[1], du[0])
        dv = v.position - center
        theta2 = np.arctan2(dv[1], dv[0])
        # Check if the arc goes clockwise.
        clockwise = np.cross(u.direction, du).item() > 0.
        # Generate a point every 5cm apart.
        da = distance / radius
        offset_a = offset / radius
        if clockwise:
            da = -da
            offset_a = -offset_a
            if theta2 > theta1:
                theta2 -= 2. * np.pi
        else:
            if theta2 < theta1:
                theta2 += 2. * np.pi
        angles = np.arange(theta1 + offset_a, theta2, da)
        offset = distance - (theta2 - angles[-1]) * radius
        points_x.extend(center[X] + np.cos(angles) * radius)
        points_y.extend(center[Y] + np.sin(angles) * radius)
    return np.stack( [np.array(points_x), np.array(points_y)], axis=1 )


class RTTNavigation(Node):

    def __init__(self, use_webots):
        super().__init__('rtt_navigation')

        # Update control every 100 ms.
        self._rate_limiter = self.create_timer(timer_period_sec=0.1, callback=self._timer_callback)
        self._publisher = self.create_publisher(Twist, 'cmd_vel', 5)
        self._path_publisher = self.create_publisher(Path, 'path', 1)

        self._slam = SLAM(node=self,use_webots=use_webots)
        self._slam_subscription = self.create_subscription(OccupancyGrid, 'map', self._slam.callback, 1)

        self._goal = GoalPose()
        self._goal_subscription = self.create_subscription(PoseStamped, '/move_base_simple/goal',
                                                           self._goal.callback, 1)

        self._frame_id = 0
        self._current_path = np.array([]);
        self._previous_time, _ = self.get_clock().now().seconds_nanoseconds()

        # Stop moving message.
        self._stop_msg = Twist()
        self._stop_msg.linear.x = 0.0
        self._stop_msg.angular.z = 0.0

        # Make sure the robot is stopped.
        self._i = 0
        print( "RRT class init!" );

    def _timer_callback(self):
        # wait about 1s for map to be populated
        if self._i < 10:
            self._i += 1
            self._publisher.publish(self._stop_msg)
            return

        self._slam.update( draw_map=False )
        current_time, _ = self.get_clock().now().seconds_nanoseconds()

        # Make sure all measurements are ready.
        # Get map and current position through SLAM:
        # This generally comes from having started slam.launch
        if not self._goal.ready or not self._slam.ready:
            print("SLAM not ready")
            return

        goal_reached = np.linalg.norm(self._slam.pose[:2] - self._goal.position) < .2
        if goal_reached:
            self._publisher.publish(self._stop_msg)
            return

        # Follow path using an assumed feedback linearization.
        position = np.array([
            self._slam.pose[X] + EPSILON * np.cos(self._slam.pose[YAW]),
            self._slam.pose[Y] + EPSILON * np.sin(self._slam.pose[YAW])], dtype=np.float32)
        v = get_velocity(position, np.array(self._current_path, dtype=np.float32))
        u, w = feedback_linearized(self._slam.pose, v, epsilon=EPSILON)
        vel_msg = Twist()
        vel_msg.linear.x = float(u)
        vel_msg.angular.z = float(w)
        self._publisher.publish(vel_msg)

        # Update plan every 1s.
        time_since = current_time - self._previous_time
        if self._current_path.any() and time_since < 2.:
            return

        self._previous_time = current_time

        # Run RRT.
        start_node, final_node = rrt.rrt(self._slam.pose, self._goal.position, self._slam.occupancy_grid)
        self._current_path = get_path(final_node)
        if not self._current_path.any():
            print('Unable to reach goal position:', self._goal.position)

        # Publish path to RViz.
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        for u in self._current_path:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = path_msg.header.stamp
            pose_msg.header.frame_id = 'map'
            pose_msg.pose.position.x = u[X]
            pose_msg.pose.position.y = u[Y]
            path_msg.poses.append(pose_msg)
        self._path_publisher.publish(path_msg)

        # can be used for debugging or book-keeping
        self._frame_id += 1


def run(args):
    rclpy.init()
    rtt_navigation_node = RTTNavigation(args.use_webots)

    rclpy.spin(rtt_navigation_node)

    rtt_navigation_node.destroy_node()
    rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Runs RRT navigation')
    parser.add_argument('--use_webots', action='store_true', default=False,help='Whether webots is used')
    args, unknown = parser.parse_known_args()
    run(args)


if __name__ == '__main__':
    main()
