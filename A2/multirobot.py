#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import rclpy
import threading
import functools
from rclpy.node import Node

# For pose information.
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu

ROBOT_RADIUS = 0.105 / 2.
ROBOT_COUNT = 5
DESTINATION_AREA_SIDE_OFFSET = 3.
MAX_SPEED = .5
EPSILON = .1
RESOLUTION = .1

X = 0
Y = 1
YAW = 2

def feedback_linearized(pose, velocity, epsilon):
    u = 0.  # [m/s]
    w = 0.  # [rad/s] going counter-clockwise.

    u = velocity[X]*np.cos(pose[YAW]) + velocity[Y]*np.sin(pose[YAW])
    w = (1/epsilon)*(velocity[Y]*np.cos(pose[YAW]) - velocity[X]*np.sin(pose[YAW]))
    return u, w

def get_topic(n, channel='/gps'):
    if n == 1:
        k = 2
    elif n == 2:
        k = 4
    elif n == 3:
        k = 1
    elif n == 4:
        k = 3
    else:
        k = 0
    return '/robot' + str(n) + '/robot' + str(k) + channel

def get_grid_pos(i, j):
    return [i*RESOLUTION + RESOLUTION/2, j*RESOLUTION + RESOLUTION/2]

def pos_from_grid(velocity):
    return int(np.floor(velocity[X]/0.1)), int(np.floor(velocity[Y]/0.1))

class Robot(object):
    def __init__(self, name: str):
        self._name = name
        self._pose = [0., 0., 0.]
        self.offset = [0., 0., 0.]
        self.velocity = [0., 0.]

    def pose_callback(self, msg):
        self._pose[0] = msg.pose.pose.position.x + self.offset[0]
        self._pose[1] = msg.pose.pose.position.y + self.offset[1]
        _, _, yaw = R.from_quat([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w]).as_euler('XYZ')
        self._pose[2] = yaw + self.offset[2]

    def _is_in_vo(self, otherPose, otherVelocity):
        a = np.array(self._pose[:2]) - np.array(otherPose[:2])
        b = np.array(otherVelocity) - np.array(self.velocity)
        d = np.linalg.norm(a)
        theta = np.arctan(2*ROBOT_RADIUS/d)
        alpha = np.arccos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
        if np.abs(alpha) <= np.abs(theta):
            return True
        else:
            return False

    def is_in_rvo(self, vDash, otherPose, otherVelocity):
        return self._is_in_vo(otherPose, (2*np.array(vDash) - np.array(otherVelocity).tolist()))

    @property
    def get_name(self):
        return self._name

    @property
    def pose(self):
        return self._pose

    @property
    def ready(self):
        return not np.isnan(self._pose[0])

class MultirobotDriver(Node):
    def __init__(self, args):
        super().__init__('multirobot_driver')
        self._destinations = self._generate_destinations()
        self._points = []
        self._yaws = []
        self._robots = [Robot('robot' + str(x)) for x in range(ROBOT_COUNT)]
        self._set_initial_pose(0)
        self._pose_subscribers = [self.create_subscription(Odometry, robot.get_name + '/odom',
                                                           robot.pose_callback, 5) for robot in self._robots]
        self._rate_limiter = self.create_timer(timer_period_sec=0.1, callback=self.timer_callback)
        self._publishers = []
        self._map = None
        for i in range(ROBOT_COUNT):
            self._publishers.append(self.create_publisher(Twist, '/robot'+str(i)+'/cmd_vel', 5))

    def _set_initial_pose(self, num):
        print("Subscribed to topic: "+get_topic(num))
        self._pointSubscription = self.create_subscription(PointStamped, get_topic(num), self.pointCallback, 5)

    def _set_initial_yaw(self, num):
        print("Subscribed to topic: /robot" + str(num) + "/imu")
        self._imuSubscription = self.create_subscription(Imu, '/robot' + str(num) + '/imu', self.imuCallback, 5)

    # set poses for robots from odom topic
    def pointCallback(self, msg):
        position = np.zeros(2)
        position[0] = msg.point.x
        position[1] = msg.point.y
        self._robots[len(self._points)].offset[:2] = position
        print("Got position from robot" + str(len(self._points)))
        self._points.append(position)
        self.destroy_subscription(self._pointSubscription)
        if len(self._points) == ROBOT_COUNT:
            print("Subscribed to topic: /robot"+str(0)+"/imu")
            self._imuSubscription = self.create_subscription(Imu, '/robot' + str(0) + '/imu', self.imuCallback, 5)
        else:
            self._set_initial_pose(len(self._points))

    # set yaws for robots from imu topic
    def imuCallback(self, msg):
        _, _, yaw = R.from_quat([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w]).as_euler('XYZ')
        self._robots[len(self._yaws)].offset[2] = yaw
        print("Got yaw from robot" + str(len(self._yaws)))
        self._yaws.append(yaw)
        self.destroy_subscription(self._imuSubscription)
        if len(self._yaws) == ROBOT_COUNT:
            for i in range(ROBOT_COUNT):
                print(self._robots[i].offset)
            self._cost, self._map = self._match_destinations(range(ROBOT_COUNT), range(ROBOT_COUNT))
        else:
            self._set_initial_yaw(len(self._yaws))

    def _get_distance(self, r, d):
        return np.linalg.norm(self._robots[r].offset[:2] - self._destinations[d][:2])

    # recursively match destinations to robots by minimising least-squares distance
    def _match_destinations(self, robots, destinations):
        # we are happy with brute force here -> 5! = 120
        if len(robots) == len(destinations) == 1:
            return self._get_distance(robots[0], destinations[0]), {robots[0] : destinations[0]}
        else:
            cost = 9999
            best = {}
            for i in range(len(destinations)):
                c, m = self._match_destinations(robots[1:], [x for j,x in enumerate(destinations) if j!=i])
                m[robots[0]] = destinations[i]
                c += self._get_distance(robots[0], destinations[i])**2
                if c < cost:
                    cost = c
                    best = m
        return cost, best
    @staticmethod
    def _generate_destinations():
        def is_valid(pose, poses):
            for p1 in poses:
                if not np.linalg.norm(pose[:2] - p1[:2]) >= 2 * ROBOT_RADIUS:
                    return False
            return -wall_offset <= pose[X] <= wall_offset and -wall_offset <= pose[Y] <= wall_offset

        poses = []
        for _ in range(ROBOT_COUNT):
            pose = np.zeros(2, dtype=np.float32)
            wall_offset = DESTINATION_AREA_SIDE_OFFSET - ROBOT_RADIUS
            pose[X] = (np.random.rand() - 0.5) * 2. * wall_offset
            pose[Y] = (np.random.rand() - 0.5) * 2. * wall_offset
            while not is_valid(pose, poses):
                pose[X] = (np.random.rand() - 0.5) * 2. * wall_offset
                pose[Y] = (np.random.rand() - 0.5) * 2. * wall_offset
            poses.append(pose)
        return poses

    # compute the preferred velocity (linear towards goal)
    def _compute_preferred_velocity(self, robotIndex):
        v = self._destinations[self._map[robotIndex]] - self._robots[robotIndex].pose[:2]
        return MAX_SPEED * v / np.linalg.norm(v)

    # move a robot based on its velocity
    def _move_robot(self, robotIndex):
        u, w = feedback_linearized(self._robots[robotIndex].pose, self._robots[robotIndex].velocity, epsilon=EPSILON)
        vel_msg = Twist()
        vel_msg.linear.x = min(float(u), MAX_SPEED)
        vel_msg.angular.z = min(float(w), 1)
        print("Move robot: "+robotIndex)
        self._publishers[robotIndex].publish(vel_msg)

    # compute a discretised rvo grid
    def _compute_rvo_grid(self, robotIndex, otherPose, otherPreferred):
        d = int(2*DESTINATION_AREA_SIDE_OFFSET/RESOLUTION)
        grid = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if self._robots[robotIndex].is_in_rvo((np.array(get_grid_pos(i,j))-np.array(otherPose[:2])).tolist(), otherPose, otherPreferred):
                    grid[i][j] += 1
        return grid

    # given a discretised rvo map, choose the best velocity
    def _choose_vel(self, grid, desiredVel, pos):
        id, jd = pos_from_grid(desiredVel)
        d = int(2 * DESTINATION_AREA_SIDE_OFFSET / RESOLUTION)
        minDist = 9999
        iout = id
        jout = jd
        for i in range(d):
            for j in range(d):
                if grid[i][j] == 0:
                    if np.sqrt((i-id)**2+(j-jd)**2) < minDist:
                        minDist = np.sqrt((i-id)**2+(j-jd)**2)
                        iout = i
                        jout = j
        return (np.array(get_grid_pos(iout, jout)) - np.array(pos)).tolist()

    def timer_callback(self):
        # An implementation for sending periodic commands to the robot
        # could potentially go here (as well as other periodic methods).
        # NOTE: Goal assignment should [likely] not happen here.
        if self._map is not None:
            for i in range(ROBOT_COUNT):
                self._robots[i].velocity = self._compute_preferred_velocity(i)
            for i in range(ROBOT_COUNT):
                # create a grid for collisions that robot i may encounter
                d = int(2 * DESTINATION_AREA_SIDE_OFFSET / RESOLUTION)
                grid = np.zeros((d, d))
                for j in range(ROBOT_COUNT):
                    if i == j: continue
                    grid += self._compute_rvo_grid(i, self._robots[j].pose, self._robots[j].velocity)
                # given the grid, choose the best velocity
                self._robots[i].velocity = self._choose_vel(grid, self._robots[i].velocity, self._robots[i].pose[:2])
            # move all of the robots
            for i in range(ROBOT_COUNT):
                self._move_robot(i)
        return


def run(args):
    rclpy.init()
    multirobot_driver_node = MultirobotDriver(args)

    rclpy.spin(multirobot_driver_node)

    multirobot_driver_node.destroy_node()
    rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Runs multi-robot navigation')
    args, unknown = parser.parse_known_args()
    run(args)


if __name__ == '__main__':
    main()
