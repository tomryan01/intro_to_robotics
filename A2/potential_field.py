from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pylab as plt
import numpy as np


WALL_OFFSET = 2.
CYLINDER_POSITION = np.array([0.5, 0.6], dtype=np.float32)
CYLINDER_RADIUS = .3
GOAL_POSITION = np.array([1.5, 1.5], dtype=np.float32)
START_POSITION = np.array([0., 0.], dtype=np.float32)
MAX_SPEED = .5
MAX_DIST = np.sqrt(2*(2*WALL_OFFSET))


# MISSING: Implementations for the following two cases (see handout).
# Solution:
CENTER_OBSTACLE = False
BONUS = False
CYLINDER_POSITION1 = np.array([0, 0.5], dtype=np.float32)
CYLINDER_POSITION2 = np.array([0.5, 0], dtype=np.float32)



def get_velocity_to_reach_goal(position, goal_position):
  v = np.zeros(2, dtype=np.float32)
  # MISSING: Compute the velocity field needed to reach goal_position
  # assuming that there are no obstacles.
  rpg = goal_position - position
  v = MAX_SPEED*(rpg/np.linalg.norm(rpg))/2.5
  v[1] -= 0.05
  return v

def get_velocity_to_avoid_obstacles(position, obstacle_positions, obstacle_radii):
  v = np.zeros(2, dtype=np.float32)
  # MISSING: Compute the velocity field needed to avoid the obstacles
  # In the worst case there might a large force pushing towards the
  # obstacles (consider what is the largest force resulting from the
  # get_velocity_to_reach_goal function). Make sure to not create
  # speeds that are larger than max_speed for each obstacle. Both obstacle_positions
  # and obstacle_radii are lists.
  a = 0.7
  for i in range(len(obstacle_positions)):
    rpo = obstacle_positions[i] - position
    b = (np.log(MAX_SPEED/2.5) - np.log(a)) / obstacle_radii[i]
    v -= normalize(rpo) * a * np.exp(b*np.linalg.norm(rpo))
  # if BONUS, encourage movement towards the center of the arena to avoid walls
  if BONUS:
    v -= position*0.1

  return v

def normalize(v):
  n = np.linalg.norm(v)
  if n < 1e-2:
    return np.zeros_like(v)
  return v / n


def cap(v, max_speed):
  n = np.linalg.norm(v)
  if n > max_speed:
    return v / n * max_speed
  return v


def get_velocity(position, mode='all'):
  if mode in ('goal', 'all'):
    v_goal = get_velocity_to_reach_goal(position, GOAL_POSITION)
  else:
    v_goal = np.zeros(2, dtype=np.float32)
  if mode in ('obstacle', 'all'):
    v_avoid = get_velocity_to_avoid_obstacles(
      position,
      [CYLINDER_POSITION1, CYLINDER_POSITION2] if BONUS else [CYLINDER_POSITION],
      ([CYLINDER_RADIUS] * 2) if BONUS else [CYLINDER_RADIUS])
  else:
    v_avoid = np.zeros(2, dtype=np.float32)
  v = v_goal + v_avoid
  # Solution (BONUS):
  
  return cap(v, max_speed=MAX_SPEED)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Runs obstacle avoidance with a potential field')
  parser.add_argument('--mode', action='store', default='all', help='Which velocity field to plot.', choices=['obstacle', 'goal', 'all'])
  args, unknown = parser.parse_known_args()

  fig, ax = plt.subplots()
  # Plot field.
  X, Y = np.meshgrid(np.linspace(-WALL_OFFSET, WALL_OFFSET, 30),
                     np.linspace(-WALL_OFFSET, WALL_OFFSET, 30))
  U = np.zeros_like(X)
  V = np.zeros_like(X)
  for i in range(len(X)):
    for j in range(len(X[0])):
      velocity = get_velocity(np.array([X[i, j], Y[i, j]]), args.mode)
      U[i, j] = velocity[0]
      V[i, j] = velocity[1]
  plt.quiver(X, Y, U, V, units='width')

  # Plot environment.
  if BONUS:
    ax.add_artist(plt.Circle(CYLINDER_POSITION1, CYLINDER_RADIUS, color='gray'))
    ax.add_artist(plt.Circle(CYLINDER_POSITION2, CYLINDER_RADIUS, color='gray'))
  else:
    ax.add_artist(plt.Circle(CYLINDER_POSITION, CYLINDER_RADIUS, color='gray'))
  plt.plot([-WALL_OFFSET, WALL_OFFSET], [-WALL_OFFSET, -WALL_OFFSET], 'k')
  plt.plot([-WALL_OFFSET, WALL_OFFSET], [WALL_OFFSET, WALL_OFFSET], 'k')
  plt.plot([-WALL_OFFSET, -WALL_OFFSET], [-WALL_OFFSET, WALL_OFFSET], 'k')
  plt.plot([WALL_OFFSET, WALL_OFFSET], [-WALL_OFFSET, WALL_OFFSET], 'k')

  # Plot a simple trajectory from the start position.
  # Uses Euler integration.
  dt = 0.01
  x = START_POSITION
  positions = [x]
  for t in np.arange(0., 20., dt):
    v = get_velocity(x, args.mode)
    x = x + v * dt
    positions.append(x)
  positions = np.array(positions)
  plt.plot(positions[:, 0], positions[:, 1], lw=2, c='r')

  plt.axis('equal')
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim([-.5 - WALL_OFFSET, WALL_OFFSET + .5])
  plt.ylim([-.5 - WALL_OFFSET, WALL_OFFSET + .5])
  plt.show()
