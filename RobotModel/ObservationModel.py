import numpy as np
import math
from . import motion_model

#  Simulation parameter
Q_sim = np.diag([0.2, np.deg2rad(1.0)]) ** 2


def fov_triangle(x_true, fov):
    '''
        left--------------------Right
        \                       /
            \               /
                \       /
                    \
                    robot
    :param x_true: robot state
    :param fov: field of view
    :return: triangle points
    '''


    dq = fov / 2.0
    theta = x_true[3, 0]
    R = 22 # replace it with any big number

    x_left = x_true[0, 0] + R * math.cos(theta - dq)
    y_left = x_true[1, 0] + R * math.sin(theta - dq)
    left = np.array([x_left, y_left])

    x_right = x_true[0, 0] + R * math.cos(theta + dq)
    y_right = x_true[1, 0] + R * math.sin(theta + dq)
    right = np.array([x_right, y_right])

    center = np.array([x_true[0, 0], x_true[1, 0]])

    return np.array([center, left, right])


def is_point_inside_triangle(point, triangle):

    v1, v2, v3 = triangle

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(point, v1, v2)
    d2 = sign(point, v2, v3)
    d3 = sign(point, v3, v1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def pi_2_pi(angle):

    return (angle + math.pi) % (2 * math.pi) - math.pi

def observation_model(x_true, u, landmarks, dt, fov):
    '''
    :param x_true: true state
    :param u: control
    :param landmarks:
    :param dt: sample time
    :return: z
    '''
    x_true = motion_model(x_true, u, dt)
    triangle = fov_triangle(x_true, fov)

    # add noise to sensor r-theta
    z = np.zeros((0, 4))

    for i, landmark in enumerate(landmarks):

        if not is_point_inside_triangle(landmark, triangle):
            continue

        dx = x_true[0, 0] - landmark[0, 0]
        dy = x_true[1, 0] - landmark[1, 0]

        d = math.hypot(dx, dy)
        q = math.atan2(dy, dx)
        angle = pi_2_pi(q - x_true[3, 0])

        dn = d + np.random.randn() * Q_sim[0, 0] ** 0.5
        angle_with_noise = angle + np.random.randn() * Q_sim[1, 1] ** 0.5
        angle_with_noise = pi_2_pi(angle_with_noise)

        zi = np.array([dn, angle_with_noise, landmark[0, 0], landmark[1, 0]])
        z = np.vstack((z, zi))

    return x_true, z



def get_noisy_reading(x, zd):
    r, q = zd
    theta = pi_2_pi(q - x[3, 0])
    xx = x[0, 0] + r * math.cos(theta)
    yy = x[1, 0] + r * math.sin(theta)
    return np.array([xx, yy])

def observe_landmark_position(x, landmarks):
    landmarks_pos = np.zeros((2 * landmarks.shape[0], 1))
    for (i, lm) in enumerate(landmarks):
        index = 2 * i
        q = Q_sim[0, 0] ** 0.5
        landmarks_pos[index] = x[0, 0] + lm[0] * math.cos(
            x[2, 0] + lm[1]) + np.random.randn() * q / np.sqrt(2)
        landmarks_pos[index + 1] = x[1, 0] + lm[0] * math.sin(
            x[2, 0] + lm[1]) + np.random.randn() * q / np.sqrt(2)
    return landmarks_pos




