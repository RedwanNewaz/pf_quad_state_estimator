#!/usr/bin/env python  

from scipy.spatial.transform import Rotation as R
import rospy
import math
import numpy as np
import tf
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Point
from tf_rviz import Bebop2StateViz
from ObservationModel import estimate_zx


def pi_2_pi(angle):

    return (angle + math.pi) % (2 * math.pi) - math.pi
class StateEstimator:
    def __init__(self) -> None:
        self.heading = np.pi / 2.0
        self.tf_listener = tf.TransformListener()
        self.timer = rospy.Timer(rospy.Duration(0.01), self.listen_transformation)

    def listen_transformation(self, event):
        z0 = self.get_transformation("camera_base_link", "tag3")
        z1 = self.get_transformation("camera_base_link", "tag2")
        z2 = self.get_transformation("camera_base_link", "tag7")
        z3 = self.get_transformation("camera_base_link", "tag4")

        Z = [z0, z1, z2, z3]

        result = []
        for i, zz in enumerate(Z):
            if zz is not None:
                zz.extend([i])
                result.append(zz)
        self.get_observation(result)


              
        # beta = 0.99
        # for y in Z:
        #     if y is not None:
        #         mes = y[-1]
        #         self.heading = beta * self.heading  + (1 - beta) * mes
        #
        #
        # z = np.zeros((0, 5))
        # for i, zz in enumerate(Z):
        #     if zz is not None:
        #         # print(i, zz)
        #         d = math.hypot(zz[0], zz[1], zz[2])
        #         theta = np.arccos(zz[2] / d)
        #         # phi = pi_2_pi(math.atan2(zz[1], zz[0]) - zz[-1])
        #         phi = np.sign(zz[1]) * np.arccos(zz[0] / np.sqrt(zz[0]**2 + zz[1]**2))
        #
        #         # zi = np.array([d, phi, theta, i, zz[-1] + np.pi ])
        #         zi = np.array([d, phi, theta, i, self.heading ])
        #         # zi = np.array([d, phi, theta, i, zz[3]])
        #         z = np.vstack((z, zi))
        #
        #         # print(z)
        #
        # if len(z) > 0:
        #     # print(z)
        #     self.get_observation(z)

        

    def get_observation(self, z):
        raise NotImplementedError

        
    
    def get_transformation(self, parent_frame, child_frame):
        try:
            (trans,rot) = self.tf_listener.lookupTransform(parent_frame, child_frame, rospy.Time(0))
            # trans = [raw_trans[1], raw_trans[0], raw_trans[2]]

            euler = euler_from_quaternion(rot)
            
            # rospy.loginfo(f"{parent_frame} -> {child_frame} :  {euler} ")
            trans.extend(list(euler))
            return trans 
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            # rospy.loginfo("transformation does not exit")
            pass


class Bebop2StateEstimator(StateEstimator):
    def __init__(self, landmarks, state_dim, num_particles, dt) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.np = num_particles 
        self.dt = dt 
        
        self.pf = None
        self.u = None
        self.xEst = np.zeros((state_dim, 1))
        self.xEst[3, 0] = np.pi / 2 
        self.xEst[2, 0] = 1.0 

        self.landmarks = np.squeeze(landmarks)
        self.viz = Bebop2StateViz(self.landmarks)
        print(self.landmarks.shape)

    def map_to_robot(self, z):
        points = []
        for obs in z:
            # x, y, z, r, p, y, id
            landmarkID = int(obs[-1])
            x0 = self.landmarks[landmarkID, 0]
            y0 = self.landmarks[landmarkID, 1]
            z0 = self.landmarks[landmarkID, 2]

            p_angle = math.atan2(obs[1] , obs[0])
            h_angle = -obs[-2]
            alpha = math.atan2(y0, x0)


            theta =  pi_2_pi(math.pi - p_angle - alpha)
            # theta = math.pi
            # print(f"{np.rad2deg(theta):.3f}, {landmarkID}")


            xx = x0 + (obs[0] * math.cos(theta) - obs[1] * math.sin(theta))
            yy = y0 - (obs[0] * math.sin(theta) + obs[1] * math.cos(theta))
            zz = z0 - obs[2]
            points.append([xx, yy, zz, h_angle])
        return points

    def get_observation(self, z):

        point_list = []
        z_points = self.map_to_robot(z.copy())
        # z_points = estimate_zx(z, self.landmarks, None)
        if len(z_points):
            val = np.array(z_points).mean(axis=0)
            self.viz(val)

            rospy.loginfo(f" z_points {val}")
        #
        # z_points = z_points.tolist()
        for i, obs in enumerate(z):
            landmarkID = int(obs[-1])
            detectedLandmark = Point()
            # detectedLandmark.x = self.landmarks[landmarkID, 0]
            # detectedLandmark.y = self.landmarks[landmarkID, 1]
            # detectedLandmark.z = self.landmarks[landmarkID, 2]

            robotPosition = Point()
            robotPosition.x = z_points[i][0]
            robotPosition.y = z_points[i][1]
            robotPosition.z = z_points[i][2]


            point_list.append(robotPosition)
            point_list.append(detectedLandmark)
            point_list.append(detectedLandmark)
            point_list.append(robotPosition)

        self.viz.add_observation(point_list)
