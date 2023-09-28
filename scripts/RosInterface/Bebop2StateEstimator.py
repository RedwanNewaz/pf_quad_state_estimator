#!/usr/bin/env python  
import rospy
import math
import numpy as np
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from . import Bebop2StateViz
from RobotModel import ParticleFilter

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

class StateEstimator:
    def __init__(self) -> None:
        self.tf_listener = tf.TransformListener()
        self.timer = rospy.Timer(rospy.Duration(0.03), self.listen_transformation)

    def listen_transformation(self, event):
        z0 = self.get_transformation("camera_base_link", "tag3")
        z1 = self.get_transformation("camera_base_link", "tag2")
        z2 = self.get_transformation("camera_base_link", "tag7")
        z3 = self.get_transformation("camera_base_link", "tag4")

        Z = [z0, z1, z2, z3]

        result = []
        for i, zz in enumerate(Z):
            if zz is not None:
                # add landmark index at the end 
                zz.extend([i])
                result.append(zz)
        self.get_observation(result)

     

    def get_observation(self, z):
        raise NotImplementedError

        
    
    def get_transformation(self, parent_frame, child_frame):
        try:
            (trans,rot) = self.tf_listener.lookupTransform(parent_frame, child_frame, rospy.Time(0))
            euler = euler_from_quaternion(rot)
            trans.extend(list(euler))
            return trans 
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
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

        self.h_angle = None 
        self.coord = []

        self.state_pub = rospy.Publisher('pf/state', Odometry, queue_size=1)

    def rotation_matrix_yaw(self, yaw):
        # Create a 3x3 identity matrix
        R = np.eye(3)

        # Compute sine and cosine of the yaw angle
        c = np.cos(yaw)
        s = np.sin(yaw)

        # Fill in the rotation matrix elements
        R[0, 0] = c
        R[0, 1] = -s
        R[1, 0] = s
        R[1, 1] = c

        return R

    def get_observation(self, z):

        if not len(z):
            return []
        
        _angle = np.array([-obs[-2] for obs in z]).mean()
        # _angle = []
        # for obs in z:
        #     y = -obs[-2] 
        #     landmarkID = int(obs[-2])
        #     x0 = self.landmarks[landmarkID, 0]
        #     y0 = self.landmarks[landmarkID, 1]
        #     z0 = self.landmarks[landmarkID, 2]

        #     alpha = math.atan2(y0, x0)
        #     theta = pi_2_pi(math.pi/2 + y - alpha)
        #     _angle.append(theta)
        
        # _angle = np.array(_angle).mean()


        if np.isnan(_angle):
            return []
        
        alpha = 0.95
        if not self.h_angle:
           self.h_angle = _angle  
        else: 
            self.h_angle = self.h_angle * alpha + (1 - alpha) * _angle

        # print(f"h_angle = {self.h_angle:.3f} | _angle = {_angle}")

        z_meas = np.zeros((0, 5))
        for obs in z:
            landmarkID = int(obs[-2])
            x0 = self.landmarks[landmarkID, 0]
            y0 = self.landmarks[landmarkID, 1]
            z0 = self.landmarks[landmarkID, 2]

            # alpha = math.atan2(y0, x0)
            d1 = math.hypot(x0, y0, z0)
            alpha = np.arccos(z0 / d1)

            # phi2 = np.sign(y0) * np.arccos(x0 / np.sqrt(x0**2 + y0**2))
            theta2 = pi_2_pi(math.pi/2 + self.h_angle - alpha)

            rotation_matrix = self.rotation_matrix_yaw(theta2)
            translation = np.array([obs[0], obs[1], obs[2]])
            zz = np.dot(rotation_matrix, translation) 

            # convert it to spherical coordinate 
            d = math.hypot(zz[0], zz[1], zz[2])
            theta = np.arccos(zz[2] / d)

            phi = np.sign(zz[1]) * np.arccos(zz[0] / np.sqrt(zz[0]**2 + zz[1]**2))
            # phi = pi_2_pi(math.pi/2 + phi - phi2)

            landmarkID = int(obs[-1])
            
            zi = np.array([d, phi, theta, landmarkID, self.h_angle ])
            z_meas = np.vstack((z_meas, zi))
            
            
        N = len(z_meas)
        if N > 0:
            # print(len(z))
            self.map_observation(z_meas)



       
        # points = []
        # for obs in z:
        #     # x, y, z, r, p, y, id
        #     landmarkID = int(obs[-1])
        #     x0 = self.landmarks[landmarkID, 0]
        #     y0 = self.landmarks[landmarkID, 1]
        #     z0 = self.landmarks[landmarkID, 2]

        #     rotation_matrix = self.rotation_matrix_yaw(self.h_angle)
        #     translation = np.array([obs[0], obs[1], obs[2]])

        #     initial_point = np.array([x0, y0, z0])
        #     transformed_point = initial_point - np.dot(rotation_matrix, translation)  
        #     xx, yy, zz = transformed_point
        #     self.coord = [xx, yy, zz, self.h_angle]  
        #     points.append(self.coord)


           

        # return points

    # def viz_estimated_landmarks(self, z):
    #     robot_landmark_coord = map(convert_spherical_to_cartesian, z)
    #     point_list = []
    #     for x in robot_landmark_coord:
    #         x0 = np.squeeze(self.xEst)[:3]
    #         l0 = x0 + x[:3]
    #         # print(l0)
    #         detectedLandmark = Point()
    #         detectedLandmark.x = l0[0]
    #         detectedLandmark.y = l0[1]
    #         detectedLandmark.z = l0[2]

    #         robotPosition = Point()
    #         robotPosition.x = x0[0]
    #         robotPosition.y = x0[1]
    #         robotPosition.z = x0[2]

    #         point_list.append(robotPosition)
    #         point_list.append(detectedLandmark)
    #         point_list.append(detectedLandmark)
    #         point_list.append(robotPosition)

    #     self.viz.add_observation(point_list)

    def map_observation(self, z):

        # print(z.shape)

        u = np.zeros((4, 1))
        if self.pf is None:
            tag_id = np.expand_dims(self.landmarks, axis=2)
            self.pf = ParticleFilter(tag_id, z, self.state_dim, self.np, self.dt)
            self.pf(z, u)

        cntrl = self.pf.estimate_control(z)
        if cntrl is not None:
            u = cntrl
        self.pf(z, u)
        self.xEst = self.pf.getState()
        self.viz(np.squeeze(self.xEst))

        self.viz.estimated_landmarks(z, self.xEst)

        pEst = self.pf.getCovariance()
        self.viz.plot_covariance_ellipse(self.xEst, pEst)
        

        odom = Odometry()
        odom.header.frame_id = "map";
        odom.child_frame_id = "camera_base_link";
        odom.pose.pose.position.x = self.xEst[0, 0]
        odom.pose.pose.position.y = self.xEst[1, 0]
        odom.pose.pose.position.z = self.xEst[2, 0]

        rotation = quaternion_from_euler(0,0,self.xEst[3, 0])
        odom.pose.pose.orientation.x = rotation[0]
        odom.pose.pose.orientation.y = rotation[1]
        odom.pose.pose.orientation.z = rotation[2]
        odom.pose.pose.orientation.w = rotation[3]


        cov = np.zeros((6, 6), dtype=np.float64)
        cov[:3, :3] = pEst[:3, :3]
        cov[5, 5] = pEst[2, 2]
        cov[5, :3] = pEst[2, :3]
        odom.pose.covariance = cov.flatten().tolist()

        self.state_pub.publish(odom)


        


        
