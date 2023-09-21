import rospy
import math
import numpy as np
import tf
from . import Bebop2StateViz
from RobotModel import pi_2_pi
from geometry_msgs.msg import Point
from RobotModel import ParticleFilter
from tf.transformations import euler_from_quaternion


class StateEstimator:
    def __init__(self) -> None:
        self.heading = np.pi / 2.0
        self.tf_listener = tf.TransformListener()
        self.timer = rospy.Timer(rospy.Duration(0.01), self.listen_transformation)

    def listen_transformation(self, event):
        z1 = self.get_transformation("camera_base_link", "tag2")
        z2 = self.get_transformation("camera_base_link", "tag7")
        z3 = self.get_transformation("camera_base_link", "tag4")

        Z = [z1, z2, z3]
              
        beta = 0.99
        for y in Z:
            if y is not None:
                mes = y[-1] + np.pi 
                self.heading = beta * self.heading  + (1 - beta) * mes


        z = np.zeros((0, 5))
        for i, zz in enumerate(Z):
            if zz is not None:
                # print(i, zz)
                d = math.hypot(zz[0], zz[1], zz[2])
                theta = np.arccos(zz[2] / d)
                # phi = pi_2_pi(math.atan2(zz[1], zz[0]) - zz[-1])
                phi = np.sign(zz[1]) * np.arccos(zz[0] / np.sqrt(zz[0]**2 + zz[1]**2))

                # zi = np.array([d, phi, theta, i, zz[-1] + np.pi ])
                zi = np.array([d, phi, theta, i, self.heading ])
                # zi = np.array([d, phi, theta, i, zz[3]])
                z = np.vstack((z, zi))

                # print(z)
        
        if len(z) > 0:
            # print(z)
            self.get_observation(z)

    def get_observation(self, z):

        raise NotImplementedError

        
    
    def get_transformation(self, parent_frame, child_frame):
        try:
            (raw_trans,rot) = self.tf_listener.lookupTransform(parent_frame, child_frame, rospy.Time(0))
            trans = [raw_trans[1], raw_trans[0], raw_trans[2]]

            euler = euler_from_quaternion(rot)
            
            rospy.loginfo(f"{parent_frame} -> {child_frame} :  {euler} ")
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
    
    def get_observation(self, z):


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

        rospy.loginfo(f"state = ({self.xEst[0, 0]:.3f}, {self.xEst[1, 0]:.3f}, {self.xEst[2, 0]:.3f}, {self.xEst[3, 0]:.3f} )")

        
        point_list = []
        for obs in z:
            landmarkID = int(obs[-2])
            detectedLandmark = Point()
            detectedLandmark.x = self.landmarks[landmarkID, 0]
            detectedLandmark.y = self.landmarks[landmarkID, 1]
            detectedLandmark.z = self.landmarks[landmarkID, 2]

            robotPosition = Point()
            robotPosition.x = self.xEst[0, 0]
            robotPosition.y = self.xEst[1, 0]
            robotPosition.z = self.xEst[2, 0]

            point_list.append(robotPosition)
            point_list.append(detectedLandmark)
            point_list.append(detectedLandmark)
            point_list.append(robotPosition)


            # alpha = math.atan2(self.landmarks[landmarkID, 1], self.landmarks[landmarkID, 0])
            # heading = pi_2_pi(alpha + obs[-1] + np.pi/2)
            # print(heading)

        self.viz.add_observation(point_list)
