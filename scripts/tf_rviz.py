from typing import Any
import rospy 
import math
from copy import deepcopy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class Bebop2StateViz:
    def __init__(self, landmarks) -> None:
        self.landmarks = landmarks
        self.timer = rospy.Timer(rospy.Duration(0.01), self.publish_state)
        self.marker_pub = rospy.Publisher('/bebop2', Marker, queue_size=1)
        self.position = [0, 0, 0]
        self.orientation = [0, 0, 0, 1]


    def __call__(self, state):
        # print(state, "second")
        self.position = state[:3]
        self.orientation = quaternion_from_euler(0, 0, state[3] + math.pi / 2)

    def publish_state(self, event):
        bebop_marker = self.populate_marker()
        bebop_marker.pose.orientation.x = self.orientation[0]  # Set the orientation of the marker
        bebop_marker.pose.orientation.y = self.orientation[1]
        bebop_marker.pose.orientation.z = self.orientation[2]
        bebop_marker.pose.orientation.w = self.orientation[3]
        self.marker_pub.publish(bebop_marker)

        landmark_marker = self.populate_marker()

        landmark_marker.type = Marker.CUBE_LIST
        landmark_marker.id = 1

        for landmark in self.landmarks:
            point = Point()
            point.x = landmark[0]
            point.y = landmark[1]
            point.z = landmark[2]
            landmark_marker.points.append(point)

        landmark_marker.color.r = 0

        landmark_marker.pose.position.x = 0  # Set the position of the marker
        landmark_marker.pose.position.y = 0
        landmark_marker.pose.position.z = 0

        landmark_marker.scale.x = 0.2
        landmark_marker.scale.y = 0.2
        landmark_marker.scale.z = 0.2
        
        self.marker_pub.publish(landmark_marker)

    def add_observation(self, point_list):
        #print("Points:", points)
        _marker = self.populate_marker()
        line_marker = deepcopy(_marker)
      
        line_marker.type = Marker.LINE_STRIP
        line_marker.action=Marker.ADD
        line_marker.id = 30
        
        for point in point_list:
            line_marker.points.append(point)
        
        line_marker.pose.position.x = line_marker.pose.position.y= line_marker.pose.position.z = 0.0
        
        line_marker.scale.x = line_marker.scale.y = line_marker.scale.z = 0.05
        line_marker.color.g = 0
       

        self.marker_pub.publish(line_marker)




    def populate_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"  # Set the frame ID as needed
        marker.header.stamp = rospy.Time.now()
        marker.ns = "bebop2"
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.pose.position.x = self.position[0]  # Set the position of the marker
        marker.pose.position.y = self.position[1]
        marker.pose.position.z = self.position[2]
        marker.pose.orientation.x = 0.0  # Set the orientation of the marker
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.001
        marker.scale.y = 0.001
        marker.scale.z = 0.001

        marker.color.r = 0.66
        marker.color.g = 0.66
        marker.color.b = 0.66
        marker.color.a = 1.0

        # Specify the mesh file path (replace with your own mesh file)
        marker.mesh_resource = "package://bebop_description/meshes/bebop_model.stl"
        return marker