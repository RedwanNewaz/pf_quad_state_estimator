#!/usr/bin/env python  
import rospy
import numpy as np 
import math 
from RobotModel import ParticleFilter
from RobotModel import observation_model
from RosInterface import Bebop2StateViz
from RosInterface import Bebop2StateEstimator
from geometry_msgs.msg import Point

NP = 1000  # Number of Particle
DT = 0.035  # time tick [s]
STATE_DIM = 8 # state variable dimension
CTRL_DIM = 4 # control variable dimension

L = 6 # length of workspace [m]

# TAG_ID positions [x, y, z]
tag_id = np.array([[1.96, 7.0, 1.2],
                    [3.11, 7.0, 1.2],
                    [4.26, 7.0, 1.2]])
tag_id = np.expand_dims(tag_id, axis=2)

def calc_input():
    SIM_TIME = 400   # total simulation time
    N = SIM_TIME // 4  # each arm traversing time
    v = L / (N * DT)
    u_f = np.array([[0, v, 0, 0] for _ in range(N)])
    u_r = np.array([[v, 0, 0, 0] for _ in range(N)])
    u_b = np.array([[0, -v, 0, 0] for _ in range(N)])
    u_l = np.array([[-v, 0, 0, 0] for _ in range(N)])

    U = np.vstack((u_f, u_r, u_b, u_l)).reshape((SIM_TIME, CTRL_DIM, 1))

    for u in U:
        yield u

def main():
    x_true = np.zeros((STATE_DIM, 1))
    x_true[3, 0] = np.pi / 2.0 # init heading 90

    FOV = np.rad2deg(120)
    x_true, Z, Z_true = observation_model(x_true, np.zeros((4, 1)), tag_id, DT, FOV)
    pf = ParticleFilter(tag_id, Z, STATE_DIM, NP, DT)

    rate = rospy.Rate(100)  # 1 Hz
    viz = Bebop2StateViz(np.squeeze(tag_id))
    for u in calc_input():
        x_true, Z, Z_true = observation_model(x_true, u, tag_id, DT, FOV)

        if(rospy.is_shutdown()):
            break

        cntrl = pf.estimate_control(Z)
        if cntrl is not None:
            u = cntrl
        pf(Z, u)
        x_est = pf.getState()
        p_est = pf.getCovariance()
        # print(x_est.T)
        viz(np.squeeze(x_est))
        rate.sleep()



if __name__ == '__main__':
    rospy.init_node('bebop2_state_estimator')
    main()

    



    


  