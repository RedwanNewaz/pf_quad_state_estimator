import matplotlib.pyplot as plt
import numpy as np
from RobotModel import get_noisy_reading

from Simulator import SimStateEstimator
from test.eight import calc_input

NP = 1000  # Number of Particle
DT = 0.1  # time tick [s]
STATE_DIM = 8 # state variable dimension
CTRL_DIM = 4 # control variable dimension


def main():
    print(__file__ + " start!!")

    # TAG_ID positions [x, y, z]
    tag_id = np.array([[0.88, 7.0, 1.2],
                      [2.03, 7.0, 1.2],
                      [3.18, 7.0, 1.2],
                      [4.33, 7.0, 1.2]])
    FOV = np.rad2deg(120)

    sim = SimStateEstimator(tag_id, FOV)

    traj = np.zeros((0, 2))
    for u in calc_input():
        Z = sim(u)

        plt.cla()
        for i, z in enumerate(Z):
            x_noise = get_noisy_reading(sim.x_true, z[:3])
            Xn = [sim.x_est[0, 0], x_noise[0]]
            Yn = [sim.x_est[1, 0], x_noise[1]]
            plt.plot(Xn, Yn, '--k')

        # show trajectory
        traj_i = np.array([sim.x_est[0, 0], sim.x_est[1, 0]])
        traj = np.vstack((traj, traj_i))
        plt.plot(traj[:, 0], traj[:, 1], ".r", alpha=0.2)


        # plt.plot(px[0, :], px[1, :], ".r", alpha=0.2)
        plt.scatter(tag_id[:, 0], tag_id[:, 1])
        plt.scatter(sim.x_true[0, 0], sim.x_true[1, 0])
        plt.scatter(sim.x_est[0, 0], sim.x_est[1, 0])

        # plot_covariance_ellipse
        px, py = sim.get_cov_ellipse()
        plt.plot(px, py, "--g")

        plt.axis([-2, 8, -2, 8])
        plt.pause(DT)


if __name__ == '__main__':
    main()