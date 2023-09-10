import matplotlib.pyplot as plt
import numpy as np
import math
from RobotModel import observation_model, get_noisy_reading
from RobotModel import pf_localization
from scipy.spatial.transform import Rotation as Rot


def rot_mat_2d(angle):
    """
    Create 2D rotation matrix from an angle

    Parameters
    ----------
    angle :

    Returns
    -------
    A 2D rotation matrix

    Examples
    --------
    >>> angle_mod(-4.0)


    """
    return Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]

NP = 1000  # Number of Particle
DT = 0.1  # time tick [s]
STATE_DIM = 8 # state variable dimension
CTRL_DIM = 4 # control variable dimension
NUM_MAX_THREADS = 6
L = 6 # length of workspace [m]

from scipy.spatial.transform import Rotation as Rot


def rot_mat_2d(angle):
    """
    Create 2D rotation matrix from an angle

    Parameters
    ----------
    angle :

    Returns
    -------
    A 2D rotation matrix

    Examples
    --------
    >>> angle_mod(-4.0)


    """
    return Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]

def calc_input():
    SIM_TIME = 100  # total simulation time
    N = SIM_TIME // 4  # each arm traversing time
    v = L / (N * DT)
    u_f = np.array([[0, v, 0, 0] for _ in range(N)])
    u_r = np.array([[v, 0, 0, 0] for _ in range(N)])
    u_b = np.array([[0, -v, 0, 0] for _ in range(N)])
    u_l = np.array([[-v, 0, 0, 0] for _ in range(N)])

    U = np.vstack((u_f, u_r, u_b, u_l)).reshape((SIM_TIME, CTRL_DIM, 1))

    for u in U:
        yield u


def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
    Pxy = PEst[0:2, 0:2]
    eig_val, eig_vec = np.linalg.eig(Pxy)

    if eig_val[0] >= eig_val[1]:
        big_ind = 0
        small_ind = 1
    else:
        big_ind = 1
        small_ind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)

    # eig_val[big_ind] or eiq_val[small_ind] were occasionally negative
    # numbers extremely close to 0 (~10^-20), catch these cases and set
    # the respective variable to 0
    try:
        a = math.sqrt(eig_val[big_ind])
    except ValueError:
        a = 0

    try:
        b = math.sqrt(eig_val[small_ind])
    except ValueError:
        b = 0

    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eig_vec[1, big_ind], eig_vec[0, big_ind])
    fx = np.stack([x, y]).T @ rot_mat_2d(angle)

    px = np.array(fx[:, 0] + xEst[0, 0]).flatten()
    py = np.array(fx[:, 1] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--g")

def main():
    print(__file__ + " start!!")

    time = 0.0

    # RF_ID positions [x, y]
    tag_id = np.array([[0.88, 7.0, 1.2],
                      [2.03, 7.0, 1.2],
                      [3.18, 7.0, 1.2],
                      [4.33, 7.0, 1.2]])
    tag_id = np.expand_dims(tag_id, axis=2)
    print(tag_id.shape)
    # State Vector [x y yaw v]'
    x_true = np.zeros((STATE_DIM, 1))
    x_true[3, 0] = np.pi / 2.0 # init heading 90

    px = np.zeros((STATE_DIM, NP)) # Particle store
    pw = np.zeros((1, NP)) + 1.0 / NP  # Particle weight

    FOV = np.rad2deg(120)
    print(tag_id.shape)

    for u in calc_input():
        x_true, Z = observation_model(x_true, u, tag_id, DT, FOV)
        x_est, p_est, px, pw = pf_localization(px, pw, Z, u, DT, NP, NUM_MAX_THREADS)

        plt.cla()
        for i, z in enumerate(Z):

            x_noise = get_noisy_reading(x_true, z[:3])
            Xn = [x_true[0, 0], x_noise[0]]
            Yn = [x_true[1, 0], x_noise[1]]
            plt.plot(Xn, Yn, '--k')

        plt.plot(px[0, :], px[1, :], ".r", alpha=0.2)
        plt.scatter(tag_id[:, 0], tag_id[:, 1])
        plt.scatter(x_true[0, 0], x_true[1, 0])
        plt.scatter(x_est[0, 0], x_est[1, 0])
        plot_covariance_ellipse(x_est, p_est)

        plt.axis([-2, L + 2, -2, L + 2])
        plt.pause(DT)


if __name__ == '__main__':
    main()