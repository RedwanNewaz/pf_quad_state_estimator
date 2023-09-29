import matplotlib.pyplot as plt
import numpy as np
import math
from RobotModel import observation_model, get_noisy_reading
from RobotModel import ParticleFilter
from scipy.spatial.transform import Rotation as Rot
from test.eight import calc_input

NP = 1000  # Number of Particle
DT = 0.1  # time tick [s]
STATE_DIM = 8 # state variable dimension
CTRL_DIM = 4 # control variable dimension




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

    # TAG_ID positions [x, y, z]
    tag_id = np.array([[0.88, 7.0, 1.2],
                      [2.03, 7.0, 1.2],
                      [3.18, 7.0, 1.2],
                      [4.33, 7.0, 1.2]])
    tag_id = np.expand_dims(tag_id, axis=2)
    print(tag_id.shape)
    # State Vector [x y yaw v]'
    x_true = np.zeros((STATE_DIM, 1))
    x_true[3, 0] = np.pi / 2.0 # init heading 90

    FOV = np.rad2deg(120)
    x_true, Z, Z_true = observation_model(x_true, np.zeros((4, 1)), tag_id, DT, FOV)
    pf = ParticleFilter(tag_id, Z, STATE_DIM, NP, DT)


    for u in calc_input():
        x_true, Z, Z_true = observation_model(x_true, u, tag_id, DT, FOV)

        cntrl = pf.estimate_control(Z)
        if cntrl is not None:
            u = cntrl
        pf(Z, u)
        x_est = pf.getState()
        p_est = pf.getCovariance()


        plt.cla()
        for i, z in enumerate(Z):
            x_noise = get_noisy_reading(x_true, z[:3])
            Xn = [x_est[0, 0], x_noise[0]]
            Yn = [x_est[1, 0], x_noise[1]]
            plt.plot(Xn, Yn, '--k')


        # plt.plot(px[0, :], px[1, :], ".r", alpha=0.2)
        plt.scatter(tag_id[:, 0], tag_id[:, 1])
        plt.scatter(x_true[0, 0], x_true[1, 0])
        plt.scatter(x_est[0, 0], x_est[1, 0])
        plot_covariance_ellipse(x_est, p_est)

        plt.axis([-2, 6 + 2, -2, 6 + 2])
        plt.pause(DT)


if __name__ == '__main__':
    main()