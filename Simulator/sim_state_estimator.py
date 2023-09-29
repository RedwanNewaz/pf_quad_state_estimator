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
    return px, py

class SimStateEstimator:
    def __init__(self, tag_id, FOV):
        self.tag_id = np.expand_dims(tag_id, axis=2)
        self.FOV = FOV

        self.x_true = self.x_est = np.zeros((STATE_DIM, 1))
        self.x_true[3, 0] = self.x_est[3, 0] = np.pi / 2.0  # init heading 90

        # initialize particle filter

        self.x_true, Z, Z_true = observation_model(self.x_true, np.zeros((4, 1)), self.tag_id, DT, FOV)
        self.pf = ParticleFilter(self.tag_id, Z, STATE_DIM, NP, DT)

    def __call__(self, u):
        self.x_true, Z, Z_true = observation_model(self.x_true, u, self.tag_id, DT, self.FOV)

        cntrl = self.pf.estimate_control(Z)
        if cntrl is not None:
            u = cntrl
        self.pf(Z, u)
        self.x_est = self.pf.getState()
        self.p_est = self.pf.getCovariance()

        return Z

    def get_cov_ellipse(self):
        px, py = plot_covariance_ellipse(self.x_est, self.p_est)
        return px, py