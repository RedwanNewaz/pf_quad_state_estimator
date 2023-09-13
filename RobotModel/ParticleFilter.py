import math
import numpy as np
from . import generate_noisy_control, motion_model
from . import get_noisy_reading, pi_2_pi
from . import estimate_zx
from .MotionModel import R_sim
from concurrent.futures import ThreadPoolExecutor
Q = np.diag([0.02, np.deg2rad(5.0)]) ** 2  # range error
np.random.seed(1234)

def gauss_likelihood(x, sigma):
    p = 1.0 / math.sqrt(2.0 * math.pi * sigma ** 2) * \
        math.exp(-x ** 2 / (2 * sigma ** 2))

    return p


def calc_covariance(x_est, px, pw):
    """
    calculate covariance matrix
    see ipynb doc
    """
    cov = np.zeros((3, 3))
    n_particle = px.shape[1]
    for i in range(n_particle):
        dx = (px[:, i:i + 1] - x_est)[0:3]
        cov += pw[0, i] * dx @ dx.T
    cov *= 1.0 / (1.0 - pw @ pw.T)

    return cov

def re_sampling(px, pw, NP):
    """
    low variance re-sampling
    """

    w_cum = np.cumsum(pw)
    base = np.arange(0.0, 1.0, 1 / NP)
    re_sample_id = base + np.random.uniform(0, 1 / NP)
    indexes = []
    ind = 0
    for ip in range(NP):
        while re_sample_id[ip] > w_cum[ind]:
            ind += 1
        indexes.append(ind)

    px = px[:, indexes]
    pw = np.zeros((1, NP)) + 1.0 / NP  # init weight

    return px, pw


# def pf_localization(px, pw, z, u, dt, NP):
#     """
#     Localization with Particle filter
#     """
#     NTh = NP / 2.0  # Number of particle for re-sampling
#     for ip in range(NP):
#         x = np.array([px[:, ip]]).T
#         w = pw[0, ip]
#
#         #  Predict with random input sampling
#         ud = generate_noisy_control(u)
#         x = motion_model(x, ud, dt)
#
#         #  Calc Importance Weight
#         for i, y in enumerate(z):
#             x_noise = get_noisy_reading(x, y[:2])
#             dx = x[0, 0] - x_noise[0]
#             dy = x[1, 0] - x_noise[1]
#             pre_z = math.hypot(dx, dy)
#             dz = pre_z - y[0]
#             w = w * gauss_likelihood(dz, math.sqrt(Q[0, 0]))
#
#         px[:, ip] = x[:, 0]
#         pw[0, ip] = w
#
#     # print(pw.sum())
#     pw = pw / pw.sum()  # normalize
#
#
#     x_est = px.dot(pw.T)
#     p_est = calc_covariance(x_est, px, pw)
#
#     N_eff = 1.0 / (pw.dot(pw.T))[0, 0]  # Effective particle number
#     if N_eff < NTh:
#         px, pw = re_sampling(px, pw)
#     return x_est, p_est, px, pw



class ParticleFilter:
    def __init__(self, landmarks, Z0, STATE_DIM, NP, DT):
        self.px = np.zeros((STATE_DIM, NP))  # Particle store
        self.pw = np.zeros((1, NP)) + 1.0 / NP  # Particle weight

        self.landmarks = landmarks
        self.DT = DT
        self.NP = NP
        self.cov = R_sim

        self.NUM_MAX_THREADS = 6

        # initialize particle distribution from noisy landmark observations
        pdf = estimate_zx(Z0, landmarks, np.pi / 2.0)
        K = NP // len(pdf) # number of particle groups
        for i, pd in enumerate(pdf):
            self.px[:4, i * K: (i + 1) * K] = np.array([pd[:4] for _ in range(K)]).T

        self.z_t_1 = pdf.mean(axis=0).reshape((4, 1))
        self.initialized = False
        self.x_est = None
        self.p_est = None

    def estimate_control(self, Z):

        u = None
        if len(Z):
            xx = estimate_zx(Z, self.landmarks, self.z_t_1[3, 0])
            zx = xx.mean(axis=0).reshape((4, 1))
            dx = zx - self.z_t_1
            v = dx / self.DT
            self.z_t_1 = zx.copy()

            var = xx.std(axis=0)
            self.cov = np.diag(var) ** 0.1


            if (self.initialized):
                u = v.copy()
            self.initialized = True
        else:
            self.initialized = False

        return u

    def __call__(self, Z, u):
        self.x_est, self.p_est, self.px, self.pw = self.pf_localization(self.px, self.pw, Z, u, self.DT, self.NP, self.NUM_MAX_THREADS)

    def getState(self):
        return self.x_est

    def getCovariance(self):
        return self.p_est

    def generate_noisy_control(self, u):
        ud1 = u[0, 0] + np.random.randn() * self.cov[0, 0]
        ud2 = u[1, 0] + np.random.randn() * self.cov[1, 1]
        ud3 = u[2, 0] + np.random.randn() * self.cov[2, 2]
        ud4 = u[3, 0] + np.random.randn() * self.cov[3, 3]
        ud = np.array([[ud1, ud2, ud3, ud4]]).T
        return ud


    def particle_filter_worker(self, ip, px, pw, z, u, dt):
        x = np.array([px[:, ip]]).T
        w = pw[0, ip]

        # Predict with random input sampling
        ud = self.generate_noisy_control(u)
        x = motion_model(x, ud, dt)

        # Calculate Importance Weight
        for i, y in enumerate(z):
            x_noise = get_noisy_reading(x, y[:3])
            dx = x[0, 0] - x_noise[0]
            dy = x[1, 0] - x_noise[1]
            dz = x[2, 0] - x_noise[2]
            pre_z = math.hypot(dx, dy, dz)

            delta_z = pre_z - y[0]
            phi = pi_2_pi(np.arctan2(dy, dx) - x[3, 0])
            phi_z = pi_2_pi(phi - y[1]) + np.pi / 2.0

            p_dz = gauss_likelihood(delta_z, math.sqrt(Q[0, 0]))
            p_dphi = gauss_likelihood(phi_z, math.sqrt(Q[1, 1]))
            w = w * (p_dz + p_dphi)

        px[:, ip] = x[:, 0]
        pw[0, ip] = w

    def pf_localization(self, px, pw, z, u, dt, NP, max_threads):
        NTh = NP / 2.0  # Number of particles for re-sampling
        num_particles = len(px[0])

        # Create a thread pool with a maximum number of threads
        with ThreadPoolExecutor(max_threads) as executor:
            # Submit particle_filter_worker for each particle
            futures = [executor.submit(self.particle_filter_worker, ip, px, pw, z, u, dt) for ip in range(num_particles)]

            # Wait for all tasks to complete
            for future in futures:
                future.result()

        pw = pw / pw.sum()  # Normalize

        x_est = px.dot(pw.T)
        p_est = calc_covariance(x_est, px, pw)

        N_eff = 1.0 / (pw.dot(pw.T))[0, 0]  # Effective particle number
        # print(N_eff, NTh)
        if N_eff <= NTh:
            # print('resampling')
            px, pw = re_sampling(px, pw, NP)

        return x_est, p_est, px, pw

