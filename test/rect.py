import numpy as np

DT = 0.1  # time tick [s]
def calc_input():
    L = 6  # length of workspace [m]
    SIM_TIME = 200  # total simulation time
    N = SIM_TIME // 4  # each arm traversing time
    v = L / (N * DT)
    u_f = np.array([[0, v, 0, 0] for _ in range(N)])
    u_r = np.array([[v, 0, 0, 0] for _ in range(N)])
    u_b = np.array([[0, -v, 0, 0] for _ in range(N)])
    u_l = np.array([[-v, 0, 0, 0] for _ in range(N)])

    U = np.vstack((u_f, u_r, u_b, u_l)).reshape((SIM_TIME, 4, 1))

    for u in U:
        yield u