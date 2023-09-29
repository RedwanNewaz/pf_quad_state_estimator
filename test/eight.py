import numpy as np
import matplotlib.pyplot as plt
from ToCsv import write_csv

DT = 0.1  # time tick [s]



def calc_input():
    # Define a parameter t
    t = np.linspace(0, 2 * np.pi, 200)
    x_scale = 3.5
    y_scale = 2.5

    # Define parametric equations for the number "8"
    x = x_scale + x_scale * np.cos(t) * np.sin(t)  # You can adjust the scaling factor (2) for size
    y = y_scale + y_scale * np.sin(t)  # You can adjust the vertical offset (+1) for position

    prevX = np.zeros((4, 1))
    for xx, yy in zip(x, y):
        X = np.array([xx, yy, 0, 0]).reshape((4, 1))
        u = (X - prevX) / DT
        prevX = X.copy()
        yield u


if __name__ == '__main__':
    filename = 'traj/traj_eight.csv'
    control = calc_input()
    write_csv(csv_file=filename, gen=control, DT=DT)

