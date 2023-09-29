import numpy as np
from threading import Thread
from time import sleep
from test.ReadCsv import read_csv
from .quad_controller  import QuadController
from queue import Queue

class WaypointController(Thread):
    def __init__(self, waypoints, queue:Queue):
        self.waypoints = waypoints
        self.time = 0.0
        self.xEst = np.zeros((4, 1))
        self.pid = QuadController(kp_position=0.08, ki_position=0.00, kd_position=0.00,
                             kp_yaw=0.0, ki_yaw=0.0, kd_yaw=0.0)
        self.terminated = False
        self.queue = queue

        Thread.__init__(self)

    def run(self):

        prevX = np.zeros((4, 1))
        for j, row in enumerate(self.waypoints):
            dt = row[0] - self.time
            x = np.array([row[0], row[1], row[2], 0])
            if j == 0:
                prevX = x.copy()
            u = (x - prevX) / dt
            self.queue.put(u.reshape((4, 1)))



            # # Set desired setpoints and current states as appropriate 3x1 or scalar values
            # setpoint_position = np.reshape(row[1:], (3, 1)) # [x, y, z]
            # current_position = self.xEst[:3, :]
            # setpoint_yaw = np.pi / 2  # π (pi) / 2 radians (90 degrees)
            # current_yaw = np.pi / 2  # Radians
            #
            # # Update the controller to get control outputs
            # # while np.linalg.norm(self.pid.error_position) > 0.3:
            # control = self.pid.update(setpoint_position, current_position, setpoint_yaw, current_yaw)
            # # print(row, control)
            # self.queue.put(control * 0.1)
            # self.time += dt
            # sleep(dt)
            # print('[xEst]: ', self.xEst.T )
            sleep(dt)
        self.terminated = True


if __name__ == '__main__':
    filename = '../test/traj/traj_eight.csv'
    wps = read_csv(filename)
    x = np.zeros((4, 1))
    q = Queue()
    wpc = WaypointController(wps, x, q)
    wpc.start()

    while not wpc.terminated:
        while not q.empty():
            u = q.get()
            print(u.T)