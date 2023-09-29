import numpy as np

class QuadController:
    def __init__(self, kp_position, ki_position, kd_position, kp_yaw, ki_yaw, kd_yaw):
        # Initialize PID gains for position control as a 3x3 matrix
        self.kp_position = np.diag(np.array([kp_position, kp_position, kp_position]))
        self.ki_position = np.diag(np.array([ki_position, ki_position, ki_position]))
        self.kd_position = np.diag(np.array([kd_position, kd_position, kd_position]))

        # Initialize PID gains for yaw control
        self.kp_yaw = kp_yaw
        self.ki_yaw = ki_yaw
        self.kd_yaw = kd_yaw

        # Initialize errors and integrals as 4x1 vectors
        self.error_position = np.zeros((3, 1))
        self.integral_position = np.zeros((3, 1))
        self.error_yaw = 0.0
        self.integral_yaw = 0.0
        self.prev_yaw_error = 0.0

    def update(self, setpoint_position, current_position, setpoint_yaw, current_yaw):
        # Calculate position errors as a 3x1 vector
        self.error_position = setpoint_position - current_position

        # Calculate position integrals as a 3x1 vector
        self.integral_position += self.error_position

        # Calculate position PID control outputs as a 3x1 vector
        control_position = np.dot(self.kp_position, self.error_position) + np.dot(self.ki_position, self.integral_position) - np.dot(self.kd_position, self.error_position)

        # Convert yaw values to radians
        setpoint_yaw = np.radians(setpoint_yaw)
        current_yaw = np.radians(current_yaw)

        # Calculate yaw error and integral in radians
        yaw_error = setpoint_yaw - current_yaw
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi  # Normalize between -π and π
        self.error_yaw = yaw_error
        self.integral_yaw += yaw_error

        # Calculate yaw PID control output
        control_yaw = self.kp_yaw * self.error_yaw + self.ki_yaw * self.integral_yaw + self.kd_yaw * (self.error_yaw - self.prev_yaw_error)
        self.prev_yaw_error = self.error_yaw

        # Combine position and yaw control outputs as a 4x1 vector
        control = np.vstack((control_position, control_yaw))

        return control

if __name__ == '__main__':
    # Example usage:
    pid = QuadController(kp_position=0.1, ki_position=0.01, kd_position=0.05,
                                kp_yaw=1.0, ki_yaw=0.1, kd_yaw=0.2)

    # Set desired setpoints and current states as appropriate 3x1 or scalar values
    setpoint_position = np.array([[5.0], [3.0], [2.0]])  # [x, y, z]
    current_position = np.array([[0.0], [0.0], [0.0]])
    setpoint_yaw = np.pi / 2  # π (pi) / 2 radians (90 degrees)
    current_yaw = np.pi / 2  # Radians

    # Update the controller to get control outputs
    control = pid.update(setpoint_position, current_position, setpoint_yaw, current_yaw)

    print(control)

    # Now, 'control' contains the combined control outputs for position and yaw control.
