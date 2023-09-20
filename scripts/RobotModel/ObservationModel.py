import numpy as np
import math



def pi_2_pi(angle):

    return (angle + math.pi) % (2 * math.pi) - math.pi



def convert_spherical_to_cartesian(Z):
    r, phi, theta = Z[:3]
    phi =  pi_2_pi(phi)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z, 0])

def estimate_control(Z_t, Z_t_1, dt):
    x_t = convert_spherical_to_cartesian(Z_t)
    x_t_1 = convert_spherical_to_cartesian(Z_t_1)
    v = (x_t - x_t_1) / dt
    return v.reshape((4, 1))


def calc_rho(rho, z_rho, beta):
    r = (rho ** 2) + (z_rho ** 2) - 2 * rho * z_rho * math.cos(beta)
    r = math.sqrt(r)
    return r

def calc_theta(alpha, z_rho, rho, beta):
    gamma = z_rho * math.sin(beta) / rho
    gamma = math.asin(gamma)
    return alpha - gamma


def estimate_zx(Z, landmarks, heading):

    markers = np.squeeze(landmarks)

    X = np.zeros((0, 4))

    beta = 0.99
    for z in Z:
        # heading = min(heading, z[-1])
        heading = beta * heading  + (1 - beta) * z[-1]

    for z in Z:
        i = int(z[-2])
        z_rho, phi, polar_angle = z[:3]
        alpha = math.atan2(markers[i, 1], markers[i, 0])
        rho = np.linalg.norm(markers[i, :2])
        z_beta = pi_2_pi(phi - alpha)
   
        # estimate robot coord
        r_hat = calc_rho(rho, z_rho, z_beta)
        theta_hat = calc_theta(alpha, z_rho, r_hat, z_beta)

        theta_hat = pi_2_pi(theta_hat + 1.57)


        if heading > np.pi / 2 or heading < -np.pi / 2:
            y = -r_hat * math.cos(theta_hat) + markers[i, 0] 
            x = -r_hat * math.sin(theta_hat) + markers[i, 1]
        else: 
            x = r_hat * math.cos(theta_hat) + markers[i, 1] 
            y = r_hat * math.sin(theta_hat) - markers[i, 0] 

       
        zz = -z_rho * math.cos(polar_angle) + markers[i, 2]

        print(heading, theta_hat)


        xx = np.array([x, y, zz, heading])
        X = np.vstack((X, xx))


    return X