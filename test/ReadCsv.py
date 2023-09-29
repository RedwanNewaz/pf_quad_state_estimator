import csv
from time import sleep

import numpy as np


def read_csv(csv_file):
    # Initialize an empty list to store the data
    data = []

    # Open and read the CSV file with a header
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)

        # Iterate through each row in the CSV file
        for row in csv_reader:
            data.append(row)

    # Print the resulting list of dictionaries
    wp = np.zeros((0, 4))
    for row in data:
        tf_row = {key: float(val) for key, val in row.items()}
        wp_i = np.array([tf_row['time'], tf_row['x'], tf_row['y'], tf_row['z']])
        wp = np.vstack((wp, wp_i))
    return wp

if __name__ == '__main__':
    filename = 'traj/traj_eight.csv'
    wps = read_csv(filename)
    print(wps)