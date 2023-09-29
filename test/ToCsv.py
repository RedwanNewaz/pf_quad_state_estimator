from collections import defaultdict
import csv
import numpy as np


def write_csv(csv_file, gen, DT):
    X = np.zeros((4, 1))
    history = defaultdict(list)
    time = 0
    for u in gen:
        X = X + u * DT
        time += DT
        # print(X)
        history['time'].append(f"{time:.3f}")
        history['x'].append(f"{X[0, 0]:.3f}")
        history['y'].append(f"{X[1, 0]:.3f}")
        history['z'].append(1.200)

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write the header row
        writer.writerow(history.keys())
        # Write the data from the defaultdict to the CSV file
        val = list(history.values())
        for item in zip(*val):
            writer.writerow(item)