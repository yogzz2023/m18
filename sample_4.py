import csv
import numpy as np
import matplotlib.pyplot as plt

def jpda_kalman_filter(data, target_id):
    # Extract relevant data for the target ID
    target_data = [(row[10], row[11], row[12], row[13]) for row in data if row[1] == target_id]

    # Initialize state vector and covariance matrix
    x = np.array([float(target_data[0][0]), float(target_data[0][1]), float(target_data[0][2]), 0, 0, 0])  # Position and velocity
    P = np.eye(6)  # Initial covariance matrix

    # Kalman filter parameters
    dt = 1  # Time step (assuming constant velocity model)
    F = np.array([[1, 0, 0, dt, 0, 0],
                  [0, 1, 0, 0, dt, 0],
                  [0, 0, 1, 0, 0, dt],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1]])  # State transition matrix
    Q = np.eye(6) * 0.01  # Process noise covariance

    # Measurement function
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0]])
    R = np.eye(3) * 0.1   # Measurement noise covariance

    measured_ranges = []
    predicted_ranges = []
    measured_times = []

    for mr_str, ma_str, me_str, mt_str in target_data:
        mr = float(mr_str)
        ma = float(ma_str)
        me = float(me_str)
        mt = float(mt_str)

        # Measurement
        z = np.array([mr, ma, me])

        # Calculate predicted values
        predicted_range = x[0]
        predicted_ranges.append(predicted_range)

        measured_ranges.append(mr)
        measured_times.append(mt)

        # Prediction step
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Update step
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ y
        P = (np.eye(6) - K @ H) @ P_pred

    return measured_ranges, predicted_ranges, measured_times, target_id

# Read data from CSV file
data = []
with open('config_1.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row
    for row in reader:
        data.append(row)

# Get unique target IDs
target_ids = set(row[1] for row in data if row[1])

# Plot measured range, predicted range vs measured time for each target ID
for target_id in target_ids:
    measured_ranges, predicted_ranges, measured_times, target_id = jpda_kalman_filter(data, target_id)

    plt.figure(figsize=(8, 6))

    plt.plot(measured_times, measured_ranges, 'o-', label='Measured Range', color='blue')
    plt.plot(measured_times, predicted_ranges, '--', label='Predicted Range', color='red')

    max_likelihood_index = np.argmax(predicted_ranges)
    plt.plot(measured_times[max_likelihood_index], predicted_ranges[max_likelihood_index], 'ro')

    for i in range(len(measured_times)):
        plt.text(measured_times[i], measured_ranges[i], f'{measured_ranges[i]:.2f}', fontsize=8, ha='right')
        plt.text(measured_times[i], predicted_ranges[i], f'{predicted_ranges[i]:.2f}', fontsize=8, ha='left')

    plt.xlabel('Measured Time')
    plt.ylabel('Range')
    plt.title(f'Measured vs Predicted Range for Target ID: {target_id}')
    plt.grid(True)
    plt.legend()
    plt.show()
