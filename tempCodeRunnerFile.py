import csv
import numpy as np
import matplotlib.pyplot as plt

def jpda_kalman_filter(data, target_id):
    # Extract relevant data for the target ID
    target_data = [(row[10], row[11], row[12], row[13], int(row[5]) if row[5] else 0) for row in data if row[1] == target_id]

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

    predicted_ranges = []
    predicted_azimuths = []
    predicted_elevations = []
    predicted_times = []
    likelihoods = []

    for mr_str, ma_str, me_str, mt_str , likelihood in target_data:
        mr = float(mr_str)
        ma = float(ma_str)
        me = float(me_str)
        mt = float(mt_str)

        # Prediction step
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Measurement
        z = np.array([mr , ma , me])

        # Update step
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ y
        P = (np.eye(6) - K @ H) @ P_pred

        # Calculate predicted values
        predicted_range = x[2]
        predicted_azimuth = x[3]
        predicted_elevation = x[4]
        predicted_time = mt

        predicted_ranges.append(predicted_range)
        predicted_azimuths.append(predicted_azimuth)
        predicted_elevations.append(predicted_elevation)
        predicted_times.append(predicted_time)
        likelihoods.append(likelihood)

    return predicted_ranges ,predicted_azimuths ,predicted_elevations ,predicted_times ,likelihoods

# Read data from CSV file
data = []
with open('config.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)   # Skip header row
    for row in reader:
        data.append(row)

# Get unique target IDs
target_ids = set([row[1] for row in data])

# Perform calculations for each target ID and plot the measured range against the predicted range
for target_id in target_ids:
    predicted_ranges ,predicted_azimuths ,predicted_elevations ,predicted_times ,likelihoods= jpda_kalman_filter(data ,target_id)

    # Plotting the data
    plt.figure(figsize=(12 ,6))
    plt.plot(predicted_times ,predicted_ranges ,label='Predicted Range' ,marker='o' ,color='red')
    
    max_likelihood_index= likelihoods.index(max(likelihoods))
    
    plt.scatter(predicted_times[max_likelihood_index] ,predicted_ranges[max_likelihood_index] ,s=200 ,color='green' ,label='Highest Likelihood Target')

    plt.xlabel('Measured Time')
    
plt.ylabel('Range')
plt.title(f'Measurement Range vs. Predicted Range for Target ID {target_id}')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()