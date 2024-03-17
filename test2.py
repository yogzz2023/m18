import numpy as np
from scipy.stats import multivariate_normal

class KalmanFilter:
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x  # Dimension of state vector
        self.dim_z = dim_z  # Dimension of measurement vector

        # State transition matrix (F)
        self.F = np.eye(dim_x)

        # Observation matrix (H)
        self.H = np.zeros((dim_z, dim_x))
        self.H[:dim_z, :dim_z] = np.eye(dim_z)  # Adjust the dimensions of H to match dim_z

        # Process noise covariance matrix (Q)
        self.Q = np.eye(dim_x)

        # Measurement noise covariance matrix (R)
        self.R = np.eye(dim_z)

        # State vector (x)
        self.x = np.zeros(dim_x)

        # Covariance matrix (P)
        self.P = np.eye(dim_x)

    def predict(self):
        # Predict the next state
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        # Update the state based on the measurement
        y = z - np.dot(self.H, self.x[:self.dim_z])  # Use only the relevant dimensions of the state vector
        S = np.dot(np.dot(self.H, self.P[:self.dim_z, :]), self.H.T) + self.R
        K = np.dot(np.dot(self.P[:, :self.dim_z], self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.dim_x)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

def load_data(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()[1:]  # Skip the header
            data = []
            for line in lines:
                values = line.strip().split(',')
                values = [value.strip() for value in values if value.strip()]  # Remove empty strings and leading/trailing whitespaces
                data.append([float(value) for value in values[1:]])  # Skip the first empty value
            data = np.array(data)
        return data
    except FileNotFoundError:
        print("Error: CSV file not found.")
        return None

def perform_kalman_filtering(data):
    if data is None:
        return

    # Extract necessary columns for calculations
    measured_values = data[:, 10:14]  # Columns 11-14 represent MR, MA, ME, MT
    target_ids = data[:, 2]  # Column 3 represents TN (Target Number)

    # Define Kalman Filter parameters
    kf = KalmanFilter(dim_x=4, dim_z=4)  # 4D state (MR, MA, ME, MT), 4D measurement

    # Define the process noise covariance
    kf.Q *= 0.01

    # Define the measurement noise covariance
    kf.R *= 0.01

    # Initialize the state
    kf.x = np.zeros(4)

    # Dictionary to store likelihood probabilities for each target ID
    likelihoods = {}

    # Perform Joint Probabilistic Data Association using Kalman Filter
    for i in range(len(measured_values)):
        measurement = measured_values[i]
        target_id = int(target_ids[i])

        # Predict the next state
        kf.predict()

        # Update the state based on the measurement
        kf.update(measurement)

        # Calculate likelihood probability
        likelihood = multivariate_normal.pdf(measurement, mean=kf.x, cov=kf.P)

        # Store the likelihood probability for the target ID
        if target_id in likelihoods:
            likelihoods[target_id].append(likelihood)
        else:
            likelihoods[target_id] = [likelihood]

    # Find the most likely target based on the maximum likelihood
    most_likely_target_id = max(likelihoods.items(), key=lambda x: sum(x[1]))[0]

    # Print the predicted ranges, azimuths, elevations, and times for the most likely target
    predicted_ranges = kf.x[0]
    predicted_azimuths = kf.x[1]
    predicted_elevations = kf.x[2]
    predicted_times = kf.x[3]

    print(f"Most likely target ID: {most_likely_target_id}")
    print(f"Predicted Range: {predicted_ranges}")
    print(f"Predicted Azimuth: {predicted_azimuths}")
    print(f"Predicted Elevation: {predicted_elevations}")
    print(f"Predicted Time: {predicted_times}")

if __name__ == "__main__":
    file_path = "config.csv"  # Replace with the path to your CSV file
    data = load_data(file_path)
    if data is not None:
        perform_kalman_filtering(data)