import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

# Define Kalman filter functions
def kalman_predict(x, P, F, Q):
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred

def kalman_update(x_pred, P_pred, z, H, R):
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_update = x_pred + K @ y
    P_update = P_pred - K @ H @ P_pred
    return x_update, P_update

# Load the CSV data into a DataFrame
df = pd.read_csv("test.csv", delimiter="\t")

# Initialize Kalman filters
num_targets = len(df)
initial_state = np.zeros((num_targets, 4))  # [MR, MA, ME, MT]
initial_covariance = np.eye(4) * 1e6  # Initial covariance matrix
F = np.eye(4)  # Transition matrix (identity for now)
Q = np.eye(4) * 0.1  # Process noise covariance
H = np.eye(4)  # Measurement matrix (identity for now)
R = np.eye(4) * 0.1  # Measurement noise covariance

kalman_filters = []
for i in range(num_targets):
    kalman_filters.append({
        'x': initial_state[i],
        'P': initial_covariance,
        'association_prob': 1 / num_targets  # Initial association probability
    })

# Loop over measurements
for index, row in df.iterrows():
    # Extract measurement
    z = np.array([row['MR'], row['MA'], row['ME'], row['MT']])
    
    # Prediction step for all Kalman filters
    for kf in kalman_filters:
        kf['x'], kf['P'] = kalman_predict(kf['x'], kf['P'], F, Q)
    
    # Data association and update step
    association_probs = np.zeros(num_targets)
    for i, kf in enumerate(kalman_filters):
        x_pred = kf['x']
        P_pred = kf['P']
        
        # Compute Mahalanobis distance
        mahalanobis_dist = np.linalg.norm(z - H @ x_pred) ** 2
        
        # Compute association probability using multivariate normal distribution
        association_probs[i] = multivariate_normal.pdf(z, mean=H @ x_pred, cov=H @ P_pred @ H.T + R)
    
    # Normalize association probabilities
    association_probs /= association_probs.sum()
    
    # Update step for all Kalman filters
    for i, kf in enumerate(kalman_filters):
        x_pred = kf['x']
        P_pred = kf['P']
        
        # Choose the measurement with the highest association probability
        max_assoc_index = np.argmax(association_probs)
        if max_assoc_index == i:
            kf['x'], kf['P'] = kalman_update(x_pred, P_pred, z, H, R)
        else:
            # If not associated, only predict without updating
            kf['x'], kf['P'] = kalman_predict(x_pred, P_pred, F, Q)

    # Update association probabilities based on JPDA
    for i, kf in enumerate(kalman_filters):
        kf['association_prob'] *= association_probs[i]

# Classify the most likely target associated with each measurement
for i, kf in enumerate(kalman_filters):
    print(f"Measurement {i+1} is most likely associated with Target {i+1} with probability {kf['association_prob']:.4f}")
