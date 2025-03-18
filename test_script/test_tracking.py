import numpy as np
import matplotlib.pyplot as plt
from numba import njit

#------------------------------------------------------------------------------
# 1. Define Orbital Elements and Conversion Function (COE to ECI State)
#------------------------------------------------------------------------------
# Orbital elements for Satellite A and Satellite B (angles in degrees)
# Format: [semi-major axis (m), eccentricity, inclination (deg), RAAN (deg),
#          argument of perigee (deg), true anomaly (deg)]

# Satellite A
a_A    = 7000e3
e_A    = 0
i_A    = 50
raan_A = 0
argp_A = 0
nu_A   = 0

# Satellite B
a_B    = 7000e3
e_B    = 0.00
i_B    = 50
raan_B = 15
argp_B = 0
nu_B   = 0

# Gravitational parameter (Earth)
MU_EARTH = 3.986e14  # [m^3/s^2]

@njit
def coe2rv(a, e, i_deg, raan_deg, argp_deg, nu_deg, mu=MU_EARTH):
    """
    Convert orbital elements to ECI state vector.
    Angles are in degrees.
    Returns state vector: [x, y, z, vx, vy, vz].
    """
    # Convert angles to radians
    i    = np.radians(i_deg)
    raan = np.radians(raan_deg)
    argp = np.radians(argp_deg)
    nu   = np.radians(nu_deg)
    
    # Calculate orbit parameters
    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(nu))
    
    # Perifocal coordinates (position and velocity)
    r_pf = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    v_pf = np.array([
        -np.sqrt(mu/p) * np.sin(nu),
         np.sqrt(mu/p) * (e + np.cos(nu)),
         0.0
    ])
    
    # Rotation matrix from perifocal to ECI
    cos_raan = np.cos(raan)
    sin_raan = np.sin(raan)
    cos_i    = np.cos(i)
    sin_i    = np.sin(i)
    cos_argp = np.cos(argp)
    sin_argp = np.sin(argp)
    
    Q = np.empty((3, 3))
    Q[0,0] = cos_raan*cos_argp - sin_raan*sin_argp*cos_i
    Q[0,1] = -cos_raan*sin_argp - sin_raan*cos_argp*cos_i
    Q[0,2] = sin_raan*sin_i
    Q[1,0] = sin_raan*cos_argp + cos_raan*sin_argp*cos_i
    Q[1,1] = -sin_raan*sin_argp + cos_raan*cos_argp*cos_i
    Q[1,2] = -cos_raan*sin_i
    Q[2,0] = sin_argp*sin_i
    Q[2,1] = cos_argp*sin_i
    Q[2,2] = cos_i
    
    r_eci = Q @ r_pf
    v_eci = Q @ v_pf
    return np.concatenate((r_eci, v_eci))

# Compute initial ECI states
A_init = coe2rv(a_A, e_A, i_A, raan_A, argp_A, nu_A)
B_init = coe2rv(a_B, e_B, i_B, raan_B, argp_B, nu_B)

#------------------------------------------------------------------------------
# 2. True State Propagation using Two-Body Dynamics (RK4 Integration)
#------------------------------------------------------------------------------
DT = 1    # Time step [s]
N  = 20000   # Number of simulation steps

@njit
def two_body_acceleration(state, mu=MU_EARTH):
    """Compute gravitational acceleration for a state vector [x, y, z, vx, vy, vz]."""
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x*x + y*y + z*z) + 1e-9  # avoid division by zero
    ax = -mu * x / (r**3)
    ay = -mu * y / (r**3)
    az = -mu * z / (r**3)
    return np.array([ax, ay, az])

@njit
def rk4_step(state, dt, mu=MU_EARTH):
    """Propagate state one step using RK4 integration."""
    def deriv(s):
        x, y, z, vx, vy, vz = s
        a = two_body_acceleration(s, mu)
        return np.array([vx, vy, vz, a[0], a[1], a[2]])
    
    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)
    return state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

#------------------------------------------------------------------------------
# 3. EKF Setup for Relative State Estimation
#------------------------------------------------------------------------------
# True relative state: x_rel_true = [r_B - r_A; v_B - v_A]
# We assume a constant-velocity model for the filter:
F = np.block([
    [np.eye(3), DT * np.eye(3)],
    [np.zeros((3, 3)), np.eye(3)]
])

# Process noise covariance (tuned for the application)
Q = 1e-4 * np.eye(6)  # Process noise covariance

# Initial EKF covariance (large uncertainty)
P = 1e8 * np.eye(6)   # Initial state error covariance

#------------------------------------------------------------------------------
# 4. Measurement Models and Noise
#------------------------------------------------------------------------------
# (a) 1-D Angular (Cosine) Measurement
# h1(x) = ( (x[:3]/||x[:3]||) · u ) - 1,
# where u is the true relative unit vector.
@njit
def h1(x, u_meas):
    """1-D measurement: cosine error between estimated unit vector and measured unit vector."""
    r = x[:3]
    norm_r = np.sqrt(np.dot(r, r)) + 1e-9
    u_est = r / norm_r
    return np.array([np.dot(u_est, u_meas) - 1.0])

@njit
def H1_jacobian(x, u_meas):
    """Jacobian of h1(x) with respect to the state x (only depends on position part)."""
    r = x[:3]
    norm_r = np.sqrt(np.dot(r, r)) + 1e-9
    u_est = r / norm_r
    I = np.eye(3)
    diff = I - np.outer(u_est, u_est)
    dH_dr = (1.0 / norm_r) * diff @ u_meas
    H = np.zeros((1, 6))
    H[0, :3] = dH_dr
    return H

# Measurement noise for the 1-D measurement (not njitted because used outside)
R1d = np.array([[1e-5]])  # 1-D measurement noise variance

# (b) Absolute Position (Pseudo) Measurement of the relative position
# h_pos(x) = r_B/A (i.e., the first 3 components of x)
@njit
def h_pos(x):
    return x[:3]

# Its Jacobian is simply:
H_pos = np.hstack([np.eye(3), np.zeros((3, 3))])  # 3x6 matrix

# Measurement noise for the absolute position measurement
R_pos = 1e3 * np.eye(3)  # Adjust as needed (here, variance ~1e4)

#------------------------------------------------------------------------------
# 5. Main Simulation Loop with Initial Error Injection
#------------------------------------------------------------------------------
# Initialize true states and EKF state estimate
A_state = A_init.copy()
B_state = B_init.copy()

# Compute the true initial relative state:
x_true_init = np.concatenate([B_init[:3] - A_init[:3], B_init[3:] - A_init[3:]])

# Inject initial error into the EKF estimate:
# For example, add a random error with std dev 100 m in position and 1 m/s in velocity.
pos_error_std = 100.0  # [m]
vel_error_std = 1.0    # [m/s]
error_r = np.random.randn(3) * pos_error_std
error_v = np.random.randn(3) * vel_error_std
x_est = np.concatenate([x_true_init[:3] + error_r, x_true_init[3:] + error_v])

# Optionally, print the initial angular error:
u_true_init = x_true_init[:3] / (np.linalg.norm(x_true_init[:3]) + 1e-9)
u_est_init = x_est[:3] / (np.linalg.norm(x_est[:3]) + 1e-9)
init_angle_error = np.arccos(np.clip(np.dot(u_true_init, u_est_init), -1.0, 1.0))
print("Initial angular error (deg):", np.degrees(init_angle_error))

# Storage arrays for plotting
A_traj = []
B_traj = []
B_est_traj = []
angle_error = []
pos_error_arr = []  # Absolute position estimation error for satellite B
time_array = []
rel_vel_true_list = []
rel_vel_est_list = []
# New arrays for relative distance tracking:
rel_dist_true_list = []
rel_dist_est_list = []

for k in range(N):
    t = k * DT
    
    # Propagate true states using RK4
    A_state = rk4_step(A_state, DT, MU_EARTH)
    B_state = rk4_step(B_state, DT, MU_EARTH)
    
    # True relative state
    r_true = B_state[:3] - A_state[:3]
    v_true = B_state[3:] - A_state[3:]
    x_true = np.concatenate([r_true, v_true])
    
    # 1-D Measurement: Compute the true unit vector from A to B
    norm_r_true = np.linalg.norm(r_true) + 1e-9
    u_meas = r_true / norm_r_true
    
    # Generate the 1-D measurement with noise
    y1_true = h1(x_true, u_meas)
    noise1 = np.random.multivariate_normal(np.zeros(1), R1d)
    y1 = y1_true + noise1  # 1-D noisy measurement
    
    # Absolute Position Measurement: Simulated from approximate absolute positions
    noise_pos = np.random.multivariate_normal(np.zeros(3), R_pos)
    z_pos = r_true + noise_pos  # 3-D measurement of relative position
    
    # EKF Prediction Step
    x_pred = F @ x_est
    P_pred = F @ P @ F.T + Q
    
    # EKF Update with 1-D Angular Measurement
    H1 = H1_jacobian(x_pred, u_meas)
    y1_pred = h1(x_pred, u_meas)
    nu1 = y1 - y1_pred  # Innovation for 1-D measurement
    S1 = H1 @ P_pred @ H1.T + R1d
    K1 = P_pred @ H1.T @ np.linalg.inv(S1)
    x_upd1 = x_pred + (K1 @ nu1).flatten()
    P_upd1 = (np.eye(6) - K1 @ H1) @ P_pred

    # EKF Update with Absolute Position Measurement (Pseudo Measurement)
    nu_pos = z_pos - h_pos(x_upd1)  # Innovation for absolute measurement
    S_pos = H_pos @ P_upd1 @ H_pos.T + R_pos
    K_pos = P_upd1 @ H_pos.T @ np.linalg.inv(S_pos)
    x_est = x_upd1 + K_pos @ nu_pos
    P = (np.eye(6) - K_pos @ H_pos) @ P_upd1
    
    # Save data for plotting
    A_traj.append(A_state[:3].copy())
    B_traj.append(B_state[:3].copy())
    # Reconstruct satellite B's estimated absolute position from A and estimated relative state
    B_est = A_state[:3] + x_est[:3]
    B_est_traj.append(B_est)
    
    # Compute angular error between true and estimated relative directions
    u_true = r_true / norm_r_true
    u_est = x_est[:3] / (np.linalg.norm(x_est[:3]) + 1e-9)
    angle_err = np.arccos(np.clip(np.dot(u_true, u_est), -1.0, 1.0))
    angle_error.append(angle_err)
    
    # Save relative velocity magnitude (true and estimated)
    rel_vel_true_list.append(np.linalg.norm(v_true))
    rel_vel_est_list.append(np.linalg.norm(x_est[3:]))
    
    # Compute position estimation error (absolute error in estimated B position)
    pos_err = np.linalg.norm(B_state[:3] - B_est)
    pos_error_arr.append(pos_err)
    
    # Save relative distance (true and estimated)
    rel_dist_true_list.append(np.linalg.norm(r_true))
    rel_dist_est_list.append(np.linalg.norm(x_est[:3]))
    
    time_array.append(t)

# Convert lists to arrays for plotting
A_traj = np.array(A_traj)
B_traj = np.array(B_traj)
B_est_traj = np.array(B_est_traj)
angle_error = np.array(angle_error)
pos_error_arr = np.array(pos_error_arr)
time_array = np.array(time_array)
rel_vel_true_list = np.array(rel_vel_true_list)
rel_vel_est_list = np.array(rel_vel_est_list)
rel_dist_true_list = np.array(rel_dist_true_list)
rel_dist_est_list = np.array(rel_dist_est_list)

#------------------------------------------------------------------------------
# 6. Visualization
#------------------------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D

# Plot 3D trajectories (absolute positions)
fig1 = plt.figure(figsize=(12, 6))
ax1 = fig1.add_subplot(121, projection='3d')
ax1.plot(A_traj[:,0], A_traj[:,1], A_traj[:,2], label='Satellite A (True)', c='b')
ax1.plot(B_traj[:,0], B_traj[:,1], B_traj[:,2], label='Satellite B (True)', c='g')
ax1.plot(B_est_traj[:,0], B_est_traj[:,1], B_est_traj[:,2],
         label='Satellite B (EKF Est.)', c='r', linestyle='--')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_zlabel('z (m)')
ax1.set_title('3D Orbits (ECI)')
ax1.legend()

# Plot pointing (angular) error over time
ax2 = fig1.add_subplot(122)
ax2.plot(time_array/60.0, angle_error*180/np.pi, c='purple')
ax2.set_xlabel('Time (min)')
ax2.set_ylabel('Pointing Error (deg)')
ax2.set_title('Pointing Error vs. Time')
plt.tight_layout()

# Plot relative velocity (true vs. estimated)
fig2 = plt.figure(figsize=(8, 5))
ax_vel = fig2.add_subplot(111)
ax_vel.plot(time_array/60.0, rel_vel_true_list, label='True Relative Velocity', c='blue')
ax_vel.plot(time_array/60.0, rel_vel_est_list, label='Estimated Relative Velocity', c='red', linestyle='--')
ax_vel.set_xlabel('Time (min)')
ax_vel.set_ylabel('Relative Velocity (m/s)')
ax_vel.set_title('Relative Velocity vs. Time')
ax_vel.grid(True)
ax_vel.legend()

# Plot position estimation error over time
fig3 = plt.figure(figsize=(8, 5))
ax_err = fig3.add_subplot(111)
ax_err.plot(time_array/60.0, pos_error_arr, label='Position Estimation Error', c='magenta')
ax_err.set_xlabel('Time (min)')
ax_err.set_ylabel('Position Error (m)')
ax_err.set_title('Absolute Position Estimation Error vs. Time')
ax_err.grid(True)
ax_err.legend()

# Plot relative distance (true vs. estimated) over time
fig4 = plt.figure(figsize=(8, 5))
ax_dist = fig4.add_subplot(111)
ax_dist.plot(time_array/60.0, rel_dist_true_list, label='True Relative Distance', c='green')
ax_dist.plot(time_array/60.0, rel_dist_est_list, label='Estimated Relative Distance', c='orange', linestyle='--')
ax_dist.set_xlabel('Time (min)')
ax_dist.set_ylabel('Relative Distance (m)')
ax_dist.set_title('Relative Distance vs. Time')
ax_dist.grid(True)
ax_dist.legend()

plt.show()
