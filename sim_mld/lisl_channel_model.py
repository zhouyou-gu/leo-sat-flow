import numpy as np

def w0_from_angular_spreading(Theta, wavelength):
    return wavelength / (np.pi * (Theta/2.0))
    
def rayleigh_range(w0, wavelength):
    return np.pi * w0**2 / wavelength

def beam_radius_at_z(w0, wavelength, z):
    zR = rayleigh_range(w0, wavelength)
    return w0 * np.sqrt(1 + (z / zR)**2)

def peak_intensity(w0, P):
    return 2.0 * P / (np.pi * w0**2)

def gaussian_beam_intensity(P, w0, wavelength, rho, z):
    I0 = peak_intensity(w0, P)
    w_z = beam_radius_at_z(w0, z, wavelength)
    return I0 * (w0 / w_z)**2 * np.exp(-2.0 * rho**2 / w_z**2)

def capacity_lower_bound(B, R, A, N0, P, w0, wavelength, rho, z):
    I = gaussian_beam_intensity(P, w0, wavelength, rho, z)
    numerator = (I * A * R)**2
    denominator = 2.0 * np.pi * np.e * (N0**2)
    argument = 1.0 + (numerator / denominator)
    return 0.5 * B * np.log2(argument) / 1e9  # Convert to Gbps

def capacity_relaxed(B, R, A, N0, P, w0, wavelength, z, sigma_jitter, epsilon):
    theta_max = sigma_jitter * np.sqrt(-2.0 * np.log(epsilon))
    rho_divergence = theta_max * z
    return capacity_lower_bound(B, R, A, N0, P, w0, wavelength, rho_divergence, z)*(1-epsilon)

def intensity_relaxed(P, w0, wavelength, z, sigma_jitter, epsilon):
    theta_max = sigma_jitter * np.sqrt(-2.0 * np.log(epsilon))
    rho_divergence = theta_max * z
    return gaussian_beam_intensity(P, w0, wavelength, rho_divergence, z)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Example parameters
    wavelength = 1.55e-6    # Wavelength in meters (1.55 microns typical in telecom)
    w0 = w0_from_angular_spreading(100e-6, wavelength)  # Beam waist in meters
    z_values = np.geomspace(1, 3e6, 100)  # from 0 to 3000 km
    rho = 0.0        # on-axis, for example
    P = 20  # Watts
    B = 1e9        # 1 GHz bandwidth
    R = 0.5        # A/W responsivity
    A = 1e-2      # Area in m^2 (example)
    N0 = 3e-7      # Example noise in A rms
    
    # Compute intensity vs. z at rho=0
    I_on_axis = np.array([gaussian_beam_intensity(P, w0, wavelength, rho, z) for z in z_values])
    
    # Compute capacity lower bound vs. z at rho=0
    C_on_axis = np.array([capacity_lower_bound(B, R, A, N0, P, w0, wavelength, rho, z) for z in z_values])
    
    sigma_jitter = 10e-6 # Jitter in radians
    epsilon = 1e-3
    # C_relaxed = np.array([capacity_relaxed(B, R, A, N0, P, w0, wavelength, z, sigma_jitter, epsilon) for z in z_values])    
    C_relaxed = capacity_relaxed(B, R, A, N0, P, w0, wavelength, z_values, sigma_jitter, epsilon)
    plt.figure(figsize=(10, 6))
    subplot = plt.subplot(131)
    subplot.set_box_aspect(1)  # Aspect ratio 1:1
    plt.plot(z_values/1e3, I_on_axis, label='Intensity on Axis')
    plt.title('Gaussian Beam Intensity vs. Distance')
    plt.xlabel('Distance (km)')
    plt.ylabel('Intensity (W/m^2)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.legend()

    subplot = plt.subplot(132)
    subplot.set_box_aspect(1)  # Aspect ratio 1:1
    plt.plot(z_values/1e3, C_on_axis, label='Capacity on Axis')
    print(C_on_axis,z_values)
    plt.title('Capacity Lower Bound vs. Distance')
    plt.xlabel('Distance (km)')
    plt.ylabel('Capacity (Gbits/s)')
    plt.xscale('log')
    plt.grid()
    plt.legend()
    plt.ylim(0, 30)  # Adjust as needed

    subplot = plt.subplot(133)
    subplot.set_box_aspect(1)  # Aspect ratio 1:1
    plt.plot(z_values/1e3, C_relaxed, label='Capacity with Jitter')
    print(C_on_axis,z_values)
    plt.title('Capacity Lower Bound vs. Distance')
    plt.xlabel('Distance (km)')
    plt.ylabel('Capacity (Gbits/s)')
    plt.xscale('log')
    plt.grid()
    plt.legend()
    # Set Y-axis limits for better visibility
    plt.ylim(0, 30)  # Adjust as needed
    

    plt.show()