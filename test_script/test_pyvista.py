import simpy
import pyvista as pv
import numpy as np
import time

# Define the simulation process that rotates a sphere
def simulation_process(env, plotter, mesh, dt):
    while True:
        # For demonstration: rotate the mesh by 2 degrees around z-axis
        mesh.rotate_z(2, inplace=True)
        # Refresh the visualization
        plotter.update()
        # Wait for the next time step (simulation time dt)
        yield env.timeout(dt)

# Create a PyVista Plotter and sphere mesh
plotter = pv.Plotter()
mesh = pv.Sphere(radius=1.0, theta_resolution=30, phi_resolution=30)
plotter.add_mesh(mesh, color='orange')

# Open the plotter in non-blocking mode
plotter.show(auto_close=False, interactive_update=True)

# Create a SimPy environment
env = simpy.Environment()
# Schedule our simulation process with a time step dt (in simulation time)
dt = 0.1  # simulation time step; adjust as needed
env.process(simulation_process(env, plotter, mesh, dt))

# Run the simulation in a loop that also sleeps a little to keep CPU usage low.
# We run for a fixed amount of simulation time (or you could run forever)
sim_time = 10  # total simulation time in seconds
start_time = time.time()

while env.now < sim_time:
    # Run one simulation step (advance until the next event)
    env.step()
    # Sleep a bit to allow the GUI event loop to process (non-blocking update)
    time.sleep(0.01)

# When simulation is complete, close the plotter window.
plotter.close()
