import simpy
import numpy as np
from vispy import app, scene
import time
import cProfile
import pstats

# ----- Simulation Code (SimPy) -----
class MovingObjects3D:
    def __init__(self, env, n_objects):
        self.env = env
        self.n = n_objects
        # Initialize positions (3D: x, y, z) within a 100×100×100 cube.
        self.positions = np.random.rand(n_objects, 3) * 100  
        # Random velocities (in pixels per second); values between -25 and +25.
        self.velocities = (np.random.rand(n_objects, 3) - 0.5) * 50  
        # Start the SimPy process.
        self.process = env.process(self.update_positions())

    def update_positions(self):
        dt = 0.05  # simulation time step (seconds)
        while True:
            # Update positions: new_position = old_position + velocity * dt
            self.positions += self.velocities * dt
            # Wrap positions to remain within the 100×100×100 cube.
            self.positions %= 100
            yield self.env.timeout(dt)

# ----- Visualization Code (Vispy) -----
# Create a Vispy SceneCanvas.
canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
view = canvas.central_widget.add_view()

# Set a 3D camera with a turntable view
view.camera = scene.cameras.TurntableCamera(fov=45, azimuth=30, elevation=30, distance=200)

# Create a Markers visual to display the moving objects in 3D.
scatter = scene.visuals.Markers()
view.add(scatter)

# Create the SimPy environment and our moving objects.
env = simpy.Environment()
n_objects = 10000  # number of objects
moving_objects = MovingObjects3D(env, n_objects)

# Variables for profiling updates.
update_count = 0
accumulated_update_time = 0

# Define an update function called by a timer.
def update(event):
    global update_count, accumulated_update_time
    start_time = time.perf_counter()
    
    try:
        env.step()  # Process one simulation event.
    except simpy.core.EmptySchedule:
        pass

    # Update the 3D scatter plot with the current positions.
    scatter.set_data(moving_objects.positions, face_color='white', size=2)
    canvas.update()

    end_time = time.perf_counter()
    elapsed = end_time - start_time
    update_count += 1
    accumulated_update_time += elapsed

    # Print average update time every 60 frames.
    if update_count % 60 == 0:
        avg_time = accumulated_update_time / 60
        print(f"Average update time over last 60 frames: {avg_time:.6f} seconds")
        accumulated_update_time = 0

# Set up a timer to update the simulation at roughly 60 FPS.
timer = app.Timer(interval=1/60.0, connect=update, start=True)

# If run as a standalone script, enable profiling.
if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        app.run()
    finally:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(20)
