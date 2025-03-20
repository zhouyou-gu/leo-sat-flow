import simpy
import numpy as np
from vispy import app, scene
import time
import cProfile
import pstats

# ----- Simulation Code (SimPy) -----
class MovingObjects:
    def __init__(self, env, n_objects):
        self.env = env
        self.n = n_objects
        # Initialize positions (2D: x, y) within a 100x100 area.
        self.positions = np.random.rand(n_objects, 2) * 100  
        # Random velocities (in pixels per second), can be positive or negative.
        self.velocities = (np.random.rand(n_objects, 2) - 0.5) * 50  
        # Start the SimPy process.
        self.process = env.process(self.update_positions())

    def update_positions(self):
        dt = 0.05  # simulation time step (seconds)
        while True:
            # Update all positions: new_position = old_position + velocity * dt
            self.positions += self.velocities * dt
            # Wrap positions to remain within the 100x100 area.
            self.positions %= 100
            yield self.env.timeout(dt)

# ----- Visualization Code (Vispy) -----
# Create a Vispy SceneCanvas.
canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
view = canvas.central_widget.add_view()
# Set a 2D camera with a fixed view rectangle.
view.camera = scene.PanZoomCamera(rect=(0, 0, 100, 100))

# Create a Markers visual to display the moving objects.
scatter = scene.visuals.Markers()
view.add(scatter)

# Create the SimPy environment and our moving objects.
env = simpy.Environment()
n_objects = 10000  # many objects
moving_objects = MovingObjects(env, n_objects)

# Variables for granular update profiling.
update_count = 0
accumulated_update_time = 0

# Define an update function called by a timer.
def update(event):
    global update_count, accumulated_update_time
    start_time = time.perf_counter()
    
    try:
        # Advance the simulation by processing one event.
        env.step()
    except simpy.core.EmptySchedule:
        pass

    # Update the scatter plot with current positions.
    scatter.set_data(moving_objects.positions, face_color='white', size=2)
    canvas.update()

    # Measure the elapsed time for this update.
    end_time = time.perf_counter()
    elapsed = end_time - start_time
    update_count += 1
    accumulated_update_time += elapsed

    # Print average update time every 60 frames.
    if update_count % 60 == 0:
        avg_time = accumulated_update_time / 60
        print(f"Average update time over last 60 frames: {avg_time:.6f} seconds")
        accumulated_update_time = 0

# Set up a timer to update at roughly 60 FPS.
timer = app.Timer(interval=1/60.0, connect=update, start=True)

# Use cProfile to profile the overall performance if run as a standalone script.
if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        app.run()
    finally:
        profiler.disable()
        # Print out the top 20 functions by cumulative time.
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(20)
