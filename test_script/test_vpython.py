import simpy
import numpy as np
from vpython import canvas, vector, color, points, rate
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
            # Wrap positions so they remain within the area.
            self.positions %= 100
            yield self.env.timeout(dt)

# ----- Visualization Code (VPython) -----
def run_simulation():
    # Create the SimPy environment and our moving objects.
    env = simpy.Environment()
    n_objects = 10000  # many objects
    moving_objects = MovingObjects(env, n_objects)
    
    # Set up a VPython canvas.
    scene = canvas(title='Moving Objects (VPython)', width=800, height=600, center=vector(50, 50, 0))
    scene.background = color.black

    # Create a VPython points object.
    # Convert the initial NumPy positions to a list of VPython vectors.
    initial_points = [vector(x, y, 0) for x, y in moving_objects.positions]
    pts = points(pos=initial_points, radius=2, color=color.white)
    
    # Variables for per-frame profiling.
    update_count = 0
    accumulated_update_time = 0

    # Main loop: update simulation and VPython visualization.
    while True:
        rate(60)  # Limit loop to ~60 iterations per second.
        start_time = time.perf_counter()

        try:
            env.step()
        except simpy.core.EmptySchedule:
            pass

        # Update the positions of the VPython points.
        # Note: Converting the entire NumPy array each frame can be costly.
        # pts.pos = [vector(x, y, 0) for x, y in moving_objects.positions]
        new_positions = [vector(x, y, 0) for x, y in moving_objects.positions]
        for i, pos in enumerate(new_positions):
            pts.modify(i, pos=pos)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        update_count += 1
        accumulated_update_time += elapsed

        # Print average update time every 60 frames (~1 second).
        if update_count % 60 == 0:
            avg_time = accumulated_update_time / 60
            print(f"Average update time over last 60 frames: {avg_time:.6f} seconds")
            accumulated_update_time = 0

if __name__ == '__main__':
    # Profile the overall performance using cProfile.
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        run_simulation()
    finally:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(20)
