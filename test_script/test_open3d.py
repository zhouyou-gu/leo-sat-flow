import simpy
import numpy as np
import open3d as o3d
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

# ----- Visualization Code (Open3D) -----
def main():
    # Create the SimPy environment and our moving objects.
    env = simpy.Environment()
    n_objects = 10000  # many objects
    moving_objects = MovingObjects(env, n_objects)
    
    # Create an Open3D Visualizer window.
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Open3D Moving Objects', width=800, height=600)
    
    # Prepare the initial point cloud.
    # Open3D uses 3D coordinates, so we add a z coordinate set to zero.
    points = np.hstack([moving_objects.positions, np.zeros((n_objects, 1))])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # Set the color of all points to white.
    colors = np.ones((n_objects, 3))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    vis.add_geometry(pcd)
    
    update_count = 0
    accumulated_update_time = 0
    
    try:
        while True:
            start_time = time.perf_counter()
            
            try:
                # Advance the simulation by processing one event.
                env.step()
            except simpy.core.EmptySchedule:
                pass
            
            # Update the point cloud positions (convert 2D positions to 3D by adding a zero z-coordinate).
            new_points = np.hstack([moving_objects.positions, np.zeros((n_objects, 1))])
            pcd.points = o3d.utility.Vector3dVector(new_points)
            
            # Refresh the Open3D visualization.
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            update_count += 1
            accumulated_update_time += elapsed
            
            # Print average update time every 60 frames.
            if update_count % 60 == 0:
                avg_time = accumulated_update_time / 60
                print(f"Average update time over last 60 frames: {avg_time:.6f} seconds")
                accumulated_update_time = 0
    except KeyboardInterrupt:
        # Allow graceful exit on Ctrl+C.
        pass
    
    vis.destroy_window()

# ----- Profiling -----
if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        main()
    finally:
        profiler.disable()
        # Print out the top 20 functions by cumulative time.
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(20)
