import simpy
import numpy as np
from vispy import app, scene
from vispy.scene import SceneCanvas
from vispy.visuals import SphereVisual, MeshVisual
from vispy.geometry import MeshData, create_sphere
import time
import cProfile
import pstats
from PIL import Image
from vispy.visuals.filters import TextureFilter
from itertools import combinations

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
            # self.positions = self.positions[0:int(self.n/2)]
            # self.velocities = self.velocities[0:int(self.n/2)]
            # self.n = int(self.n/2)
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


def compute_face_coor(vertices):
    result = any(p[0] < 0 and q[0] < 0 and p[1] * q[1] < 0 for p, q in combinations(vertices, 2))  
    texcoords = []      
    for v in vertices:
        x, y, z = v
        theta = np.arctan2(y, x)
        phi = np.arccos(z / np.linalg.norm(v))
        if result and theta < 0:
            theta += 2 * np.pi
        u = (theta + np.pi) / (2 * np.pi)/2
        v = phi / np.pi
        texcoords.append([u, v])
    return texcoords

def compute_texcoords(vertices):
    texcoords = []
    for v in vertices:
        co = compute_face_coor(v)
        texcoords.extend(co)
    texcoords = np.array(texcoords)
    return texcoords

texture_path = "land_sea_texture.png" # Replace with your texture path
texture_image = Image.open(texture_path)
texture = np.array(texture_image)
texture = np.hstack([texture, texture])
print(texture.shape)
sphere_object = scene.visuals.Sphere(radius=0.1, method='latitude', cols=19, rows=20, edge_color='black')
sphere = create_sphere(rows=20,cols=40,radius=0.1,method='latitude',offset=False)
vertices = sphere.get_vertices(indexed='faces')
normals = sphere.get_vertex_normals()
print(vertices.shape,"vertices")

texcoords = compute_texcoords(vertices)
# texture = np.flipud(texture)
print(texture.shape)
# # Compute UV coordinates for the sphere
# u = 0.5 + np.arctan2(normals[:, 2], normals[:, 0]) / (2 * np.pi)
# v = 0.5 - np.arcsin(normals[:, 1]) / np.pi
# texcoords = np.column_stack([u, v])


mesh_data = MeshData(vertices=vertices)
texture_filter = TextureFilter(texture, texcoords)

sphere_visual = scene.visuals.Mesh(meshdata=mesh_data)
sphere_visual.attach(texture_filter)
view.add(sphere_visual)
# view.add(sphere_object)
view.add(sphere)

vector = np.array([[0, 0, 0], [1, 1, 1]])

# Create an Arrow visual with desired properties
arrow = scene.visuals.Arrow(
    pos=vector,
    color='blue',
    width=3,          # line width in pixels
    arrow_size=20,    # size of the arrowhead
    arrow_type='stealth',  # style of arrow head
    parent=view.scene
)
view.add(arrow)
axes = scene.visuals.XYZAxis(parent=view.scene)



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
