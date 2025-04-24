import simpy
import numpy as np
from vispy import app, scene
from vispy.scene import SceneCanvas
from vispy.visuals import SphereVisual, MeshVisual
from vispy.geometry import MeshData, create_sphere
from vispy.visuals.transforms import MatrixTransform, STTransform

import time
import cProfile
import pstats
from PIL import Image
from vispy.visuals.filters import TextureFilter
from itertools import combinations

# ----- Visualization Code (Vispy) -----
# Create a Vispy SceneCanvas.
canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor=(1.0, 1.0, 1.0, 0))
view = canvas.central_widget.add_view()

# Set a 3D camera with a turntable view
view.camera = scene.cameras.TurntableCamera(fov=45, azimuth=0, elevation=45, distance=2)


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
print(texture.shape)
texture = np.hstack([texture, texture])

sphere = create_sphere(rows=20,cols=40,radius=1,method='latitude',offset=False)
vertices = sphere.get_vertices(indexed='faces')
texcoords = compute_texcoords(vertices)

mesh_data = MeshData(vertices=vertices)
texture_filter = TextureFilter(texture, texcoords)

sphere_visual = scene.visuals.Mesh(meshdata=mesh_data,color=(1.0, 1.0, 1.0, 1.0),shading=None)
sphere_visual.attach(texture_filter)
# sphere_visual.set_gl_state('translucent',cull_face=True, blend=True, depth_test=False,
                    # blend_func=('src_alpha', 'one_minus_src_alpha'))
view.add(sphere_visual)

# sphere_object = scene.visuals.Sphere(radius=1, method='latitude', cols=19, rows=20, edge_color='black')
# view.add(sphere_object)

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
view.add(axes)
# Define the scaling factors for each axis
scale_factors = (2, 2, 2)  # For example, double the length of each axis
# Apply the scaling transformation
axes.transform = STTransform(scale=scale_factors)


# Variables for profiling updates.
update_count = 0
accumulated_update_time = 0

# Define an update function called by a timer.

transform = MatrixTransform()
sphere_visual.transform = transform

angle = 0
def update(event):
    global angle
    global update_count, accumulated_update_time
    start_time = time.perf_counter()
    sphere_visual.update()

    try:
        pass
        angle += 1  # You can adjust the increment for speed.
        # Reset the transformation matrix to clear previous transformations.
        sphere_visual.transform.reset()
        # Apply a new rotation around the y-axis.
        sphere_visual.transform.rotate(angle, (0, 0, 1))
    except simpy.core.EmptySchedule:
        pass

    # Update the 3D scatter plot with the current positions.
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