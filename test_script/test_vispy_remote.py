from vispy import scene, app
import numpy as np

# Create a canvas with interactive keys and display it
canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
view = canvas.central_widget.add_view()

# Generate some random 3D data points
np.random.seed(42)
points = np.random.normal(size=(200, 3))

# Create a scatter plot visual using the generated data
scatter = scene.visuals.Markers()
scatter.set_data(points, edge_color='black', face_color='blue', size=8)
view.add(scatter)

# Set up a turntable camera for an interactive 3D view
view.camera = scene.cameras.TurntableCamera(fov=45, azimuth=30, elevation=30)

# Run the VisPy application
if __name__ == '__main__':
    app.run()