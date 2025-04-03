import vispy
# Set the backend to jupyter_rfb before creating any canvas
vispy.use('jupyter_rfb')
from vispy import scene

# Create a SceneCanvas that will render within the notebook
canvas = scene.SceneCanvas(keys='interactive', bgcolor='black', size=(800, 600), show=True)

# (Add your visuals here...)
canvas