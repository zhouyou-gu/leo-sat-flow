import numpy as np
from vispy import app, scene

N = 300_000                          # 300 k triangles
vertices = np.random.uniform(-1, 1,  (N*3, 3)).astype(np.float32)
faces    = np.arange(N*3, dtype=np.uint32).reshape(N, 3)

# optional per-triangle colors (repeat each color 3× so every vertex gets it)
colors   = np.repeat(np.random.uniform(0, 1, (N, 4)).astype(np.float32), 3, axis=0)

canvas = scene.SceneCanvas(bgcolor='black', keys='interactive', show=True)
view   = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(distance=4)

scene.visuals.Mesh(vertices=vertices,
                   faces=faces,
                   vertex_colors=colors,
                   shading=None,            # cheaper than 'flat' or 'smooth'
                   parent=view.scene)

if __name__ == '__main__':
    app.run()