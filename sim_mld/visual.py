from itertools import combinations
import numpy as np
from PIL import Image

from vispy import app, scene
from vispy.geometry import MeshData, create_sphere
from vispy.visuals.filters import TextureFilter
from vispy.visuals.transforms import MatrixTransform, STTransform
from vispy import gloo

def compute_face_texcoords(vertices: np.ndarray) -> list:
    """
    Compute texture coordinates for a single face of vertices.

    Parameters:
        vertices (np.ndarray): Array of vertex coordinates for one face.

    Returns:
        list: List of [u, v] texture coordinates for the face.
    """
    has_negative = any(
        p[0] < 0 and q[0] < 0 and p[1] * q[1] < 0 for p, q in combinations(vertices, 2)
    )
    face_texcoords = []
    for v in vertices:
        x, y, z = v
        theta = np.arctan2(y, x)
        phi = np.arccos(z / np.linalg.norm(v))
        if has_negative and theta < 0:
            theta += 2 * np.pi
        u = (theta + np.pi) / (2 * np.pi) / 2
        v_coord = phi / np.pi
        face_texcoords.append([u, v_coord])
    return face_texcoords


def compute_texcoords(faces: np.ndarray) -> np.ndarray:
    """
    Compute texture coordinates for all faces.

    Parameters:
        faces (np.ndarray): Array where each element represents a face's vertices.

    Returns:
        np.ndarray: Array of texture coordinates.
    """
    texcoords = []
    for face in faces:
        texcoords.extend(compute_face_texcoords(face))
    return np.array(texcoords)

def setup_visualization(size = (1200, 800), position = (0, 0), view=None, idx = 0) -> dict:
    """
    Set up the Vispy visualization environment including canvas, view, sphere, and markers.

    Returns:
        dict: Dictionary containing references to visualization components.
    """
    if view is None:
        canvas = scene.SceneCanvas(title='Mega-Constellation Simulation',size=size, position=position,
            keys='interactive', show=True, bgcolor=(1.0, 1.0, 1.0, 0))
        view = canvas.central_widget.add_view()
        top_left_coord = 0, 0
        bot_left_coord = 0, size[1]
        top_right_coord = size[0], 0
        bot_right_coord = size[0], size[1]
        view.camera = scene.cameras.TurntableCamera(fov=45, azimuth=0, elevation=45, distance=2.5)
    else:
        top_left_coord = view.pos
        bot_left_coord = view.pos[0], view.pos[1] + view.size[1]
        top_right_coord = view.pos[0] + view.size[0], view.pos[1]
        bot_right_coord = view.pos[0] + view.size[0], view.pos[1] + view.size[1]
    # # Set up the OpenGL state for non-transparent objects.
    # gloo.set_state(depth_test=True, depth_mask=True, blend=False, cull_face=False)

    # Global arrow for reference
    # global_arrow = scene.visuals.Arrow(
    #     pos=np.array([[0, 0, 0], [1, 1, 1]]), color='black', width=3, arrow_size=20,
    #     arrow_type='stealth', parent=view.scene
    # )
    # view.add(global_arrow)

    # Axes with scaling.
    axes = scene.visuals.XYZAxis(parent=view.scene)
    view.add(axes)
    axes.transform = STTransform(scale=(2, 2, 2))

    # Load texture image for the Earth sphere.
    texture_path = "population_density_texture.png"
    try:
        texture_image = Image.open(texture_path)
    except Exception as e:
        print("Error loading texture image: %s" % e)
        raise

    texture = np.array(texture_image)
    # Duplicate texture horizontally.
    texture = np.hstack([texture, texture])
    # Create sphere mesh and compute texture coordinates.
    sphere = create_sphere(rows=20, cols=40, radius=1, method='latitude', offset=False)
    vertices = sphere.get_vertices(indexed='faces')
    texcoords = compute_texcoords(vertices)

    mesh_data = MeshData(vertices=vertices)
    texture_filter = TextureFilter(texture, texcoords)

    sphere_visual = scene.visuals.Mesh(
        meshdata=mesh_data, color=(1.0, 1.0, 1.0, 1.0), shading=None
    )
    sphere_visual.attach(texture_filter)
    sphere_visual.set_gl_state('opaque')
    view.add(sphere_visual)

    # Apply transformation to the sphere.
    sphere_visual.transform = MatrixTransform()

    # Enable depth testing for transparent objects.
    gloo.set_state(depth_test=True, depth_mask=True, blend=True,
               blend_func=('src_alpha', 'one_minus_src_alpha'))
    
    # Create markers for satellite positions.
    scatter = scene.visuals.Markers()
    view.add(scatter)
    
    # Create markers for opional points
    o_scatter = scene.visuals.Markers()
    view.add(o_scatter)

    # Satellite arrows for LT directions.
    satellite_arrow = scene.visuals.Arrow()
    view.add(satellite_arrow)

    # Lines for satellite LISL.
    p_lisl = scene.visuals.Arrow()
    view.add(p_lisl)

    # Lines for connected LISL.
    o_lisl = scene.visuals.Arrow()
    view.add(o_lisl)
    # Create Text visual
    
    WAITING_TEXT = ""
    FONT_SIZE = 10
    text_top_left = scene.visuals.Text(text=WAITING_TEXT,
            face='FreeMono',  # Change font here
            font_size=FONT_SIZE,
            bold=False,
            pos=top_left_coord,
            anchor_x='left',  # horizontal alignment
            anchor_y='bottom',  # vertical alignment
            parent=view.parent,
            )
    
    text_bot_left = scene.visuals.Text(text=WAITING_TEXT,
            color='black',
            face='FreeMono',  # Change font here
            font_size=FONT_SIZE,
            bold=False,
            pos=bot_left_coord,
            anchor_x='left',  # horizontal alignment
            anchor_y='top',  # vertical alignment
            parent=view.parent,
            )
    
    text_top_right = scene.visuals.Text(text=WAITING_TEXT,
            color='black',
            face='FreeMono',  # Change font here
            font_size=FONT_SIZE,
            bold=False,
            pos=top_right_coord,
            anchor_x='right',  # horizontal alignment
            anchor_y='bottom',  # vertical alignment
            parent=view.parent,
            )

    text_bot_right = scene.visuals.Text(text=WAITING_TEXT,
            color='black',
            face='FreeMono',  # Change font here
            font_size=FONT_SIZE,
            bold=False,
            pos=bot_right_coord,
            anchor_x='right',  # horizontal alignment
            anchor_y='top',  # vertical alignment
            parent=view.parent,
            )

    top_middle_coord = (top_left_coord[0] + top_right_coord[0]) / 2, top_left_coord[1]
    text_title = scene.visuals.Text(text="Title-" + str(idx),
            color='black',
            face='FreeMono',  # Change font here
            font_size=FONT_SIZE+5,
            bold=True,
            pos=top_middle_coord,
            anchor_x='center',  # horizontal alignment
            anchor_y='bottom',  # vertical alignment
            parent=view.parent,
            )

    # Position the text at the center
    # text.transform = scene.STTransform(translate=(1, 1))

    return {
        "index": idx,
        "view": view,
        "axes": axes,
        "sphere_visual": sphere_visual,
        "scatter": scatter,
        "o_scatter": o_scatter,
        "arrow": satellite_arrow,
        "p_lisl": p_lisl,
        "o_lisl": o_lisl,
        "text_title": text_title,
        "text_top_left": text_top_left,
        "text_bot_left": text_bot_left,
        "text_top_right": text_top_right,
        "text_bot_right": text_bot_right,
    }

            
def setup_viz_list_one_canvas(sceen_size=(1200, 800), shape=(3, 3)):
    ret = []
    canvas = scene.SceneCanvas(title='Mega-Constellation Simulation',size=sceen_size, position=(0, 0),
            keys='interactive', show=True, bgcolor=(1.0, 1.0, 1.0, 1.0))
    grid = canvas.central_widget.add_grid()
    camera = scene.cameras.TurntableCamera(fov=45, azimuth=0, elevation=45, distance=3)

    for i in range(shape[0]):
        for j in range(shape[1]):
            view = grid.add_view(row=i, col=j)
            view.pos = (j * (sceen_size[0] // shape[1]), i * (sceen_size[1] // shape[0]))
            view.size = (sceen_size[0] // shape[1], sceen_size[1] // shape[0])
            view.camera = camera
            ret.append(setup_visualization(view=view, idx=(i, j)))
    return canvas, ret