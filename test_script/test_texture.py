#!/usr/bin/env python3
"""
Debugging Texture Mapping in VPython

This script performs the following:
  1. Checks if the texture file ("simple_earth_texture.png") exists.
  2. Opens the texture using Pillow and prints its dimensions and aspect ratio.
  3. Optionally displays the image with Matplotlib for visual inspection.
  4. Creates a VPython sphere using the texture and runs a simple interactive scene.
  
If the texture image is not 2:1 in aspect ratio, VPython’s mapping onto a sphere may appear distorted.
"""

import os
from PIL import Image
import matplotlib.pyplot as plt
from vpython import sphere, vector, scene, color, rate

def debug_texture_file(filename):
    """
    Check and display properties of the texture file.
    """
    if not os.path.exists(filename):
        print(f"Error: Texture file '{filename}' not found!")
        return False
    try:
        img = Image.open(filename)
        width, height = img.size
        aspect_ratio = width / height
        print(f"Texture file '{filename}' found.")
        print(f"Image size: {width}x{height} (aspect ratio: {aspect_ratio:.2f})")
        
        # Optionally display the image using matplotlib.
        plt.imshow(img)
        plt.title("Debug: Texture Image")
        plt.axis('off')
        plt.show()
        return True
    except Exception as e:
        print(f"Error opening texture file: {e}")
        return False

def run_vpython_simulation(texture_file):
    """
    Set up a simple VPython scene with an Earth sphere using the specified texture.
    """
    scene.width = 800
    scene.height = 600
    scene.background = color.black
    
    # Create the Earth sphere with the texture.
    earth = sphere(pos=vector(0, 0, 0), radius=1,
                   texture=texture_file,
                   shininess=0.8)
    print("VPython Earth sphere created with texture:", texture_file)
    
    # Run an infinite loop to keep the VPython window interactive.
    while True:
        rate(30)  # 30 frames per second

if __name__ == "__main__":
    texture_filename = "simple_earth_texture.png"
    # Debug the texture file: check existence and properties.
    if debug_texture_file(texture_filename):
        print("Launching VPython simulation...")
        run_vpython_simulation(texture_filename)
    else:
        print("Texture debugging failed. Please check your texture generation process.")
