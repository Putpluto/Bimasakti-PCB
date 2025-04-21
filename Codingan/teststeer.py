from PIL import Image
import numpy as np
import OpenGL.GL as gl
from imgui_bundle import imgui, immapp

import glfw
from OpenGL.GL import *

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW could not be initialized!")

# Create a window and OpenGL context
window = glfw.create_window(800, 600, "OpenGL Context", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window could not be created!")

# Make the OpenGL context current
glfw.make_context_current(window)

slider_value = [0]
def load_texture(image_path):
    # Load image using PIL
    image = Image.open(image_path)
    image = image.convert('RGBA')
    image_data = np.array(image)
    
    # Generate a texture ID
    texture_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

    # Set texture parameters
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

    # Upload texture data
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D,
        0,
        gl.GL_RGBA,
        image.width,
        image.height,
        0,
        gl.GL_RGBA,
        gl.GL_UNSIGNED_BYTE,
        image_data
    )

    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return texture_id, image.width, image.height

def rotate_image(image_path, angle):
    # Load and rotate image
    image = Image.open(image_path)
    rotated_image = image.rotate(angle, expand=True)
    return rotated_image

def main():
    image_path = "Steer.png"
    angle = slider_value[0]  # Example rotation angle

    # Rotate image and load as texture
    rotated_image = rotate_image(image_path, angle)
    rotated_image.save("rotated_steer.png")  # Save rotated image temporarily
    texture_id, width, height = load_texture("rotated_steer.png")

    def gui():
        changed, slider_value[0] = imgui.slider_int(
        "Index", slider_value[0], 0, 4000
        )
        imgui.begin("Steering Wheel Visualization")
        imgui.text("Rotated Steering Wheel:")
        imgui.image(texture_id, width, height)
        imgui.end()

    immapp.run(gui, with_implot=False)

if __name__ == "__main__":
    main()
