"""This script shows another example of using the PyWavefront module."""
# This example was created by intrepid94
import ctypes
import utils
import numpy as np
import os
import scipy.misc


import pyglet
from pyglet.gl import *

from pywavefront import Wavefront


# -------------- PARAMETERS -------------- #
step = 5
output_size = (64, 64)

light = False

obj_path = '../obj/example/earth.obj'
result_folder = '../data_earth_s2_%d_%d_v2/' % (output_size[0], step)


# ---------------------------------------- #

alpha = 0
beta = 0
it = -1
utils.create_dir('frames')

meshes = Wavefront(obj_path)

window = pyglet.window.Window(output_size[0], output_size[1], caption='Demo', resizable=False, vsync=60)

lightfv = ctypes.c_float * 4

frames = []


@window.event
def on_resize(width, height):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(40.0, float(width) / height, 1, 100.0)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_MODELVIEW)
    return True


@window.event
def on_draw():
    global alpha, beta
    window.clear()
    glLoadIdentity()
    if light:
        glLightfv(GL_LIGHT0, GL_POSITION, lightfv(-40, 200, 100, 0.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, lightfv(0.2, 0.2, 0.2, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, lightfv(0.5, 0.5, 0.5, 1.0))
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glMatrixMode(GL_MODELVIEW)
    # Rotations for sphere on axis - useful
    glTranslatef(0, .0, -20)
    glRotatef(beta, 1, 0, 0)
    glRotatef(alpha, 0, 1, 0)
    meshes.draw()

    frame_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
    frame = np.frombuffer(frame_data.data, np.uint8)
    frame = frame.reshape((frame_data.width, frame_data.height, 4))
    frame = frame[:, :, :3]
    frame = np.flip(frame, 0)
    frames.append(frame)
    # scipy.misc.imsave('frames/frame%04d.png' % it, frame)


def update(dt):
    global alpha, beta, it
    it += 1

    if it == 0:
        alpha = 0
        beta = 0
        return

    beta += step
    if beta >= 360:
        alpha += step
        beta = 0
        if alpha >= 360:
            pyglet.app.exit()


pyglet.clock.schedule(update)
pyglet.app.run()


frames = np.array(frames)
print 'Data size:', frames.shape
# utils.create_dir(result_folder)
# np.save(os.path.join(result_folder, 'frames.npy'), frames)
