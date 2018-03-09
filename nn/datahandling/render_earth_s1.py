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
step = 1
output_size = (128, 128)

light = False

obj_path = '../obj/example/earth.obj'
result_folder = '../data_earth_nolight_%d_%.3f/' % (output_size[0], step)


# ---------------------------------------- #

rotation = 0
it = -1
utils.create_dir('frames')

meshes = Wavefront(obj_path)

window = pyglet.window.Window(output_size[0], output_size[1], caption='Demo', resizable=False, vsync=0)

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
    global rotation
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
    # glTranslated(0, 4, -8)
    #    glRotatef(90, 0, 1, 0)
    #    glRotatef(-60, 0, 0, 1)
    # Rotations for sphere on axis - useful
    glTranslated(0, .8, -20)
    glRotatef(-66.5, 0, 0, 1)
    glRotatef(rotation, 1, 0, 0)
    glRotatef(90, 0, 0, 1)
    glRotatef(0, 0, 1, 0)
    meshes.draw()

    frame_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
    frame = np.frombuffer(frame_data.data, np.uint8)
    frame = frame.reshape((frame_data.width, frame_data.height, 4))
    frame = frame[:, :, :3]
    frame = np.flip(frame, 0)
    frames.append(frame)
    # scipy.misc.imsave('frames/frame%04d.png' % it, frame)


def update(dt):
    global rotation, it
    rotation += step
    it += 1
    if rotation >= 360:
        pyglet.app.exit()


pyglet.clock.schedule(update)
pyglet.app.run()


frames = np.array(frames)
print 'Data size:', frames.shape
utils.create_dir(result_folder)
np.save(os.path.join(result_folder, 'frames.npy'), frames)
