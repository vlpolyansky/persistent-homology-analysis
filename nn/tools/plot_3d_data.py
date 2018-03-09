import numpy as np
import os
import phom.pyqtgraph as pg
from phom.pyqtgraph.Qt import QtCore, QtGui, QtOpenGL
import phom.pyqtgraph.opengl as gl
from time import time
import subprocess
import shutil
import json

# -------------- PARAMETERS -------------- #
#results_folder = '../results/results_mnist_blur0_mxdenc/'
#run_id = 3
#label_id = 'all'
#train_test = 'train'
#record = False
#load_decoder = False

#prefix = 'dens_100_'
# prefix = ''
# ---------------------------------------- #

# -------------- PARAMETERS -------------- #
results_folder = '../results/results_all_rotation_3d/'
run_id = 1
label_id = 'all'
train_test = 'train'
record = False
load_decoder = False

# prefix = 'density_'
prefix = ''
# ---------------------------------------- #

# # # -------------- PARAMETERS -------------- #
# results_folder = '../results/results_label6_rotation/'
# run_id = 1
# label_id = 6
# train_test = 'test'
# record = False
# load_decoder = False
# prefix = 'density_'
# #prefix = ''
# # ---------------------------------------- #


def norm_images(images):
    images = ((images + 0.5) * 255)
    images = np.maximum(0, images)
    images = np.minimum(255, images)
    return images.astype(np.uint8)


def edges_to_curve(data, edges):
    nxt = dict()
    for edge in edges:
        if edge[0] not in nxt.keys():
            nxt[edge[0]] = []
        nxt[edge[0]].append(edge[1])
        if edge[1] not in nxt.keys():
            nxt[edge[1]] = []
        nxt[edge[1]].append(edge[0])

    prev = edges[0, 0]
    cur = edges[0, 1]
    first = cur

    curve = np.zeros((edges.shape[0] + 1, data.shape[1]))
    curve[0] = data[prev]

    for i in range(1, curve.shape[0]):
        curve[i] = data[cur]
        if nxt[cur][0] != prev:
            prev = cur
            cur = nxt[cur][0]
        else:
            prev = cur
            cur = nxt[cur][1]

    return curve


class MyView(gl.GLViewWidget):
    def __init__(self, data, filtered_ids, images, record, decode):
        super(MyView, self).__init__()

        self.record = record
        self.frames = []
        self.durations = []
        self.last_frame_time = None

        self.decode = decode

        self.mousePos = None
        self.setMouseTracking(True)

        self.show_filtered = False
        self.data = data
        self.filtered_ids = filtered_ids
        self.images = images
        self.colors = np.repeat([[.5, .5, 1, 1]], data.shape[0], axis=0)
        self.data_filtered = data[filtered_ids]
        self.colors_filtered = np.repeat([[.5, .5, 1, 1]], self.data_filtered.shape[0], axis=0)

        self.images_dec = None
        self.show_decoded = False
        if decode is not None:
            batch = 32
            self.images_dec = []
            for i in range(0, data.shape[0], batch):
                self.images_dec.append(decode(data[i:i + batch]))
            self.images_dec = np.concatenate(self.images_dec)
            self.images_dec = norm_images(self.images_dec)

        self.active_cycle = 0
        self.cycles = []
        self.cycle_vertices = []
        self.selected_cycle = -1
        self.show_killer = False
        self.killers = []
        self.killer_images = []

        self.curMousePos = None
        self.closestPoint = None

        self.projected_array = None
        self._update_projected_array()
        self.selected_idx = -1

        self.setWindowTitle('Data plot')

        self._init_grid()

        self.sp = gl.GLScatterPlotItem(pos=data, size=1, color=self.colors, pxMode=True)
        self.sp_f = gl.GLScatterPlotItem(pos=self.data_filtered, size=1, color=self.colors_filtered, pxMode=True)
        self.sp_f.setVisible(False)
        self.sp2 = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), size=6, color=(1, 0, 0, 1.), pxMode=True)
        self.sp2.setGLOptions('opaque')

        self.addItem(self.sp)
        self.addItem(self.sp_f)
        self.addItem(self.sp2)

    def mouseMoveEvent(self, ev):
        self.curMousePos = ev.pos()
        if self.mousePos is None:
            self.mousePos = ev.pos()
        super(MyView, self).mouseMoveEvent(ev)
        if not ev.buttons():
            self._update_selected()
        else:
            self.projected_array = None

    def mouseReleaseEvent(self, ev):
        super(MyView, self).mouseReleaseEvent(ev)
        self._update_projected_array()
        self.update()

    def mousePressEvent(self, ev):
        self.selected_idx = -1
        super(MyView, self).mouseReleaseEvent(ev)

    def wheelEvent(self, ev):
        self.selected_idx = -1
        super(MyView, self).wheelEvent(ev)
        self._update_projected_array()

    def keyPressEvent(self, ev):
        gl.GLViewWidget.keyPressEvent(self, ev)
        key = ev.key() - QtCore.Qt.Key_0
        if 0 <= key <= len(self.cycle_vertices):
            self.active_cycle = key - 1
            self._activate_cycle(key - 1)
        elif ev.key() == QtCore.Qt.Key_F:
            self.show_filtered = not self.show_filtered
            self.sp.setVisible(not self.show_filtered)
            self.sp_f.setVisible(self.show_filtered)
        elif ev.key() == QtCore.Qt.Key_K:
            self.show_killer = not self.show_killer
            self._activate_cycle(self.active_cycle)
        elif ev.key() == QtCore.Qt.Key_D:
            self.show_decoded = not self.show_decoded
            self.repaint()

    def _activate_cycle(self, k):
        if k >= 0:
            for cycle in self.cycles:
                for lp in cycle:
                    lp.hide()
            for killer in self.killers:
                if killer is not None:
                    killer.hide()
            for lp in self.cycles[k]:
                lp.show()
            if self.show_killer and self.killers[k] is not None:
                self.killers[k].show()
        else:
            for killer in self.killers:
                if killer is not None:
                    killer.hide()
            for cycle in self.cycles:
                for lp in cycle:
                    lp.show()
        self.selected_cycle = k

    def _init_grid(self):
        spacing = .2
        size = 2
        color = (.1, .1, .1, 1)

        g = gl.GLGridItem(color=color)
        g.setSpacing(spacing, spacing, spacing)
        g.setSize(size, size, size)
        g.translate(0, 0, -size / 2)
        self.addItem(g)

        g = gl.GLGridItem(color=color)
        g.setSpacing(spacing, spacing, spacing)
        g.setSize(size, size, size)
        g.rotate(90, 1, 0, 0)
        g.translate(0, -size / 2, 0)
        self.addItem(g)

        g = gl.GLGridItem(color=color)
        g.setSpacing(spacing, spacing, spacing)
        g.setSize(size, size, size)
        g.rotate(90, 0, 1, 0)
        g.translate(-size / 2, 0, 0)
        self.addItem(g)

    def _update_projected_array(self):
        m = self.projectionMatrix() * self.viewMatrix()
        self._m = m
        self.projected_array = np.zeros((self.data.shape[0], 2))
        for i in range(data.shape[0]):
            pt = m.map(QtGui.QVector3D(self.data[i, 0],
                                       self.data[i, 1],
                                       self.data[i, 2]))
            # origin range [-1, 1]
            self.projected_array[i, 0] = (pt.x() + 1) / 2
            self.projected_array[i, 1] = (- pt.y() + 1) / 2

    def _update_selected(self):
        view_w = self.width()
        view_h = self.height()
        mouse_x = self.curMousePos.x() * 1.0
        mouse_y = self.curMousePos.y() * 1.0

        distance_array = np.power(np.power(self.projected_array[:, 0] - mouse_x / view_w, 2) +
                                  np.power(self.projected_array[:, 1] - mouse_y / view_h, 2), 0.5)

        if self.selected_idx != -1:
            self.colors[self.selected_idx, 3] = 1
        self.selected_idx = np.nanargmin(distance_array)
        if distance_array[self.selected_idx] * view_w > 5:
            self.selected_idx = -1

        if self.selected_idx == -1:
            self.sp2.setData(pos=np.zeros((0, 3)))
        else:
            self.sp2.setData(pos=self.data[self.selected_idx: self.selected_idx + 1, :])
            self.colors[self.selected_idx, 3] = 0

    def plot_cycle(self, filename, killer_filename, shift=0.0, color=(0, 0, 1, 1)):
        edges = np.loadtxt(filename, dtype=int, skiprows=1)
        if edges.shape[1] == 2:
            self.cycle_vertices.append(np.unique(edges.flatten()))
            curve = edges_to_curve(self.data, edges)
            lp = gl.GLLinePlotItem(pos=curve, color=color, width=1.5)
            lp.setGLOptions('translucent')
            cycle = [lp]
            self.addItem(lp)
            # cycle = []
            # for i in range(edges.shape[0]):
            #     v1 = edges[i, 0]
            #     v2 = edges[i, 1]
            #     points = np.vstack((self.data[v1], self.data[v2])) + shift
            #     lp = gl.GLLinePlotItem(pos=points, color=color, width=1.5)
            #     lp.setGLOptions('translucent')
            #     cycle.append(lp)
            #     self.addItem(lp)
            self.cycles.append(cycle)

            if os.path.exists(killer_filename):
                face = np.loadtxt(killer_filename, dtype=int, skiprows=1)
                points = np.vstack((self.data[face[0]], self.data[face[1]], self.data[face[2]])) + shift
                mesh_data = gl.MeshData(points, np.array([[0, 1, 2]]))
                mesh_item = gl.GLMeshItem(meshdata=mesh_data, color=color)
                mesh_item.setGLOptions('translucent')
                mesh_item.hide()
                self.killers.append(mesh_item)
                self.addItem(mesh_item)

                if self.decode is not None:
                    center_point = (self.data[face[0]] + self.data[face[1]] + self.data[face[2]]) / 3.0
                    image = self.decode(np.array([center_point]))
                    self.killer_images.append(norm_images(image)[0])
                else:
                    self.killer_images.append(None)

            else:
                self.killers.append(None)
                self.killer_images.append(None)
        elif edges.shape[1] == 3:
            faces = edges.astype(np.int64)
            mesh_data = gl.MeshData(self.data, faces)
            surface = gl.GLMeshItem(meshdata=mesh_data, edgeColor=color, color=(color[0], color[1], color[2], 0.1),
                                    drawEdges=True, drawFaces=True)
            surface.setGLOptions('additive')
            self.addItem(surface)
            self.cycles.append([surface])
            self.cycle_vertices.append([])
            self.killers.append(None)
            self.killer_images.append(None)

        else:
            print str(edges.shape[1] - 1) + "-dim cycles are not supported"

    def paintEvent(self, ev):
        images = self.images if not self.show_decoded or self.images_dec is None else self.images_dec
        qp = QtGui.QPainter()
        qp.begin(self)
        self.paintGL()
        qp.setPen(QtGui.QPen(QtCore.Qt.red))
        fmt = QtGui.QImage.Format_RGB32 if len(images[0].shape) == 3 and images[0].shape[2] == 3 else \
            QtGui.QImage.Format_Indexed8
        if self.selected_cycle != -1 and self.projected_array is not None:
            for idx in self.cycle_vertices[self.selected_cycle]:
                w = images[idx].shape[0]
                qp.drawImage(self.projected_array[idx, 0] * self.width() + 2,
                             self.projected_array[idx, 1] * self.height() - w + 2,
                             QtGui.QImage(images[idx].data, w, w, fmt))
            if self.show_killer and self.killer_images[self.selected_cycle] is not None:
                image = self.killer_images[self.selected_cycle]
                w = image.shape[0]
                pt = self.killers[self.selected_cycle].opts['meshdata'].vertexes()
                pt = (pt[0] + pt[1] + pt[2]) / 3.0
                pt = self._m.map(QtGui.QVector3D(pt[0],
                                                 pt[1],
                                                 pt[2]))
                qp.drawImage((pt.x() + 1) / 2 * self.width() - w * 0.5 + 2,
                             (- pt.y() + 1) / 2 * self.height() - w * 0.5 + 2,
                             QtGui.QImage(image.data, w, w, fmt))

        if self.selected_idx != -1:
            w = images[self.selected_idx].shape[0]
            qp.drawImage(self.curMousePos.x() + 2, self.curMousePos.y() - w + 2,
                         QtGui.QImage(images[self.selected_idx].data, w, w, fmt))
        qp.end()

        if self.record:
            t = time()
            if self.last_frame_time is None or t - self.last_frame_time > 0.05:
                self.frames.append(self.grabFrameBuffer())
                if self.last_frame_time is not None:
                    self.durations.append(t - self.last_frame_time)
                self.last_frame_time = t

    def export_video(self):
        if not os.path.exists('_frames'):
            os.mkdir('_frames')
        with open('_frames.txt', 'w') as f:
            for i in range(1, len(self.durations)):
                f.write('file \'_frames/frame_%d.png\'\nduration %.4f\n' % (i, self.durations[i]))
                self.frames[i].save('_frames/frame_%d.png' % i)
        subprocess.call(['ffmpeg', '-y', '-f', 'concat', '-i', '_frames.txt', 'video.gif'])
        os.remove('_frames.txt')
        shutil.rmtree('_frames')


app = QtGui.QApplication([])

label_folder = '%srun%d/%slabel%s%s/' % (results_folder, run_id, prefix, str(label_id), train_test)
# label_folder = './'

data = np.loadtxt(label_folder + 'data.txt', skiprows=1)
filtered_ids = np.array([0])
if os.path.exists(label_folder + 'filtered_ids.txt'):
    filtered_ids = np.loadtxt(label_folder + 'filtered_ids.txt', dtype=np.int32)

images = np.load('%s_data/%s_images.npy' % (results_folder, train_test))
images = norm_images(images)
labels = np.load('%s_data/%s_labels.npy' % (results_folder, train_test))
if label_id != 'all':
    images = images[labels == label_id, :, :]
assert data.shape[0] == images.shape[0]

decode = None
if load_decoder:
    import main, algo, encoders
    import matplotlib.pyplot as plt

    with open(results_folder + 'params.json') as data_file:
        params = json.load(data_file)
    params['save_dir'] = '%srun%d/%s' % (results_folder, run_id, params['save_dir'])
    params['logs_dir'] = '%srun%d/%s' % (results_folder, run_id, params['logs_dir'])
    kwargs = main.extract_kwargs(params)
    encoder = getattr(encoders, params['algo'])
    decode = algo.decoder_wrapper(encoder, params, **kwargs)
    # image = decode(np.array([[0, 0, 0]]))
    # plt.imshow(image[0])
    # plt.show()


w = MyView(data, filtered_ids, images, record, decode)
w.show()

sh = min(np.max(data[:, 0]) - np.min(data[:, 0]),
         np.max(data[:, 1]) - np.min(data[:, 1]),
         np.max(data[:, 2]) - np.min(data[:, 2])) * 0.002
shifts = [0, sh, -sh, 2 * sh, -2 * sh]
# shifts = [0, 0, 0, 0, 0]
colors = [(1, 0, 0, .75), (0, 1, 0, .75), (1, 0, 1, .75), (1, 1, 0, .75), (0, 1, 1, .75)]
# cycles = [label_folder + 'repr_0.txt',
#           label_folder + 'repr_1.txt',
#           label_folder + 'repr_2.txt',
#           label_folder + 'repr_3.txt',
#           label_folder + 'repr_4.txt']
# killers = [label_folder + 'kill_0.txt',
#           label_folder + 'kill_1.txt',
#           label_folder + 'kill_2.txt',
#           label_folder + 'kill_3.txt',
#           label_folder + 'kill_4.txt']
# cycles = [cycles[0]]

for i in range(5):
    cycle_fname = label_folder + 'repr_%d.txt' % i
    killer_fname = label_folder + 'kill_%d.txt' % i
    if os.path.exists(cycle_fname):
        w.plot_cycle(cycle_fname, killer_fname, shifts[i], colors[i])


QtGui.QApplication.instance().exec_()
if record:
    w.export_video()
