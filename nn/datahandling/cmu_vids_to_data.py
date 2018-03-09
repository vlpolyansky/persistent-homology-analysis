import os
import sys
import subprocess as sp
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

from utils import create_dir

# -------------- PARAMETERS -------------- #
videos_folder = '../cmu_videos/'
data_folder = '../data_cmu/'

frame_mod = 3
input_size = (240, 320, 3)
output_size = (128, 128)


# ---------------------------------------- #


def get_pipe(path):
    command = ['ffmpeg',
               '-i', path,
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vcodec', 'rawvideo', '-']
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10 ** 8)
    return pipe


def next_frame(pipe, frame_size):
    length = frame_size[0] * frame_size[1] * frame_size[2]
    raw_image = pipe.stdout.read(length)
    if not raw_image:
        return None
    image = np.fromstring(raw_image, dtype='uint8')
    image = image.reshape(frame_size)
    pipe.stdout.flush()
    return image


def main():
    create_dir(data_folder)

    frames = []

    for fname in os.listdir(videos_folder):
        file_path = os.path.join(videos_folder, fname)
        print 'Loading', file_path

        pipe = get_pipe(file_path)
        i = 0
        while True:
            # if i % frame_mod == 0:
            #     print 'Current frame: %i' % i
            image = next_frame(pipe, input_size)
            if image is None:
                break
            if i % frame_mod == 0:
                resized = Image.fromarray(image).resize(output_size, Image.ANTIALIAS)
                # image_path = os.path.join(data_folder, str(i / frame_mod) + '.png')
                resized = np.array(resized)
                frames.append(resized)
            i += 1

    frames = np.array(frames)
    print 'Dataset size:', frames.shape
    np.save(os.path.join(data_folder, 'frames.npy'), frames)

if __name__ == '__main__':
    main()
