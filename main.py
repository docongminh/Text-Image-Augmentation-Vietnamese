# -*- coding:utf-8 -*-
# Author: RubanSeven

import cv2
import glob
import os
import tqdm
import imageio
from multiprocessing import Process
import random
from augment import distort, stretch, perspective

def check_exits(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_gif(image_list, gif_name, duration=0.1):
    frames = []
    for image in image_list:
        frames.append(image)
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


def augmentation(data_input, output):
    """
        Run augmentor for text augmentation
    """
    lines = open('{}/labels.txt'.format(data_input), 'r').readlines()
    image_dirs = os.path.join(output, 'images')
    check_exits(image_dirs)
    augment = [2, 3, 4, 5]
    with open('{}/labels.txt'.format(output), 'w') as f:
        for line in tqdm.tqdm(lines):
            distort_aug = random.choice(augment)
            stretch_aug = random.choice(augment)
            try:
                filename, label = line.strip().split('\t', 1)
                name = os.path.splitext(filename)[0]
                img_path = os.path.join(image_dirs, filename)
                im = cv2.imread(img_path)
                im = cv2.resize(im, (32*3, 32))

                for i in range(20):
                    distort_img = distort(im, distort_aug)
                    cv2.imwrite('{}/distort_{}_{}.jpg'.format(image_dirs, name, i), distort_img)
                    f.write('distort_{}_{}.jpg'.format(name, i))
                    f.write('\t')
                    f.write(label)
                    f.write('\n')

                    stretch_img = stretch(im, stretch_aug)
                    cv2.imwrite('{}/stretch_{}_{}.jpg'.format(image_dirs, name, i), stretch_img)
                    f.write('stretch_{}_{}.jpg'.format(name, i))
                    f.write('\t')
                    f.write(label)
                    f.write('\n')

                    perspective_img = perspective(im)
                    cv2.imwrite('{}/perspec_{}_{}.jpg'.format(image_dirs, name, i), perspective_img)
                    f.write('perspec_{}_{}.jpg'.format(name, i))
                    f.write('\t')
                    f.write(label)
                    f.write('\n')
            except Exception as e:
                continue

if __name__ == '__main__':

    _input = 'data'
    _output = 'output'
    processes = []
    for i in range(16):
        p = Process(target=augmentation, args=(_input, _output))
        p.daemon = True
        p.start()
        processes.append(p)

    for p in processes:
        p.join()