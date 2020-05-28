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
from imgaug import augmenters as iaa
import imgaug as ia
import PIL
from PIL import Image
import numpy as np
from natsort import natsorted

def check_exits(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_gif(image_list, gif_name, duration=0.1):
    frames = []
    for image in image_list:
        frames.append(image)
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
sometimes2 = lambda aug: iaa.Sometimes(0.1, aug)

class ImgAugTransform:
    
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Resize({"height":32,"width": "keep-aspect-ratio"}),
            sometimes(iaa.Crop(percent=(0, 0.02))),
    #         sometimes(iaa.AddToHueAndSaturation(value=(-20, 20), per_channel=True)),
    # #         #iaa.Scale((94, 24)),
    #         sometimes((iaa.Add((-20, 20), per_channel=0.5))),
    #         sometimes(iaa.OneOf([
    #             iaa.LinearContrast((0.8, 1.2), per_channel=0.5),
    #             iaa.Multiply((0.8,1.2),per_channel=0.5),
    #         ])),
    #         sometimes((iaa.Add((-10, 10), per_channel=0.5))),
            sometimes(iaa.OneOf([
            iaa.GaussianBlur((0, 2.0)),
            iaa.AverageBlur(k=(1, 3)),
            iaa.MedianBlur(k=(1, 3)),
            ])),

        sometimes(iaa.ElasticTransformation(alpha=(1, 2), sigma=0.25)),
#         sometimes(iaa.Dropout((0.005, 0.01), per_channel=0.5)), # randomly remove up to 10% of the pixels
#         iaa.PerspectiveTransform(scale=(0.02, 0.02)),
#         iaa.Sometimes(0.9,iaa.Affine(
#             scale={"x": (0.95, 1), "y": (0.95, 1)}, # scale images to 80-120% of their size, individually per axis
#             translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, # translate by -20 to +20 percent (per axis)
#             rotate=(-8, 8), # rotat032E02B4-0499-0531-2C06-5B0700080009e by -45 to +45 degrees
#             order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
#             cval=(0, 255), # if mode is constant, use a cval between 0 and 255
#             mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
#         )),
#         iaa.Clouds(),
#         iaa.OneOf([sometimes2(iaa.CloudLayer(intensity_mean=(160,240),intensity_freq_exponent=(-2.5,-1.5),intensity_coarse_scale = (0,5),alpha_min = 0.0,alpha_multiplier =(0.4,0.8),alpha_size_px_max=16,alpha_freq_exponent=-2,sparsity=1,density_multiplier=1,)),
#         iaa.Resize({"height":24,'width':96}),
#         iaa.PadToFixedSize(width=96,height=32,position='right-center',),
        iaa.Resize({"height":32,'width':90}),
                iaa.Invert(0.2, per_channel=False), # invert color channels
                  # ]),
        sometimes(iaa.Crop(percent=(0,0.04),keep_size=False)),
        iaa.Resize({"height":32,"width": "keep-aspect-ratio"},interpolation=ia.ALL),
        sometimes(iaa.Sequential([
                iaa.Resize({"height":24,"width": "keep-aspect-ratio"},interpolation=ia.ALL),
                iaa.Resize({"height":32,"width": "keep-aspect-ratio"},interpolation=ia.ALL),
        ])),
    ])
      
    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


def augment_useimgaug(_input, _output):
    """

    """
    image_dirs = os.path.join(_output, 'images')
    imgs = os.path.join(_input, 'images')
    check_exits(image_dirs)

    transforms = ImgAugTransform()
    lines = open('{}/labels.txt'.format(_input)).readlines()
    with open('{}/labels.txt'.format(_output), 'w') as f:
        for line in tqdm.tqdm(lines):
            try:
                filename, label = line.strip().split('\t', 1)
                img_goc = cv2.imread(os.path.join(imgs, filename))
                for i in range(20):
                    img = transforms(img_goc)
                    cv2.imwrite(image_dirs + '/'  + str(i) + '_' + filename,img)
                    f.write(str(i) + '_' + filename)
                    f.write('\t')
                    f.write(label)
                    f.write('\n')
            except Exception as e:
                continue


def augmentation(data_input, output):
    """
        Run augmentor for text augmentation
    """
    lines = open('{}/labels.txt'.format(data_input), 'r').readlines()
    image_dirs = os.path.join(output, 'images')
    imgs = os.path.join(data_input, 'images')
    check_exits(image_dirs)
    augment = [2, 3, 4, 5]
    with open('{}/labels.txt'.format(output), 'w') as f:
        for line in tqdm.tqdm(lines):
            distort_aug = random.choice(augment)
            stretch_aug = random.choice(augment)
            try:
                filename, label = line.strip().split('\t', 1)
                name = os.path.splitext(filename)[0]
                img_path = os.path.join(imgs, filename)
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

    _input = 'data/train'
    _output = 'output'
    use_imgaug = True
    if use_imgaug:
        target = augment_useimgaug
    else:
        target = augmentation
    processes = []
    for i in range(16):
        p = Process(target=target, args=(_input, _output))
        p.daemon = True
        p.start()
        processes.append(p)

    for p in processes:
        p.join()