#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from config import num_joints, dataset


# 高斯生成
def getGaussianMap(joint=(16, 16), heat_size=128, sigma=2):
    # by default, the function returns a gaussian map with range [0, 1] of typr float32
    heatmap = np.zeros((heat_size, heat_size), dtype=np.float32)
    tmp_size = sigma * 3
    ul = [int(joint[0] - tmp_size), int(joint[1] - tmp_size)]
    br = [int(joint[0] + tmp_size + 1), int(joint[1] + tmp_size + 1)]
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma ** 2)))
    g.shape

    g_x = max(0, -ul[0]), min(br[0], heat_size) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], heat_size) - ul[1]
    # image range
    img_x = max(0, ul[0]), min(br[0], heat_size)
    img_y = max(0, ul[1]), min(br[1], heat_size)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return heatmap


# read annotations
annotations = loadmat("./dataset/" + dataset + "/joints.mat")
if dataset == "lsp":
    # LSP
    number_images = 2000
    label = annotations["joints"].swapaxes(0, 2)  # shape (3, 14, 2000) -> (2000, 14, 3)
else:
    # LSPET
    number_images = 10000
    label = annotations["joints"].swapaxes(0, 1)  # shape (14, 3, 10000) -> (3, 14, 10000)
    label = label.swapaxes(0, 2)  # shape (3, 14, 10000) -> (10000, 14, 3)

# read images
data = np.zeros([number_images, 256, 256, 3])
heatmap_set = np.zeros((number_images, 128, 128, num_joints), dtype=np.float32)
print("Reading dataset...")
for i in range(number_images):
    if dataset == "lsp":
        # lsp
        FileName = "./dataset/" + dataset + "/images/im%04d.jpg" % (i + 1)
    else:
        # lspet
        FileName = "./dataset/" + dataset + "/images/im%05d.jpg" % (i + 1)
    img = tf.io.read_file(FileName)
    img = tf.image.decode_image(img)
    img_shape = img.shape
    # Attention here img_shape[0] is height and [1] is width
    label[i, :, 0] *= (256 / img_shape[1])
    label[i, :, 1] *= (256 / img_shape[0])
    data[i] = tf.image.resize(img, [256, 256])
    # generate heatmap set
    for j in range(num_joints):
        _joint = (label[i, j, 0:2] // 2).astype(np.uint16)
        heatmap_set[i, :, :, j] = getGaussianMap(joint=_joint, heat_size=128, sigma=4)
    # print status
    if not i % (number_images // 80):
        print(">", end='')

coordinates = label[:, :, 0:2]
visibility = label[:, :, 2:]

print("Done.")
