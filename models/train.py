#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2022/07/31 16:20:58
@Author  :   ykzhou 
@Version :   0.0
@Contact :   ykzhou@stu.xidian.edu.cn
@Desc    :   None
'''

import os
import pathlib
import tensorflow as tf
from model import BlazePose
from config import total_epoch, train_mode, best_pre_train, continue_train, batch_size, dataset
from models.data import coordinates, visibility, heatmap_set, data, number_images

checkpoint_path_heatmap = "checkpoints_heatmap"
checkpoint_path_regression = "checkpoints_regression"
loss_func_mse = tf.keras.losses.MeanSquaredError()
loss_func_bce = tf.keras.losses.BinaryCrossentropy()

model = BlazePose().call()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer, loss=[loss_func_bce, loss_func_mse, loss_func_bce])

if train_mode:
    checkpoint_path = checkpoint_path_regression
else:
    checkpoint_path = checkpoint_path_heatmap
pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

# Define the callbacks
model_folder_path = os.path.join(checkpoint_path, "models")
pathlib.Path(model_folder_path).mkdir(parents=True, exist_ok=True)
mc = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(
    model_folder_path, "model_ep{epoch:03d}.h5"), save_freq=5, save_weights_only=True, save_format="h5", verbose=1)

# continue train
if continue_train > 0:
    print("Load heatmap weights", os.path.join(checkpoint_path, "models/model_ep{}.h5".format(continue_train)))
    model.load_weights(os.path.join(checkpoint_path, "models/model_ep{}.h5".format(continue_train)))
else:
    if train_mode:
        print("Load heatmap weights",
              os.path.join(checkpoint_path_heatmap, "models/model_ep{}.h5".format(best_pre_train)))
        model.load_weights(os.path.join(checkpoint_path_heatmap, "models/model_ep{}.h5".format(best_pre_train)))

if train_mode:
    print("Freeze these layers:")
    for layer in model.layers:
        if not layer.name.startswith("regression"):
            print(layer.name)
            layer.trainable = False
# 冻结, 停止梯度回传
else:
    print("Freeze these layers:")
    for layer in model.layers:
        if layer.name.startswith("regression"):
            print(layer.name)
            layer.trainable = False

if dataset == "lsp":
    x_train = data[:(number_images - 400)]
    y_train = [heatmap_set[:(number_images - 400)], coordinates[:(number_images - 400)],
               visibility[:(number_images - 400)]]

    x_val = data[-400:-200]
    y_val = [heatmap_set[-400:-200], coordinates[-400:-200], visibility[-400:-200]]
else:
    x_train = data[:(number_images - 2000)]
    y_train = [heatmap_set[:(number_images - 2000)], coordinates[:(number_images - 2000)],
               visibility[:(number_images - 2000)]]

    x_val = data[-2000:-1000]
    y_val = [heatmap_set[-2000:-1000], coordinates[-2000:-1000], visibility[-2000:-1000]]

model.fit(x=x_train, y=y_train,
          batch_size=batch_size,
          epochs=total_epoch,
          validation_data=(x_val, y_val),
          callbacks=mc,
          verbose=1)

model.summary()
print("Finish training.")
