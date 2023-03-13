#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2022/07/31 15:08:15
@Author  :   ykzhou 
@Version :   0.0
@Contact :   ykzhou@stu.xidian.edu.cn
@Desc    :   None
'''

num_joints = 14
batch_size = 128
total_epoch = 200
dataset = 'lsp'

# 0-heatmap 1-regression
train_mode = 0

# eval mode: 0-output image,1-pck score
eval_mode = 0

continue_train = 0
best_pre_train = 0
epoch_to_test = 106

