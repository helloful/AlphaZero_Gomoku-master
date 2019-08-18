# -*- coding: utf-8 -*-
"""
Implement the policy value network using numpy, so that we can play with the
trained AI model without installing any DL framwork

@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np


# some utility functions
def softmax(x): # 分类函数 -- 分类器
    probs = np.exp(x - np.max(x)) # np.max()求序列的最值
    probs /= np.sum(probs)
    return probs


def relu(X):# 激活函数
    out = np.maximum(X, 0) # X逐个与0进行比较，取最大值
    return out

'''
N, C, H, W分别代表输入数据的批数据量、每个数据的通道数、每个通道的高、每个通道的宽
'''
def conv_forward(X, W, b, stride=1, padding=1):
    n_filters, d_filter, h_filter, w_filter = W.shape # 32，,6,128个卷积核4*3*3，也即是多核卷积  输入层 4*8*8
    # print(W.shape)
    # theano conv2d flips the filters (rotate 180 degree) first
    # while doing the calculation
    W = W[:, :, ::-1, ::-1]
    n_x, d_x, h_x, w_x = X.shape
    # print(X.shape)
    # 计算输出的h,w
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    h_out, w_out = int(h_out), int(w_out)
    X_col = im2col_indices(X, h_filter, w_filter,
                           padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    out = (np.dot(W_col, X_col).T + b).T   # dot()点积运算
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2) # 转置
    return out


def fc_forward(X, W, b):
    out = np.dot(X, W) + b
    return out


def get_im2col_indices(x_shape, field_height,
                       field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    # 没有看懂如何计算输出的大小
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    # im2col的作用就是优化卷积运算
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')  # 将numpy数组按指定的方法填充成指定的形状,此处把四周全部填充为0
    k, i, j = get_im2col_indices(x.shape, field_height,
                                 field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


class PolicyValueNetNumpy():
    """policy-value network in numpy """
    def __init__(self, board_width, board_height, net_params):
        self.board_width = board_width
        self.board_height = board_height
        self.params = net_params
        # print(net_params)

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables # list类型
        current_state = board.current_state() # state shape: 4*width*height
        # print(current_state)

        X = current_state.reshape(-1, 4, self.board_width, self.board_height)
        # 经过测试X与输入之前的current_state一样
        # first 3 conv layers with ReLu nonlinearity
        for i in [0, 2, 4]:
            X = relu(conv_forward(X, self.params[i], self.params[i+1]))
            print(X)

        # policy head
        X_p = relu(conv_forward(X, self.params[6], self.params[7], padding=0))
        X_p = fc_forward(X_p.flatten(), self.params[8], self.params[9])
        act_probs = softmax(X_p)
        # value head
        X_v = relu(conv_forward(X, self.params[10],
                                self.params[11], padding=0))
        X_v = relu(fc_forward(X_v.flatten(), self.params[12], self.params[13]))
        value = np.tanh(fc_forward(X_v, self.params[14], self.params[15]))[0]
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value
