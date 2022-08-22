# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + tags=[]
import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layers
from .test import test_imgSet


# + tags=[]
def orth_mat(phi):
    '''对矩阵按行正交化
    
    Args:
        phi: the original matrix
    
    Returns:
        The orthogonal matrix
    '''
    def normalize(v):
        return v / np.sqrt(v.dot(v))     
    phi[0, :] = normalize(phi[0, :])
    for i in range(1, phi.shape[0]):
        Ai = phi[i, :]
        for j in range(0, i):
            Aj = phi[j, :]
            t = Ai.dot(Aj)
            Ai = Ai - t * Aj
        phi[i, :] = normalize(Ai)
    return phi


# + tags=[]
def gene_orth_mat(m, n, seed=42, mode = 'normal'):
    '''生成随机正交矩阵
    
    Args:
        m: the height
        n: the width
        seed: the seed of np.random
        mode: 'normal' for normal distribution, else for random distribution
    
    Returns:
        the orthogonal matrix
    '''
    if seed != -1:
        np.random.seed(seed)
    if mode == 'normal':
        phi = np.random.normal(0, 1 / m, (m, n))
    else:
        phi = np.random.rand(m, n)
    phi = orth_mat(phi).astype(np.float32)
    return phi


# + tags=[]
def conv_layer_for_mm(m, n, blk_size = 32, seed = 42):
    """generate two conv layers for sampling and initial reconstruction
    
    Args:
        m(int): the length after sampling
        n(int): the length of original singal
        blk_size(int): the block size for sampling
        seed(int): the random seed for generating the orthogonal matrix
        
    Returns:
        two conv layers, a conv layer for sampling and the other for initial 
        restruction
    """
    assert n % blk_size == 0
    phi = gene_orth_mat(m, n, seed = seed).T
    re = tf.linalg.pinv(phi)
    phi_conv = layers.Conv2D(m, blk_size, strides = blk_size, padding = 'same',
                             use_bias = False, weights = [tf.reshape(phi, (blk_size, blk_size, -1, m))])
    re_conv = layers.Conv2D(n, 1, strides = 1, padding = 'same',
                            use_bias = False, weights = [tf.reshape(re, (1, 1, m, n))])
    return phi_conv, re_conv


# -

class MeanSquaredError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))/2


class Callback_TestModel(tf.keras.callbacks.Callback):
    
    def __init__(self, sr=0.1, test_folder='data/dataset/Set11'):
        super().__init__()
        self.sr = 0.1
        self.test_folder = test_folder
        
    def on_epoch_end(self, epoch, logs=None):
        self.model.sr = self.sr
        self.model.ar = self.model.get_ar(self.sr)
        
        data = test_imgSet(model = self.model, dataset = self.test_folder, mode="L")
        psnr = np.mean(data['psnr'])
        ssim = np.mean(data['ssim'])
        logs['psnr'] = psnr
        logs['ssim'] = ssim
