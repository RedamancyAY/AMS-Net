# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + tags=[]
import pywt, math
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers


# + tags=["active-ipynb"]
# %load_ext autoreload
# %autoreload 2
#
# import matplotlib.pyplot as plt
# %matplotlib inline
# from utils import get_net_img
# -

# # 卷积实现小波变换

# + tags=[]
def get_dwt_filters(dwt_base):
    """
        Get dwt filters for a specifical base.
        Input:
            dwt_base -> one of ['haar', 'db', 'sym', 'coif', 'bior', 'rbio',
                                'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor']
        Output:
            filters -> the filters to implement DWT2
            inv_filters -> the filters to implement iDWT2
    """
    w = pywt.Wavelet(dwt_base)
    dec_hi, dec_lo = w.dec_hi[::-1], w.dec_lo[::-1]
    rec_hi, rec_lo = w.rec_hi, w.rec_lo
    filters = tf.stack([
        np.outer(dec_lo, dec_lo),
        np.outer(dec_lo, dec_hi),
        np.outer(dec_hi, dec_lo),
        np.outer(dec_hi, dec_hi),
    ], axis = -1)
    filters = np.expand_dims(filters,axis = 2)
    inv_filters = tf.stack([
        np.outer(rec_lo, rec_lo),
        np.outer(rec_lo, rec_hi),
        np.outer(rec_hi, rec_lo),
        np.outer(rec_hi, rec_hi),
    ], axis = -1)
    inv_filters = np.expand_dims(inv_filters, axis = 2)
    return filters, inv_filters


# -

# * 小波变换：
#     * coe = dwt_layer(img)
#         * 输入： img => (B, H, W, 1)
#         * 输出： coe => (B, H/2, W/2, 4)
# * 逆小波变换：
#     * img = idwt_layer(coe)
#         * 输入： coe => (B, H/2, W/2, 4)
#         * 输出： img => (B, H, W, 1)

# + tags=[]
def get_dwt_layer(dwt_base):
    filters, inv_filters = get_dwt_filters(dwt_base)
    d = filters.shape[0]
    dwt_layer = tf.keras.layers.Conv2D(4, d, strides = 2, padding='VALID',\
                                       use_bias = False, weights = [filters])
    idwt_layer = tf.keras.layers.Conv2DTranspose(1, d, strides = 2, padding='VALID',\
                                                 use_bias = False, weights = [inv_filters])
    dwt_layer.trainable = False
    idwt_layer.trainable = False
    return dwt_layer, idwt_layer


# -

# # 小波变换

# 把4个系数矩阵拼接成一个系数矩阵。

# + tags=[]
class DWT:
    def __init__(self, dwt_base = 'haar'):
        self.dwt, self.idwt = get_dwt_layer(dwt_base)
    def get_dwt_coe(self, x, enlargeLL = 1.):
        coe = self.dwt(x)
        a = tf.unstack(coe, axis = -1)
        w1 = tf.concat([a[0] * enlargeLL , a[1]], axis = -1)
        w2 = tf.concat([a[2], a[3]], axis = -1)
        h = tf.concat([w1, w2], axis = -2)
        h = tf.expand_dims(h, axis = -1)
        return h
    def coe2img(self, coe):
        b, h, w, c = coe.shape
        patch = tf.image.extract_patches(images=coe,\
                                         sizes=[1, h // 2, w // 2, 1], strides=[1, h // 2, w // 2, 1],\
                                         rates=[1, 1, 1, 1], padding='VALID')
        a = tf.reshape(patch, (1, 4, h // 2, w // 2))
        a = tf.transpose(a, (0, 2, 3, 1))
        a = self.idwt(a)
        return a

# + tags=["active-ipynb"]
# image = get_net_img()
# img = image[None, :, :, 0:1].astype(np.float32)
# n = 3

# + tags=["active-ipynb"]
# plt.figure(figsize=(10, 5))
# plt.subplot(1, n, 1)
# plt.imshow(np.squeeze(img), cmap = 'gray')
# plt.title("Original image")
#
# dwt = DWT(dwt_base='bior2.2')
# coe = dwt.get_dwt_coe(img, 1)
# plt.subplot(1, n, 2)
# plt.imshow(np.squeeze(coe), cmap = 'gray')
# plt.title("Coefficient")
#
# img_re = dwt.coe2img(coe)
# plt.subplot(1, n, 3)
# plt.imshow(np.squeeze(img_re), cmap = 'gray')
# plt.title("inverse dwt")
# print("The error between original image and the inversed image after idwt is %.2f" %(np.mean((img - img_re)**2)))
