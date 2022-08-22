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
# %load_ext autoreload
# %autoreload 2

# + tags=[]
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

import os

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
# -

from utils.cs_nn import conv_layer_for_mm
from utils.DnCNN import DnCNN
from utils.dwt import DWT
from utils.tools import nB2mask, saliency_info, sr_assign


# + tags=[]
def load_DnCNN_models(depth, width, T):
    # print("Load pretrained DnCNN")
    assert width in [32, 64]
    assert depth in [3, 5, 7, 9, 11]
    filepath = "./data/DnCNN/D%d-W%d-BN0-N0.h5" % (depth, width)
    models = [DnCNN.load_model(filepath, depth=depth, width=width) for _ in range(T)]
    return models


# -

class CS_model(tf.keras.Model):
    def __init__(self, width=64, depth=5, T=10, projection=True):
        super().__init__()
        self.blk_size_LL = 16
        self.width = width
        self.depth = depth
        self.T = T
        self.projection = projection
        self.train_ar = 0.9
        self.test_ar = 0.95

        self.dwt = DWT()
        self.phi_ll, self.re_ll = conv_layer_for_mm(256, 256, blk_size=16)
        self.phi_hh, self.re_hh = conv_layer_for_mm(768, 768, blk_size=16)
        self.denoising_blocks = load_DnCNN_models(depth, width, T)

    def get_ar(self, x, test=False):
        ar = self.test_ar if test else self.train_ar
        if x < 0.1:
            y = ar
        else:
            y = -x + ar
        return y

    def get_mask(self, x, sr, ar):
        B, H, W, C = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        _saliency_info = saliency_info(x, block_size=16, norm=True)
        sr_ll = tf.math.minimum(1.0, sr * 4 * ar)
        sr_hh = (sr * 4 - sr_ll) / 3
        nB_ll = sr_assign(sr_ll, _saliency_info, H, W, C=1, max_nB=256)
        nB_ll = tf.clip_by_value(nB_ll, 1, 256)
        nB_hh = sr_assign(sr_hh, _saliency_info, H, W, C=3, max_nB=768)
        nB_hh = tf.clip_by_value(nB_hh, 1, 768)
        mask_ll = nB2mask(nB_ll, length=256)
        mask_hh = nB2mask(nB_hh, length=768)
        return mask_ll, mask_hh


    def sampling_base(self, x, phi, re, blk_size, mask):
        y = phi(x)
        y_mask = y * mask
        x = re(y_mask)
        x = tf.nn.depth_to_space(x, block_size=blk_size)
        return x

    def Approximation(self, x0, xi, phi, re, blk_size, mask):
        xj = self.sampling_base(xi, phi, re, blk_size, mask)
        return xi + x0 - xj

    def deep_re(self, x_ll, x_hh, x, mask_ll, mask_hh):
        for layer in self.denoising_blocks:
            if self.projection:
                x = self.dwt.dwt(x)
                LL, HH = x[..., 0:1], x[..., 1:4]
                p = self.Approximation(x_ll, LL, self.phi_ll, self.re_ll, 16, mask_ll)
                q = self.Approximation(x_hh, HH, self.phi_hh, self.re_hh, 16, mask_hh)
                x = self.dwt.idwt(tf.concat([p, q], axis=-1))
            x = layer(x)
        return x

    @tf.function
    def train(self, x, sr=-1, ar=-1):
        # get sr and ar
        if sr == -1:
            sr = tf.math.floor(tf.random.uniform([], minval=1, maxval=50)) / 100
            ar = self.get_ar(sr)

        # get mask
        coe4 = self.dwt.dwt(x)
        LL, HH = coe4[..., 0:1], coe4[..., 1:4]
        mask_ll, mask_hh = self.get_mask(LL, sr, ar)
        B, H, W = x.shape[0], x.shape[1], x.shape[2]
        mask_ll.set_shape([B, H // 32, W // 32, 256])
        mask_hh.set_shape([B, H // 32, W // 32, 768])

        # sampling and reconstruction
        x_ll = self.sampling_base(LL, self.phi_ll, self.re_ll, 16, mask_ll)
        x_hh = self.sampling_base(HH, self.phi_hh, self.re_hh, 16, mask_hh)
        x0 = self.dwt.idwt(tf.concat([x_ll, x_hh], axis=-1))
        x = self.deep_re(x_ll, x_hh, x0, mask_ll, mask_hh)
        return x

    def call(self, x):
        return self.train(x)

    def test(self, x):
        return self.train(x, self.sr, self.ar)
