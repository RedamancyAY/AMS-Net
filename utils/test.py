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
# %load_ext autoreload
# %autoreload 2

# + tags=[]
import time, os

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2 as cv

# + tags=[]
from .image import mod_pad, read_imgs


# -

def test_img(model, img, patch=32):
    """test model for the input image

    Args:
        model: the CS model
        img: the tested image
        patch: the height and width should be divisible by ``patch``

    Returns:
        a tuple (img_re, psnr, ssim, time)
    """

    if "patch_size" in dir(model):
        patch = model.patch_size
    elif "blk_size" in dir(model):
        patch = model.blk_size

    img_pad = mod_pad(img, patch)

    s = time.time()
    img_norm = tf.expand_dims(img_pad / 255, 0)
    img_norm = tf.cast(img_norm, tf.float32)
    img_re = model.test(img_norm)[0, 0 : img.shape[0], 0 : img.shape[1], :]
    img_re = np.clip(np.round(img_re * 255), a_min=0, a_max=255).astype(np.uint8)
    e = time.time()

    psnr = tf.image.psnr(img, img_re, max_val=255)
    ssim = tf.image.ssim(img, img_re, max_val=255)
    return img_re, psnr.numpy(), ssim.numpy(), e - s


# + tags=[]
def test_imgSet(model, dataset, mode="L", write_img=False):
    """使用CS模型测试一整个数据集

    Args:
        model: the CS model
        dataset: the tested dataset
        mode: the mode to reading every image
        patch: the height and width should be divisible by ``patch``

    Returns:
        a table of ``pd.DataFrame`` format
    """
    imgs, img_names = read_imgs(folder=dataset, mode=mode)
    columns = ["dataset", "sr", "image", "psnr", "ssim", "time"]
    data = pd.DataFrame(columns=columns)
    dataset = dataset.split("/")[-1]
    for img, img_name in zip(imgs, img_names):
        img_re, psnr, ssim, t = test_img(model, img)
        data = data.append(
            dict(zip(columns, [dataset, model.sr, img_name, psnr, ssim, t])),
            ignore_index=True,
        )
        if write_img:
            _dir = os.path.join("result/reconstructed_imgs/", model.model_name + '-' + mode, str(int(model.sr * 100)))
            _file = "%s-%s-%.4f-%.4f.png" % (dataset, img_name, psnr, ssim)
            if not os.path.exists(_dir):
                os.makedirs(_dir)
            cv.imwrite(os.path.join(_dir, _file), img_re)
    return data

