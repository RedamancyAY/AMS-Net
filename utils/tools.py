# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import math

# This is matlab version fft operation
def matlab_fft(y):
    y_t = tf.transpose(y, [0,2,1])
    f_y = tf.signal.fft(tf.cast(y_t, tf.complex64))
    f_y = tf.transpose(f_y, [0,2,1])
    return f_y

def matlab_ifft(y):
    y_t = tf.transpose(y, [0,2,1])
    f_y = tf.signal.ifft(tf.cast(y_t, tf.complex64))
    f_y = tf.transpose(f_y, [0,2,1])
    return f_y   

def dct2d_core(x):

    N = tf.shape(x)[0]
    n = tf.shape(x)[1]
    m = tf.shape(x)[2]

    y = tf.reverse(x, axis = [1])
    y = tf.concat([x, y], axis = 1)
    f_y = matlab_fft(y)
    f_y = f_y[:, 0:n, :]

    t = tf.complex(tf.constant([0.0]), tf.constant([-1.0])) * tf.cast(tf.linspace(0.0, tf.cast(n-1, tf.float32), n), tf.complex64)
    t = t * tf.cast(math.pi / (2.0 * tf.cast(n, tf.float64)), tf.complex64)
    t = tf.exp(t) / tf.cast(tf.sqrt(2.0 * tf.cast(n, tf.float64)), tf.complex64)

    # since tensor obejct does not support item assignment, we have to concat a new tensor
    t0 = t[0] / tf.cast(tf.sqrt(2.0), tf.complex64)
    t0 = tf.expand_dims(t0, 0)
    t = tf.concat([t0, t[1:]], axis = 0)
    t = tf.expand_dims(t, -1)
    t = tf.expand_dims(t, 0)
    W = tf.tile(t, [N,1,m])

    dct_x = W * f_y
    dct_x = tf.cast(dct_x, tf.complex64)
    dct_x = tf.math.real(dct_x)

    return dct_x


def idct2d_core(x):

    N = tf.shape(x)[0]
    n = tf.shape(x)[1]
    m = tf.shape(x)[2]

    temp_complex = tf.complex(tf.constant([0.0]), tf.constant([1.0]))
    t = temp_complex * tf.cast(tf.linspace(0.0, tf.cast(n-1, tf.float32), n), tf.complex64)
    t = tf.cast(tf.sqrt(2.0 * tf.cast(n, tf.float64)), tf.complex64) * tf.exp(t * tf.cast(math.pi / (2.0 * tf.cast(n, tf.float64)), tf.complex64))

    t0 = t[0] * tf.cast(tf.sqrt(2.0), tf.complex64)
    t0 = tf.expand_dims(t0, 0)
    t = tf.concat([t0, t[1:]], axis = 0)
    t = tf.expand_dims(t, -1)
    t = tf.expand_dims(t, 0)
    W = tf.tile(t, [N,1,m])

    x = tf.cast(x, tf.complex64)
    yy_up = W * x
    temp_complex = tf.complex(tf.constant([0.0]), tf.constant([-1.0]))
    yy_down = temp_complex * W[:, 1:n, :] * tf.reverse(x[:,1:n, :], axis = [1])
    yy_mid = tf.cast(tf.zeros([N, 1, m]), tf.complex64)
    yy = tf.concat([yy_up, yy_mid, yy_down], axis = 1)
    y = matlab_ifft(yy)
    y = y[:, 0:n, :]
    y = tf.math.real(y)

    return y


def dct2d(x):
    x = dct2d_core(x)
    x = tf.transpose(x, [0,2,1])
    x = dct2d_core(x)
    x = tf.transpose(x, [0,2,1])
    return x

def idct2d(x):
    x = idct2d_core(x)
    x = tf.transpose(x, [0,2,1])
    x = idct2d_core(x)
    x = tf.transpose(x, [0,2,1])
    return x




def saliency_map(img):
    x = tf.transpose(img, perm=[0, 3, 1, 2])
    P = tf.signal.fft2d(tf.complex(x, 0.))
    P = tf.transpose(P, perm=[0, 2, 3, 1])
    myLogAmplitude = tf.math.log(tf.math.abs(P))
    myPhase = tf.math.angle(P)
    mySpectralResidual = myLogAmplitude - tfa.image.mean_filter2d(myLogAmplitude, filter_shape=3)
    t = tf.math.exp(tf.complex(mySpectralResidual, myPhase))
    t = tf.transpose(t, perm=[0, 3, 1, 2])
    t = tf.signal.ifft2d(t)
    t = tf.transpose(t, perm=[0, 2, 3, 1])
    smap = tf.math.abs(t)**2
    smap = tfa.image.gaussian_filter2d(smap, 10, 3)
    return smap


def saliency_info(img, block_size, norm=False):
    """计算图像的每一块的 ``saliency info``
    
    Args:
        img: an image of size (B, H, W)
        block_size: the block size for claculating the saliency info
    
    Returns:
        saliency infomations for all image blocks, 
        (B, H//block_size, W//block_size)
    
    """
    sm = saliency_map(img)
    if tf.rank(sm) == 3:
        sm = sm[..., None]
    if norm:
        a_min = tf.reduce_min(sm, axis=[1, 2, 3], keepdims=True)
        a_max = tf.reduce_max(sm, axis=[1, 2, 3], keepdims=True)
        sm = (sm - a_min) / (a_max - a_min)
        
    b = tf.nn.space_to_depth(sm, block_size)
    b = tf.reduce_sum(b, axis = -1, keepdims=True)
    si = b / tf.reduce_sum(sm, axis=[1, 2, 3], keepdims=True)
    return tf.squeeze(si, axis=-1)

def adjust_si(si, max_ratio):
    """调整图像块的 ``saliency info``， 防止过大
    
    Args:
        si: the saliency info for every block
        max_ratio: the maximum value for every block
    
    Returns:
        the adjusted saliency info
    
    """
    a = tf.where(si > max_ratio, max_ratio, si)
    old = tf.reduce_sum(tf.where(si < max_ratio, si, 0), axis=[1, 2], keepdims=True)
    new = 1. - tf.reduce_sum(tf.cast(a == max_ratio, dtype = tf.float32),
                            axis=[1, 2], keepdims=True) * max_ratio
    si = tf.where(a < max_ratio, a * new / old, a)
    return si 


def sr_assign(sr, _saliency_info, H, W, C, max_nB=1024):
    """根据采样率和 ``saliency info`` 为每一块分配采样资源
    
    Args:
        sr: the sampling ratio
        _saliency_info: the saliency info for every block, (B, H//block_size, W//block_size)
        H: Height of image
        W: Width of image
        C: Channels of image
        max_nB: the maximum number of measurements for image block
    
    Returns:
        number of measurements for every block, (B, H//block_size, W//block_size)
    
    """
    total = tf.math.floor(sr * H * W * C)
    basic = tf.math.ceil(sr / 3  * max_nB)
    rest = tf.math.maximum(0., total - basic * (H//16) * (W//16))

    
    max_ratio = (max_nB-basic) / rest
    if tf.reduce_max(_saliency_info) > max_ratio:
        _saliency_info = adjust_si(_saliency_info, max_ratio)
    if tf.reduce_max(_saliency_info) > max_ratio:
        _saliency_info = adjust_si(_saliency_info, max_ratio)
    if tf.reduce_max(_saliency_info) > max_ratio:
        _saliency_info = adjust_si(_saliency_info, max_ratio)
        
    nB = basic + tf.math.floor(_saliency_info * rest)
    return nB



def nB2mask(nB, length):
    nB = tf.cast(nB, dtype = tf.int32)
    x = tf.range(length)[None, None, None, :] * nB[..., None]
    mask = tf.where(x >= nB[..., None] ** 2, 0, 1)
    mask = tf.cast(mask, tf.float32)
    return mask 




    

def set_used_gpu(use_gpus):
    """设置要使用的gpu
        
        Args:
            use_gpus(Sequence[int]): a list of gpu index, for example [0,1]
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        if type(use_gpus) == int:
            use_gpus = [use_gpus]
        gpus = [gpus[i] for i in use_gpus]
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True) #设置GPU显存用量按需使用
        tf.config.set_visible_devices(gpus,"GPU") 
        print("Set GPU" + str(use_gpus) + "successfully!!!")

