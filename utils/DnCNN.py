# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + tags=[]
# %load_ext autoreload
# %autoreload 2

# + tags=[]
import numpy as np
import tensorflow as tf
from   tensorflow.keras import layers


# + tags=[]
class DnCNN(object):
    """生成、加载DnCNN去噪模型
    """
    
    @classmethod
    def get_model(cls, depth, bn=False, width=64, add=False):
        """生成新DnCNN模型
        
        Args:
            depth: the number of conv layers in the model
            bn: if use the BN layer
            width: width for every conv layer, except for the last layer
        
        Returns:
            model: the new DnCNN model
        """
        In = layers.Input(shape = (None, None, 1))
        def conv(x):
            for i in range(depth - 1):
                x = layers.Conv2D(width, 3, padding='same', use_bias=False)(x)
                if i > 0 and bn:
                    x = layers.BatchNormalization()(x)
                x = layers.ReLU()(x)
            x = layers.Conv2D(1, 3, padding='same', use_bias=False)(x)
            return x
        if add:
            Out = In + conv(In)
        else:
            Out = In - conv(In)
        model = tf.keras.Model(inputs = In, outputs = Out)
        return model
    
    @classmethod
    def load_model(cls, weights_path, depth, bn=False, width=64, add=False):
        """加载训练好的DnCNN模型
        
        Args:
            weights_path: the saved weights of the DnCNN model
            depth: the number of conv layers in the model
            bn: if use the BN layer
            width: width for every conv layer, except for the last layer
        
        Returns:
            model: the trained DnCNN model
        
        """
        model = cls.get_model(depth, bn, width, add)
        model.load_weights(weights_path)
        return model
