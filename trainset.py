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
import os
import re
import shutil

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm.auto import tqdm

# + tags=[]
import ay.common
import ay.common.device as device
import ay.common.image as image
import ay.data.read as R

# + tags=[]
from utils.image import aug_v1, crop_img, read_file_paths, read_img

# + tags=[]
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir_dataset", type=str, default="./data/dataset")
parser.add_argument("--dir_trainset", type=str, default="./data/trainset")
parser.add_argument("--dataset", type=str, default="BSDS500")
parser.add_argument("--mode", type=str, default="L")
parser.add_argument("--crop_size", type=int, default=128)
parser.add_argument("--crop_num", type=int, default=28)
args = parser.parse_args()


# + tags=[]
def gene_tfrecords(dataset, mode="L", crop_size=128, crop_num=28):
    """generate a tfrecords file from a dataset

    Args:
        dataset(str): the image dataset
        mode(str): the reading mode for image
        crop_size(int): crop every image into sub-images
        crop_num(int): the number of sub-images for every image
        aug_func(function): the augment function

    Returns:
        The name of tfrecords
    """
    file_name = "{dataset}-{mode}-n{crop_num}-s{crop_size}".format(
        dataset=dataset, mode=mode, crop_size=crop_size, crop_num=crop_num
    )
    file_tmp = args.dir_trainset + "/tmp.tfrecords"

    len_X = 0
    path_of_dataset = args.dir_dataset + "/" + args.dataset
    paths = read_file_paths(folder=path_of_dataset)
    print("There are total %d images in %s" % (len(paths), path_of_dataset))

    with tf.io.TFRecordWriter(file_tmp) as writer:
        with tqdm(total=len(paths)) as pbar:
            for path in paths:

                # read image -> deal image -> augment -> crop
                img = image.read_img(path, mode=mode)
                if img.shape[0] < crop_size or img.shape[1] < crop_size:
                    continue
                imgs = aug_v1(img)
                for _img in imgs:
                    _imgs = image.crop_img(img, crop_size=crop_size, num=crop_num)
                    for _img in _imgs:
                        img_string = tf.io.serialize_tensor(_img)
                        writer.write(img_string.numpy())
                        len_X += 1
                pbar.update(1)

    X = tf.data.TFRecordDataset(file_tmp)
    X = X.map(
        lambda x: tf.io.parse_tensor(x, out_type=tf.uint8),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).shuffle(8192)
    file_name = "%s/%s-N%d.tfrecords" % (args.dir_trainset, file_name, len_X)
    with tf.io.TFRecordWriter(file_name) as writer:
        with tqdm(total=len_X) as pbar:
            for img in X:
                img_string = tf.io.serialize_tensor(img)
                writer.write(img_string.numpy())
                pbar.update(1)
    os.remove(file_tmp)

    print("Generate a tfrecord file in: %s" % file_name)

    return file_name


# + tags=[]
gene_tfrecords(
    dataset=args.dataset,
    mode=args.mode,
    crop_size=args.crop_size,
    crop_num=args.crop_num,
)
