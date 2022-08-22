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
import warnings

from silence_tensorflow import silence_tensorflow

warnings.simplefilter("ignore", UserWarning)
silence_tensorflow()

# + tags=[]
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

from AMS_Net import CS_model
from utils.cs_nn import MeanSquaredError, Callback_TestModel
from utils.tools import set_used_gpu


# + tags=[]
def get_train_set(args):
    crop_size = eval(re.findall("-s(\d*)", args.trainset)[0])
    len_X = eval(re.findall("-N(\d*)", args.trainset)[0])

    def _parse_img_func_norm(example):
        img_tensor = tf.io.parse_tensor(example, out_type=tf.uint8)
        img_tensor = tf.cast(img_tensor, tf.float32)
        img_tensor = img_tensor / 255.0
        img_tensor.set_shape([crop_size, crop_size, 1])
        return (img_tensor, img_tensor)

    img_ds = tf.data.TFRecordDataset(os.path.join(args.dir_trainset, args.trainset))
    X = img_ds.map(
        _parse_img_func_norm, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    X = (
        X.shuffle(1024)
        .batch(args.batch_size)
        .repeat()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    steps_per_epoch = len_X // args.batch_size
    return X, steps_per_epoch


# + tags=[]
if __name__ == "__main__":
    tf.random.set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--projection", type=int, default=1)

    parser.add_argument("--dir_dataset", type=str, default="./data/dataset")
    parser.add_argument("--dir_trainset", type=str, default="./data/trainset")
    parser.add_argument("--dir_modelsave", type=str, default="./data/AMS-Net2")
    parser.add_argument(
        "--trainset", type=str, default="BSDS500-L-n28-s128-N89600.tfrecords"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    set_used_gpu([args.gpu])


    model = CS_model(
        width=args.width, depth=args.depth, T=args.T, projection=args.projection
    )
    x = np.random.rand(1, 128, 128, 1).astype(np.float32)
    _ = model(x)
    model_name = "W%d-D%d-T%d-Proj%d" % (
        args.width,
        args.depth,
        args.T,
        args.projection,
    )
    
    X, steps_per_epoch = get_train_set(args)
    lr_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=0.0001,
        decay_steps=args.epochs * steps_per_epoch,
        end_learning_rate=0.00001,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_fn)
    model_save = ModelCheckpoint(os.path.join(args.dir_modelsave, model_name), save_weights_only=True)
    csv_logger = CSVLogger("result/logs/%s.csv"%model_name, append=True)
    model_test = Callback_TestModel(sr=0.1, test_folder=os.path.join(args.dir_dataset, "Set11"))
    
    model.compile(optimizer=optimizer, loss=MeanSquaredError())
    model.fit(
        x=X,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[model_save, model_test, csv_logger],
    )

# + tags=[]
# !python model_train.py --gpu 1 --width 64 --depth 5 --T 10 --projection 1 \
#     --dir_modelsave "./data/AMS-Net2"  --dir_dataset "./data/dataset" \
#     --dir_trainset "./data/trainset" --trainset "BSDS500-L-n28-s128-N89600.tfrecords"
