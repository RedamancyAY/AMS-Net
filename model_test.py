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

warnings.filterwarnings("ignore")
silence_tensorflow()

# + tags=[]
import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as layers
from tqdm.auto import tqdm

# import ay.cs.test as Test
from AMS_Net import CS_model
from utils.image import read_file_paths
from utils.test import test_imgSet
from utils.tools import set_used_gpu

# + tags=[]
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--dir_dataset", type=str, default="./data/dataset")
parser.add_argument("--dir_modelsave", type=str, default="./data/AMS-Net")
parser.add_argument("--mode", type=str, default="L")
parser.add_argument(
    "--datasets", type=str, nargs="+", default=["Set5", "Set11", "Set14", "BSD100"]
)
parser.add_argument("--write_img", type=int, nargs="+", default=[1, 1, 1, 0])
parser.add_argument("--T", type=int, default=10)
parser.add_argument("--width", type=int, default=64)
parser.add_argument("--depth", type=int, default=5)
parser.add_argument("--projection", type=int, default=1)
args = parser.parse_args()
set_used_gpu(args.gpu)
assert len(args.datasets) == len(args.write_img)
print("Test AMS-Net on ", args.datasets)


# load model
model = CS_model(
    width=args.width, depth=args.depth, T=args.T, projection=args.projection
)
x = np.random.rand(1, 128, 128, 1).astype(np.float32)
_ = model(x)

model_name = "W%d-D%d-T%d-Proj%d" % (args.width, args.depth, args.T, args.projection)
model.load_weights("%s/%s" % (args.dir_modelsave, model_name)).expect_partial()
model.model_name = model_name

# + tags=[]
# test model
for dataset in args.datasets:
    path_dataset = os.path.join(args.dir_dataset, dataset)
    paths = read_file_paths(path_dataset)
    print("There are %d images in dataset %s" % (len(paths), path_dataset))

data = pd.DataFrame()
test_srs = [0.01, 0.03, 0.04, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
for sr in tqdm(test_srs, desc="Test sr", leave=True):
    model.sr = sr
    model.ar = model.get_ar(sr, test=True)
    for i in tqdm(
        range(len(args.datasets)), desc=f"Test datasets at sr={sr}", leave=True
    ):
        _data = test_imgSet(
            model,
            os.path.join(args.dir_dataset, args.datasets[i]),
            mode=args.mode,
            write_img=args.write_img[i],
        )
        data = data.append(_data, ignore_index=True)

data.to_csv("result/%s-%s.csv" % (model_name, args.mode), index=False)
