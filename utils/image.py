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
import os
import pathlib

# + tags=[]
import math
import numpy as np

from io import BytesIO
from PIL import Image
from typing import Sequence


# + tags=[]
def read_file_paths(folder, ext = [".png", ".jpg", ".pgm",
                                   ".tif", ".pgm", ".jpeg", '.bmp']):
    """read the paths of all images in a folder
    
    Args:
        folder(str): the folder path
        ext(Sequence[str]): the wanted ext of the image
        
    Returns:
        Sequence[str]: the list of file paths
    """
    folder = pathlib.Path(folder)
    all_img_paths = sorted(list(folder.glob('*.*')))
    all_img_paths = [str(path) 
                     for path in all_img_paths
                     if path.suffix.lower() in ext]
    return all_img_paths


# -

def rgb2ycbcr(img, only_y=True):
    '''calculate YCbCr，same with the results of ``matlab rgb2ycbcr``
    
    Args:
        img(uint8, float): the input image
        only_y: only return Y channel
    
    Returns:
        the converted image
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


# + tags=[]
def read_img(img_path: str, mode: str = 'L') -> np.ndarray:
    """read image
    
    Args:
        img_path: the local path of the image
        mode: 1，L，P，RGB，RGBA，CMYK，YCbCr，I，F

    Returns:
        The obtained image from the url.
        
    """
    img_org = Image.open(img_path)
    
    
    if mode == 'YCbCr_Y' and np.array(img_org).ndim == 2:
        mode = 'L'
    
    if mode == 'YCbCr_Y':
        img = np.array(img_org)
        img = rgb2ycbcr(img, only_y=True)
    else:
        img = img_org.convert(mode)
        img = np.array(img)

    if img.ndim == 2:
        img = img[:, :, None]
    return img


# -

def read_imgs(folder, mode = 'L', ext = [".png", ".jpg", ".pgm",
                                   ".tif", ".pgm", ".jpeg", '.bmp']):
    """read all images from the folder
    
    Args:
        folder(str): the folder name
        mode(str): the reading mode for every image
        ext(Sequence[str]): the wanted ext of file
        
    Returns:
        a tuple (imgs, names) where imgs is the list of image, names is the list
        of corresponding image name.
    
    """
    paths = read_file_paths(folder, ext=ext)
    imgs, names = [], []
    for path in paths:
        img = read_img(path, mode = mode)
        imgs.append(img)
        names.append( os.path.splitext(os.path.split(path)[1])[0] )
    return imgs, names


# + tags=[]
def mod_pad(img: np.ndarray, block_size: int) -> np.ndarray:
    """pad an image to make its height/width be integral multiple of ``block_size``

    Args:
        img: the image of size HxWxC
        block_size: base length for the height or width of the image

    Returns:
        The changed image

    """
    assert img.ndim == 3
    H, W, C = img.shape
    if H % block_size == 0 and W % block_size == 0:
        return img

    H2, W2 = (math.ceil(H / block_size) * block_size, 
              math.ceil(W / block_size) * block_size)
    img2 = np.zeros((H2, W2, C))
    img2[0:H, 0:W, :] = img
    return img2


# + tags=[]
def crop_img(img: np.ndarray, crop_size: int, num: int) -> np.ndarray:
    """randomly crop the image
    
    Args:
        img: the original image
        crop_size: the height and width of sub-image
        num: the nubmer of the sub-images
        
    Returns:
        a np.ndarray of size (num, crop_size, crop_size, channels)
    
    """
    assert img.ndim == 3
    assert img.shape[0] >= crop_size and img.shape[1] >= crop_size

    h, w = img.shape[0], img.shape[1]
    I = np.random.randint(0, h - crop_size + 1, (num))
    J = np.random.randint(0, w - crop_size + 1, (num))
    imgs = [img[i : i + crop_size, j : j + crop_size, :] for i, j in zip(I, J)]
    ans = np.stack(imgs, axis = 0)
    return ans

    


# + tags=[]
def aug_v1(img: np.ndarray) -> Sequence[np.ndarray]:
    """ 图像 ``Augment V1``
        
    对图像进行增广，分别进行以下七种操作：左右翻转, 旋转90度, 旋转90度后左右翻转,
    旋转180度, 旋转180度后左右翻转,  旋转270度, 旋转270度后左右翻转。
    
    Args:
        img: the original image
        
    Returns:
        a list of 8 images: the original image + 7 augmented images
                                      
    """

    assert img.ndim == 3
    img1 = np.fliplr(img)
    img2 = np.rot90(img, k = 1)
    img3 = np.fliplr(img2)
    img4 = np.rot90(img, k = 2)
    img5 = np.fliplr(img4)
    img6 = np.rot90(img, k = 3)
    img7 = np.fliplr(img6)
    return [img, img1, img2, img3, img4, img5, img6, img7]
