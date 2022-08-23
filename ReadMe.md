# AMS-Net: Adaptive Multi-Scale Network for Image Compressive Sensing

This is the training code and test code for the paper [AMS-Net: Adaptive Multi-Scale Network for Image Compressive Sensing](https://ieeexplore.ieee.org/document/9855869). 

If you have any question, please contact me (zkyhitsz@gmail.com).

## Parapare

We have put the trainset, image datasets, pre-trained model in the `data` directory:
* `data/AMS-Net/`: the pre-trained model
* `data/dataset/`: the image datasets
* `data/trainset/`: the training set to train our AMS-Net

**If you have to put these files or your trained model in your custome folders, you can change the args in the next training or test cmd**:
* dir_dataset: where you put your image datasets
* dir_trainset: where you put your training set
* dir_modelsave: where you put your pre-trained AMS-Net

## Train Model

### TrainSet

You can download the training set from [BaiduYunDrive(code: 9ene)](https://pan.baidu.com/s/15-hbc_0J0CAGWoUPkuoN9Q), or [Google Dirve](https://drive.google.com/file/d/1rSl8wAZVhu1YLPYnHs6QII-Nst26IR-9/view?usp=sharing). Then put the training set in `data/trainset/`. 

#### Generate custom training set

Our training set is constructed using the training set and validation set of [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500). We directly put all the images of the training set and validation set in the path `./data/dataset/BSDS500/`. 

To generate your custom training set, you can run the following code:
```shell
python trainset.py --dir_dataset "./data/dataset" --dir_trainset "./data/trainset" \
    --dataset "BSDS500" --mode "L" --crop_size 128 --crop_num 28
```

where the optional parameters are
* dataset: the dataset which contain $n$ images.
* mode: For the color image in the dataset, seting mode to "L" denotes that we directly convert the color image into gray image. You can set mode to "YCbCr_Y" for extracting the Y channel from YCbCr space to train the model.
* crop_size: the image size in the training set
* crop_num: the number of cropped images for each augmented iamge.

Note that the total number of images in the training set is ![](https://latex.codecogs.com/svg.latex?n \times 8 \times crop-num), and each image is of size $crop-size \times crop-size$.

### Train 

You can run the following code to train the model:

```python
python model_train.py --gpu 1 --width 64 --depth 5 --T 10 --projection 1 \
    --dir_modelsave "./data/AMS-Net2"  --dir_dataset "./data/dataset" \
    --dir_trainset "./data/trainset" --trainset "BSDS500-L-n28-s128-N89600.tfrecords"
```

where the optional parameters are:
* gpu: set used gpu
* width, depth, T, projection: the parameters to construct the AMS-Net
* trainset: the file of trianing set that put in the dir_trainset

## Test model

### Test

You can run the following code to obtain the reconstruction results:
```python
python model_test.py --gpu 0 --dir_modelsave "./data/AMS-Net"  --dir_dataset "./data/dataset" \
    --mode "L" --width 64 --depth 5 --T 10 --projection 1 \
    --datasets "Set5" "Set11" "Set14" "BSD100" \
    --write_img 1 1 1 0
```
where the optional parameters are:
* gpu: set used gpu
* mode: For the color image in the dataset, seting mode to "L" denotes that we directly convert the color image into gray image. You can set mode to "YCbCr_Y" for extracting the Y channel from YCbCr space to test the model.
* width, depth, T, projection: the parameters to construct the AMS-Net
* datasets: the test sets
* write_img: whether should save reconstructed image for each test set. this arg should have the same length with the arg `datasets` 

### Reconstrction results

The test results are put in the directory `result`:
* `result/W{width}-D{depth}-T{T}-Proj{projection}-{mode}.csv`: all the psnr, ssim scores for all images in the test sets at each sampling ratio.
* `result/reconstructed_imgs/W{width}-D{depth}-T{T}-Proj{projection}-{mode}/*/*.png`: the reconstructed images


You can use the following code to load the reconstruction results and obtain the averge PSNR and SSIM score:
```python
import pandas as pd
data = pd.read_csv("result/xxx.csv")
data.groupby(["dataset", "sr"]).mean()
```

