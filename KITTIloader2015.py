import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(cnn_imgs_path, crf_imgs_path, disp_L_path):

    image = [img for img in os.listdir(crf_imgs_path) if img.find('_10') > -1]

    train = image[:160]
    val   = image[160:]

    cnn_train  = [os.path.join(cnn_imgs_path,img) for img in train]
    crf_train = [os.path.join(crf_imgs_path,img) for img in train]
    disp_train_L = [os.path.join(disp_L_path,img) for img in train]
    #disp_train_R = [filepath+disp_R+img for img in train]

    cnn_val  = [os.path.join(cnn_imgs_path,img) for img in val]
    crf_val = [os.path.join(crf_imgs_path,img) for img in val]
    disp_val_L = [os.path.join(disp_L_path,img) for img in val]
    #disp_val_R = [filepath+disp_R+img for img in val]

    return cnn_train, crf_train, disp_train_L, cnn_val, crf_val, disp_val_L
