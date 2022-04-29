import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
from dataloader import preprocess

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return Image.open(path)

def to_shape(x, target_shape):
    padding_list = []
    for x_dim, target_dim in zip(x.shape, target_shape):
        pad_value = max((target_dim - x_dim),0)
        pad_tuple = ((pad_value//2, pad_value//2 + pad_value%2))
        padding_list.append(pad_tuple)

    return torch.tensor(np.pad(x, tuple(padding_list), mode='constant'))

class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader= disparity_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        cnn_img_path  = self.left[index]
        crf_img_path = self.right[index]
        disp_L= self.disp_L[index]

        cnn_img = self.dploader(cnn_img_path)
        crf_img = self.dploader(crf_img_path)
        dataL = self.dploader(disp_L)

        wCNN, hCNN = cnn_img.size
        wCRF, hCRF = crf_img.size
        w,h = dataL.size

        #cnn_img = cnn_img.crop((wCNN-1226, hCNN-368, wCNN, hCNN))
        #crf_img = crf_img.crop((wCRF-1226, hCRF-368, wCRF, hCRF))
        cnn_img = np.ascontiguousarray(cnn_img,dtype=np.float32)/256
        crf_img = np.ascontiguousarray(crf_img,dtype=np.float32)/256
        dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256

        cnn_img = to_shape(cnn_img,(375,1242) )
        crf_img = to_shape(crf_img,(375,1242) )
        dataL = to_shape(dataL,(375,1242) )

        return cnn_img_path, crf_img_path, disp_L, cnn_img, crf_img, dataL

    def __len__(self):
        return len(self.left)
