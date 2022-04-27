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

        cnn_img = self.loader(cnn_img_path)
        crf_img = self.loader(crf_img_path)
        dataL = self.dploader(disp_L)


        wCNN, hCNN = cnn_img.size
        wCRF, hCRF = crf_img.size
        print("CNN Size: ", wCNN, hCNN)
        print("CRF Size: ", wCRF, hCRF)
        #left_img = left_img.crop((w-1232, h-368, w, h))
        #right_img = right_img.crop((w-1232, h-368, w, h))

        dataL = dataL.crop((w-1232, h-368, w, h))
        dataL = np.ascontiguousarray(dataL,dtype=np.float32)/256

        processed = preprocess.get_transform(augment=False)
        cnn_img       = processed(cnn_img)
        crf_img      = processed(crf_img)


        return cnn_img_path, crf_img_path, disp_L, cnn_img, crf_img, dataL

    def __len__(self):
        return len(self.left)
