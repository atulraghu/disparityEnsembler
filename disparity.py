from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
import copy
from dataloader import KITTIloader2015 as ls
from dataloader import KITTILoader as DA

from models import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--cnn_imgs_path')
parser.add_argument('--crf_imgs_path')
parser.add_argument('--disp_L_path')
parser.add_argument('--savedisp', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(args.cnn_imgs_path, args.crf_imgs_path, args.disp_L_path)
batch_size = 1

TrainingImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img,all_right_img,all_left_disp, False),
         batch_size= batch_size, shuffle= False, num_workers= 2, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img+test_left_img,all_right_img+test_right_img,all_left_disp+test_left_disp, False),
         batch_size= batch_size, shuffle= False, num_workers= 2, drop_last=False)

ValImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False),
         batch_size= batch_size, shuffle= False, num_workers= 2, drop_last=False)

def loss(disp_true,pred_disp):
    # disp_true = disp_true.cpu()
    # pred_disp = pred_disp.cpu()
    true_disp = copy.deepcopy(disp_true)
    index = np.argwhere(true_disp>0)
    true_disp[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
    correct = (true_disp[index[0][:], index[1][:], index[2][:]] < 5)|(true_disp[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05) #CHANGE BACK TO 3PX
    torch.cuda.empty_cache()
    return 1-(float(torch.sum(correct))/float(len(index[0])))


def get_loss(cnn_img_path, crf_img_path, disp_L, cnn_disp, crf_disp, disp_true):
    #ONLY FOR BATCH SIZE 1
    for i in range(len(cnn_disp)):
        #print(disp_true.shape,cnn_disp.shape)
        cnn_loss = loss(disp_true, cnn_disp)
        crf_loss = loss(disp_true, crf_disp)
        # if (cnn_loss>crf_loss):
        #     print("CRF",cnn_loss, crf_loss)
        # else:
        #     print("CNN",cnn_loss, crf_loss)
    return cnn_loss, crf_loss

def label(cnn_img_path, crf_img_path, disp_L, cnn_disp, crf_disp, disp_true):
    results = []
    for i in range(len(cnn_disp)):
        #print(disp_true.shape,cnn_disp.shape)
        cnn_loss = loss(disp_true, cnn_disp)
        crf_loss = loss(disp_true, crf_disp)
        if (cnn_loss>crf_loss):
            #print("CRF",cnn_loss, crf_loss)
            results.append(1)
        else:
            #print("CNN",cnn_loss, crf_loss)
            results.append(0)
    return torch.tensor(results)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,1,6)
        self.conv2 = nn.Conv2d(1,1,6)
        self.pool1 = nn.MaxPool2d(6, 20)
        self.pool2 = nn.MaxPool2d(6, 20)
        self.fc1 = nn.Linear(2356 , 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, cnn, crf):
        cnn = cnn.unsqueeze(0)
        cnn = self.conv1(cnn)
        crf = crf.unsqueeze(0)
        crf = self.conv2(crf)
        cnn = self.pool1(cnn)
        crf = self.pool2(crf)
        cnn = torch.flatten(cnn, 1) # flatten all dimensions except batch
        crf = torch.flatten(crf, 1)
        x = torch.cat((cnn,crf),1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
#net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

def main():
    cnn_count = 0
    crf_count = 0
    net.train()
    # cnn_losses = []
    # crf_losses = []
    # for batch_idx, (cnn_img_path, crf_img_path, disp_L, cnn_disp, crf_disp, disp_true) in enumerate(TestImgLoader):
    #     cnn_loss, crf_loss = get_loss(cnn_img_path, crf_img_path, disp_L, cnn_disp, crf_disp, disp_true)
    #     cnn_losses.append(cnn_loss)
    #     crf_losses.append(crf_loss)
    #     if (cnn_loss>crf_loss):
    #         crf_count+=1
    #     else:
    #         cnn_count+=1
    # print("CNN LOSS", sum(cnn_losses)/len(cnn_losses), "CNN WINS", cnn_count)
    # print("CRF LOSS", sum(crf_losses)/len(crf_losses), "CRF WINS", crf_count)

    cnn_count = 5
    crf_count = 5
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for batch_idx, (cnn_img_path, crf_img_path, disp_L, cnn_disp, crf_disp, disp_true) in enumerate(TrainingImgLoader):
            #cnn_disp, crf_disp, disp_true=cnn_disp.cuda(), crf_disp.cuda(), disp_true.cuda()
            labels = label(cnn_img_path, crf_img_path, disp_L, cnn_disp, crf_disp, disp_true)
            if labels[0] == np.random.randint(2):#np.random.binomial(1, cnn_count/(cnn_count+crf_count)):
                continue
            if labels[0] == 0:
                cnn_count+=1
            else:
                crf_count+=1
            optimizer.zero_grad()
            output = net(cnn_disp,crf_disp)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 20 == 19:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0
    print('finished training', cnn_count-5, crf_count-5)

    acc = 0
    total = 0
    predict_cnn = 0
    predict_crf = 0
    ensemble_loss = []
    for batch_idx, (cnn_img_path, crf_img_path, disp_L, cnn_disp, crf_disp, disp_true) in enumerate(TestImgLoader):
        net.eval()
        prediction= torch.max(net(cnn_disp,crf_disp), 1).indices
        real_label = label(cnn_img_path, crf_img_path, disp_L, cnn_disp, crf_disp, disp_true)
        cnn_loss, crf_loss = get_loss(cnn_img_path, crf_img_path, disp_L, cnn_disp, crf_disp, disp_true)
        if prediction == 0:
            predict_cnn+=1
            ensemble_loss.append(cnn_loss)
        else:
            predict_crf+=1
            ensemble_loss.append(crf_loss)

        if prediction==real_label:
            acc+=1
        total +=1
        # print("Predict",prediction)
        # print("Label",real_label)
    print("Accuracy",acc/total)
    print("#CNN preds", predict_cnn, "#CRF preds", predict_crf)
    print("Ensemble Loss", sum(ensemble_loss)/len(ensemble_loss))

    acc = 0
    total = 0
    predict_cnn = 0
    predict_crf = 0
    ensemble_loss = []
    for batch_idx, (cnn_img_path, crf_img_path, disp_L, cnn_disp, crf_disp, disp_true) in enumerate(ValImgLoader):
        net.eval()
        prediction= torch.max(net(cnn_disp,crf_disp), 1).indices
        real_label = label(cnn_img_path, crf_img_path, disp_L, cnn_disp, crf_disp, disp_true)
        cnn_loss, crf_loss = get_loss(cnn_img_path, crf_img_path, disp_L, cnn_disp, crf_disp, disp_true)
        if prediction == 0:
            predict_cnn+=1
            ensemble_loss.append(cnn_loss)
        else:
            predict_crf+=1
            ensemble_loss.append(crf_loss)

        if prediction==real_label:
            acc+=1
        total +=1
        # print("Predict",prediction)
        # print("Label",real_label)
    print("Accuracy",acc/total)
    print("#CNN preds", predict_cnn, "#CRF preds", predict_crf)
    print("Ensemble Loss", sum(ensemble_loss)/len(ensemble_loss))



if __name__ == '__main__':
    main()
