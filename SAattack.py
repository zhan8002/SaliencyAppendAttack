#!/bin/env python

import numpy as np
import argparse
import os
import random
import cv2
from sklearn.preprocessing import LabelEncoder
import torch
import torchvision.transforms as transforms
from PIL import Image


def random_padding(image, be_image, padding_rate, target_class):
    image_trans = np.asarray(transform(image))
    ori_image = image_trans
    channel, height, width = image_trans.shape
    padding_height = int(np.ceil(height*padding_rate))

    be_image = transform(be_image)
    be_image = np.asarray(be_image)

    idx = []
    for i in range(round(224*padding_rate)):
        idx.append(random.randint(0, 223))
    benign = np.zeros((channel, len(idx), np.asarray(be_image).shape[1]))

    for c in range(channel):
        for row in range(len(idx)):
            benign[c][row]= np.asarray(be_image)[c][idx[row]]


    pad_image = np.transpose(benign, (1, 2, 0))
    pad_image = cv2.resize(pad_image,(width, padding_height))
    pad_image = np.transpose(pad_image, (2, 0, 1))

    padding_image = np.zeros([channel, padding_height + height, width])
    padding_image[:, :height, :] = ori_image
    padding_image[:, height:, :] = pad_image
    return padding_image

def benign_padding(image, be_image, padding_rate, target_class): # Sample Injection proposed in COPYCAT
    image_trans = np.asarray(transform(image))
    channel, height, width = image_trans.shape
    ori_image = image_trans
    padding_height = int(np.floor(height*padding_rate))

    be_image = np.asarray(be_image)

    be_transformed = transform(Image.fromarray(be_image))
    be_transformed = np.asarray(be_transformed)

    pad_image = np.transpose(be_transformed, (1, 2, 0))
    pad_image = cv2.resize(pad_image, (width, padding_height))
    pad_image = np.transpose(pad_image, (2, 0, 1))

    padding_image = np.zeros([channel, padding_height + height, width])
    padding_image[:, :height, :] = ori_image
    padding_image[:, height:, :] = pad_image

    return padding_image

def benign2_padding(image, be_image, padding_rate, target_class): # Benign append proposed by suciu
    image_trans = np.asarray(transform(image))
    channel, height, width = image_trans.shape
    ori_image = image_trans
    padding_height = int(np.floor(height*padding_rate))

    be_image = np.asarray(be_image)

    be_transformed = transform(Image.fromarray(be_image))
    be_transformed = np.asarray(be_transformed)

    pad_image = be_transformed[:, :padding_height, :]


    padding_image = np.zeros([channel, padding_height + height, width])
    padding_image[:, :height, :] = ori_image
    padding_image[:, height:, :] = pad_image

    return padding_image

def cam_padding(image, be_image, padding_rate, target_class): # the proposed Saliency Attack
    image_trans = np.asarray(transform(image))
    ori_image = image_trans
    channel, height, width = image_trans.shape
    padding_height = int(np.floor(height*padding_rate))

    be_image = transform(be_image)
    be_image = np.asarray(be_image)

    target_name = target_class
    filepath = './common_hot2/'+target_name+'.txt'
    f = open(filepath, 'r+')
    heat = f.read()
    heat = heat.strip('[]')
    s_heat = np.array(heat.split())


    idx = s_heat.argsort()[-round(224*padding_rate/(1+padding_rate)):]
    idx = sorted(idx)
    benign = np.zeros((channel, len(idx), np.asarray(be_image).shape[1]))
    for c in range(channel):
        for row in range(len(idx)):
            benign[c][row]= np.asarray(be_image)[c][idx[row]]


    pad_image = np.transpose(benign, (1, 2, 0))
    pad_image = cv2.resize(pad_image,(width, padding_height))
    pad_image = np.transpose(pad_image, (2, 0, 1))

    padding_image = np.zeros([channel, padding_height + height, width])
    padding_image[:, :height, :] = ori_image
    padding_image[:, height:, :] = pad_image
    return padding_image



la = LabelEncoder()
la.fit_transform(['Allaple.A', 'Allaple.L', 'Benign', 'Fakerean', 'Instantaccess','VB.AT', 'Yuner.A'])
cate = la.classes_

# 定义超参数
parser = argparse.ArgumentParser()
parser.add_argument('--image_width', type=int, default=224, help="Width of each input images")
parser.add_argument('--image_height', type=int, default=224, help="Height of each input images")
parser.add_argument('--image_resize', type=int, default=224, help="Resize scale")
parser.add_argument('--image_channel', type=int, default=3, help="Gray or RGB.")
parser.add_argument('--iter_num', type=int, default=200, help="iteration numbers")
parser.add_argument('--decay_factor', type=float, default=1.0, help="momentum weight")
parser.add_argument('--alpha', type=float, default=0.01, help="iteration parameter")
parser.add_argument('--input_dir', type=str, default='./chen/minival', help="Input directory with images")
parser.add_argument('--output_dir', type=str, default='./output', help="Output directory with images")
parser.add_argument('--mask_dir', type=str, default='./mask', help="Mask directory with images")
#parser.add_argument('--checkpoint_path', type=str, default='/home/zh/adversarial_samples/code/zhan/img-malware/classifier/cnn/models/model_malimg_gray.pth', help=" Path to checkpoint for inception network")
parser.add_argument('--checkpoint_path', type=str, default='./models/vgg_cls7.pth', help=" Path to checkpoint for network")


args = parser.parse_args()

use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
padding_rate_set = np.linspace(1, 0, 21)

args.input_dir = '/home/ubuntu/zhan/dataset/1plus6/val/'
# Load model
net = torch.load(args.checkpoint_path)
net = net.to(device)
net.eval()

transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=args.image_channel),
     transforms.Resize(size=(args.image_width, args.image_height)),
     transforms.ToTensor(),
     ])

for padding_rate in padding_rate_set[:-1]:
    num_file = 0
    num_success = 0.0001
    confusion = []
    for target_class in cate:
        imagepath = args.input_dir + target_class
        pathdir = os.listdir(imagepath)
        for oc in cate:
            #if target_class != 'Benign' or oc == 'Benign' or target_class==oc: #M2B
            if target_class == 'Benign' or oc == 'Benign' or target_class==oc: #M2M
                continue
            clsname = oc
            cls_success = 0
            cls_number = 0
            for i in range(len(os.listdir(args.input_dir + "/" + clsname))):
                sample = random.sample(pathdir, 1)
                be_image = Image.open(imagepath + '/' + sample[0])

                filename = os.listdir(args.input_dir + "/" + clsname)[i]
                image = Image.open(args.input_dir + "/" + clsname + "/" + filename)
                image_transformed = transform(image)

                # Get the initial predictions
                image_transformed = image_transformed.unsqueeze(0)
                image_transformed = image_transformed.to(device)
                output = net(image_transformed)
                original_value, original_idx = torch.max(output, 1)

                # Choose attack method
                padding_image = cam_padding(image, be_image, padding_rate=padding_rate, target_class=target_class)


                pad_image = cv2.resize(np.transpose(padding_image,(1, 2, 0)), (224, 224))
                pad_image = np.transpose(pad_image, (2, 0, 1))
                pad_image = torch.from_numpy(pad_image)
                pad_image = pad_image.unsqueeze(0)

                adv_output = net(pad_image.type(torch.FloatTensor).cuda())
                adv_value, adv_idx = torch.max(adv_output, 1)

                if oc != target_class:
                    num_file += 1
                    cls_number += 1

                    if cate[adv_idx] == target_class:
                        num_success += 1
                        cls_success += 1
                torch.cuda.empty_cache()
    print('success rate is :', num_success/num_file, "with appending rate ", padding_rate)


