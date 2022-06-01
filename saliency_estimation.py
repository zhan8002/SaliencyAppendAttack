import os
import PIL
import numpy as np
import torch
import torchvision.transforms as transforms
from gradcam import GradCAM, GradCAMpp

transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=3),
     transforms.Resize(size=(224, 224)),
     transforms.ToTensor(),
     ])

input_dir = '/home/ubuntu/zhan/dataset/1plus6/val'

vgg = torch.load('./models/vgg_cls7.pth')
vgg.eval()
vgg.cuda()

resnet = torch.load('./models/resnet_cls7.pth')
resnet.eval()
resnet.cuda()

densenet = torch.load('./models/dense_cls7.pth')
densenet.eval()
densenet.cuda()

squeezenet = torch.load('./models/squeeze_cls7.pth')
squeezenet.eval()
squeezenet.cuda()

mean_hot = []


for c in range(len(os.listdir(input_dir))):
    clsname = os.listdir(input_dir)[c]
    hmeans = 0

    filepath = './common_hot2/'+clsname+'.txt'
    f = open(filepath, 'a')
    for i in range(len(os.listdir(input_dir + "/" + clsname))):
        filename = os.listdir(input_dir + "/" + clsname)[i]
        pil_img = PIL.Image.open(input_dir + "/" + clsname + "/" + filename)

        torch_img = transform(pil_img)
        normed_torch_img = torch_img.unsqueeze(0)
        normed_torch_img = normed_torch_img.cuda()

        cam_dict = dict()

        vgg_model_dict = dict(type='vgg', arch=vgg, layer_name='features_29', input_size=(224, 224))
        vgg_gradcam = GradCAM(vgg_model_dict, True)
        vgg_gradcampp = GradCAMpp(vgg_model_dict, True)
        cam_dict['vgg'] = [vgg_gradcam, vgg_gradcampp]

        resnet_model_dict = dict(type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        resnet_gradcam = GradCAM(resnet_model_dict, True)
        resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
        cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]

        densenet_model_dict = dict(type='densenet', arch=densenet, layer_name='features_norm5', input_size=(224, 224))
        densenet_gradcam = GradCAM(densenet_model_dict, True)
        densenet_gradcampp = GradCAMpp(densenet_model_dict, True)
        cam_dict['densenet'] = [densenet_gradcam, densenet_gradcampp]

        squeezenet_model_dict = dict(type='squeezenet', arch=squeezenet, layer_name='features_12_expand3x3_activation',
                                     input_size=(224, 224))
        squeezenet_gradcam = GradCAM(squeezenet_model_dict, True)
        squeezenet_gradcampp = GradCAMpp(squeezenet_model_dict, True)
        cam_dict['squeezenet'] = [squeezenet_gradcam, squeezenet_gradcampp]

        hsum=0
        for gradcam, gradcam_pp in cam_dict.values():
            mask, _ = gradcam(normed_torch_img)
            hot = np.array(mask[0][0].cpu())
            h = np.mean(hot, axis=1)
            hsum += h
        hmean_model = hsum/4  #hot means in all models
        hmeans += hmean_model
    hmean_cls = hmeans / len(os.listdir(input_dir + "/" + clsname))
    f.write(str(hmean_cls).replace('\n', ' '))
    f.close()


