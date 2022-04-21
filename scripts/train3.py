#!/usr/bin/env python3

# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

from __future__ import print_function

"""
HOW TO TRAIN DOPE

This is the DOPE training code.  
It is provided as a convenience for researchers, but it is otherwise unsupported.

Please refer to `python3 train.py --help` for specific details about the 
training code. 

If you download the FAT dataset 
(https://research.nvidia.com/publication/2018-06_Falling-Things)
you can train a YCB object DOPE detector as follows: 

```
python3 train.py --data path/to/FAT --object soup --outf soup 
--gpuids 0 1 2 3 4 5 6 7 
```

This will create a folder called `train_soup` where the weights will be saved 
after each epoch. It will use the 8 gpus using pytorch data parallel. 
"""


import argparse
import configparser
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.models as models
import datetime
import json
import glob
import os
import cv2
import imutils

from tqdm import tqdm

from PIL import Image
from PIL import ImageDraw
from PIL import ImageEnhance

from math import acos
from math import sqrt
from math import pi    

from os.path import exists

import cv2
import colorsys

from dope.utils import make_grid

from torchsummary import summary

import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

n_belief = 4    # number of belief map
##################################################
# NEURAL NETWORK MODEL
##################################################

class DopeNetwork(nn.Module):
    def __init__(
            self,
            pretrained=False,
            numBeliefMap=n_belief,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)
        ):
        super(DopeNetwork, self).__init__()

        self.stop_at_stage = stop_at_stage

        if pretrained is False: print("Training network without imagenet weights.")
        else: print("Training network pretrained on imagenet.")

        # vgg_full = models.vgg19(pretrained=pretrained).features
        mobilenetV2_full = models.mobilenet_v2(pretrained=True).features
        # print(mobilenetV2_full)
        # print(vgg_full)
        # self.vgg = nn.Sequential()
        self.mobile = nn.Sequential()
        # for i_layer in range(24): self.vgg.add_module(str(i_layer), vgg_full[i_layer])
        for i_layer in range(7): self.mobile.add_module(str(i_layer), mobilenetV2_full[i_layer])

        # Add some layers
        # i_layer = 23
        # self.vgg.add_module(str(i_layer), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1))
        # self.vgg.add_module(str(i_layer+1), nn.ReLU(inplace=True))
        # self.vgg.add_module(str(i_layer+2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        # self.vgg.add_module(str(i_layer+3), nn.ReLU(inplace=True))
        i_layer = 7
        self.mobile.add_module(str(i_layer), nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1))
        self.mobile.add_module(str(i_layer+1), nn.ReLU(inplace=True))
        # self.mobile.add_module(str(i_layer+2), nn.Conv2d(160, 80, kernel_size=3, stride=1, padding=1))
        # self.mobile.add_module(str(i_layer+3), nn.ReLU(inplace=True))

        # print('---Belief------------------------------------------------')
        # _2 are the belief map stages
        # self.m1_2 = DopeNetwork.create_stage(128, numBeliefMap, True)
        # self.m2_2 = DopeNetwork.create_stage(128 + numBeliefMap, numBeliefMap, False)
        # self.m3_2 = DopeNetwork.create_stage(128 + numBeliefMap, numBeliefMap, False)
        # self.m4_2 = DopeNetwork.create_stage(128 + numBeliefMap, numBeliefMap, False)
        # self.m5_2 = DopeNetwork.create_stage(128 + numBeliefMap, numBeliefMap, False)
        # self.m6_2 = DopeNetwork.create_stage(128 + numBeliefMap, numBeliefMap, False)
        self.m1_2 = DopeNetwork.create_stage(64, numBeliefMap, True)
        self.m2_2 = DopeNetwork.create_stage(64 + numBeliefMap, numBeliefMap, False)
        self.m3_2 = DopeNetwork.create_stage(64 + numBeliefMap, numBeliefMap, False)
        self.m4_2 = DopeNetwork.create_stage(64 + numBeliefMap, numBeliefMap, False)
        self.m5_2 = DopeNetwork.create_stage(64 + numBeliefMap, numBeliefMap, False)
        self.m6_2 = DopeNetwork.create_stage(64 + numBeliefMap, numBeliefMap, False)

    def forward(self, x):
        '''Runs inference on the neural network'''
        # out1 = self.vgg(x)
        out1 = self.mobile(x)
        out1_2 = self.m1_2(out1)
        if self.stop_at_stage == 1: return [out1_2]
        out2 = torch.cat([out1_2, out1], 1)
        out2_2 = self.m2_2(out2)
        if self.stop_at_stage == 2: return [out1_2, out2_2]
        out3 = torch.cat([out2_2, out1], 1)
        out3_2 = self.m3_2(out3)
        if self.stop_at_stage == 3: return [out1_2, out2_2, out3_2]
        out4 = torch.cat([out3_2, out1], 1)
        out4_2 = self.m4_2(out4)
        if self.stop_at_stage == 4: return [out1_2, out2_2, out3_2, out4_2]
        out5 = torch.cat([out4_2, out1], 1)
        out5_2 = self.m5_2(out5)
        if self.stop_at_stage == 5: return [out1_2, out2_2, out3_2, out4_2, out5_2]
        out6 = torch.cat([out5_2, out1], 1)
        out6_2 = self.m6_2(out6)
        return [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2]

    @staticmethod
    def create_stage(in_channels, out_channels, first=False):   # Create the neural network layers for a single stage.
        model = nn.Sequential()
        mid_channels = 64
        if first:
            padding = 1
            kernel = 3
            count = 6
            # count = 4
            final_channels = 128
        else:
            padding = 3
            kernel = 7
            count = 10
            # count = 6
            final_channels = mid_channels

        # First convolution
        model.add_module("0",
                         nn.Conv2d(
                             in_channels,
                             mid_channels,
                             kernel_size=kernel,
                             stride=1,
                             padding=padding)
                        )

        # Middle convolutions
        i = 1
        while i < count - 1:
            model.add_module(str(i), nn.ReLU(inplace=True))
            i += 1
            model.add_module(str(i),
                             nn.Conv2d(
                                 mid_channels,
                                 mid_channels,
                                 kernel_size=kernel,
                                 stride=1,
                                 padding=padding))
            i += 1

        # Penultimate convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=1))
        i += 1

        # Last convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(final_channels, out_channels, kernel_size=1, stride=1))
        i += 1

        return model



##################################################
# UTILS CODE FOR LOADING THE DATA
##################################################

def default_loader(path):
    return Image.open(path).convert('RGB')          

def loadjson(path, objectofinterest):
    """
    Loads the data from a json file.
    If there are no objects of interest, then load all the objects.
    """
    with open(path) as data_file:
        data = json.load(data_file)
    pointsBelief = []
    # centroids = []

    translations = []
    rotations = []
    points = []

    for i_line in range(len(data['objects'])):
        info = data['objects'][i_line]
        if not objectofinterest is None and \
                not objectofinterest in info['class'].lower():
            continue

        # 3d bbox with belief maps
        points3d = []

        pointdata = info['projected_cuboid']
        for p in pointdata:
            points3d.append((p[0], p[1]))

        # if len(points3d) == 8:
        #     # NDDS format: 8 points in 'projected_cuboid', 1 point in 'projected_cuboid_centroid'
        #     pcenter = info['projected_cuboid_centroid']
        #     points3d.append((pcenter[0], pcenter[1]))
        # elif len(points3d) == 9:
        #     # nvisii format: 9 points in 'projected_cuboid', no 'projected_cuboid_centroid' key
        #     pcenter = points3d[-1]
        # else:
        #     raise RuntimeError(f'projected_cuboid has to have 8 or 9 points while reading "{path}"')

        pointsBelief.append(points3d)
        points.append(points3d)  # NOTE: Adding the centroid again is probably a bug.
        # centroids.append((pcenter[0], pcenter[1]))

        # load translations
        location = info['location']
        translations.append([location[0], location[1], location[2]])

        # quaternion
        rot = info["quaternion_xyzw"]
        rotations.append(rot)

    return {
        "pointsBelief": pointsBelief,
        "rotations": rotations,
        "translations": translations,
        # "centroids": centroids,
        "points": points,
        "keypoints_2d": [],
    }

def loadimages(root):
    """
    Find all the images in the path and folders, return them in imgs. 
    """
    imgs = []

    def add_json_files(path,):
        for imgpath in glob.glob(path+"/*.png"):
            if exists(imgpath) and exists(imgpath.replace('png',"json")):
                imgs.append((imgpath,imgpath.replace(path,"").replace("/",""),
                    imgpath.replace('png',"json")))
        for imgpath in glob.glob(path+"/*.jpg"):
            if exists(imgpath) and exists(imgpath.replace('jpg',"json")):
                imgs.append((imgpath,imgpath.replace(path,"").replace("/",""),
                    imgpath.replace('jpg',"json")))

    def explore(path):
        if not os.path.isdir(path):
            return
        folders = [os.path.join(path, o) for o in os.listdir(path) 
                        if os.path.isdir(os.path.join(path,o))]
        if len(folders)>0:
            for path_entry in folders:                
                explore(path_entry)
        add_json_files(path)

    explore(root)

    return imgs

class MultipleVertexJson(data.Dataset):
    """
    Dataloader for the data generated by NDDS (https://github.com/NVIDIA/Dataset_Synthesizer). 
    This is the same data as the data used in FAT.
    """
    def __init__(self, root,transform=None, nb_vertex = 8,
            keep_orientation = True, 
            normal = None, test=False, 
            target_transform = None,
            loader = default_loader, 
            objectofinterest = "",
            img_size = 480,
            save = False,  
            noise = 2,
            data_size = None,
            sigma = 16,
            random_translation = (25.0,25.0),
            random_rotation = 15.0,
            ):
        ###################
        self.objectofinterest = objectofinterest
        self.img_size = img_size
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.imgs = []
        self.test = test
        self.normal = normal
        self.keep_orientation = keep_orientation
        self.save = save 
        self.noise = noise
        self.data_size = data_size
        self.sigma = sigma
        self.random_translation = random_translation
        self.random_rotation = random_rotation

        def load_data(path):
            '''Recursively load the data.  This is useful to load all of the FAT dataset.'''
            imgs = loadimages(path)

            # Check all the folders in path 
            for name in os.listdir(str(path)):
                imgs += loadimages(path +"/"+name)
            return imgs


        self.imgs = load_data(root)

        # Shuffle the data, this is useful when we want to use a subset. 
        np.random.shuffle(self.imgs)

    def __len__(self):
        # When limiting the number of data
        if not self.data_size is None:
            return int(self.data_size)

        return len(self.imgs)   

    def __getitem__(self, index):
        path, name, txt = self.imgs[index]
        img = self.loader(path)
        img_size = img.size
        loader = loadjson
        data = loader(txt, self.objectofinterest)
        pointsBelief = data['pointsBelief']
        img_original = img.copy()

        # Create the belief map
        beliefsImg = CreateBeliefMap(img, pointsBelief=pointsBelief, nbpoints = n_belief, sigma = self.sigma)

        # Create the image maps for belief
        transform = transforms.Compose([transforms.Resize(min(img_size))])
        totensor = transforms.Compose([transforms.ToTensor()])

        for j in range(len(beliefsImg)):
            beliefsImg[j] = self.target_transform(beliefsImg[j])
            # beliefsImg[j].save('{}.png'.format(j))
            beliefsImg[j] = totensor(beliefsImg[j])

        beliefs = torch.zeros((len(beliefsImg),beliefsImg[0].size(1),beliefsImg[0].size(2)))
        for j in range(len(beliefsImg)):
            beliefs[j] = beliefsImg[j][0]

        ## Convert image (PIL) -> Tensor ##
        img = self.transform(img)
        # w_crop = np.random.randint(0, img.size[0] - img_size[0] + 1)
        # h_crop = np.random.randint(0, img.size[1] - img_size[1] + 1)

        # transform = transforms.Compose([transforms.Resize(min(img_size))])
        totensor = transforms.Compose([transforms.ToTensor()])
        # img = crop(img, h_crop, w_crop, img_size[1], img_size[0])
        img = totensor(img)
        if not self.normal is None:
            normalize = transforms.Compose([transforms.Normalize
                                            ((self.normal[0], self.normal[0], self.normal[0]),
                                             (self.normal[1], self.normal[1], self.normal[1])),
                                            AddNoise(self.noise)])
        else:
            normalize = transforms.Compose([AddNoise(0.0001)])
        img = normalize(img)

        ## Visualize target belief map
        # beliefs_cpu = beliefs.cpu().detach().numpy()
        # img_original = np.asarray(img_original)
        # overlay = np.zeros((480, 640), dtype=np.float32)
        # for i in range(4): overlay += imutils.resize(beliefs_cpu[i], height=img_original.shape[0])
        # overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        # overlay = np.array(overlay * 255, dtype=np.uint8)
        # canvas = cv2.addWeighted(img_original, 0.3, overlay, 0.7, 0)
        # cv2.imshow("A", canvas)
        # cv2.waitKey(0)

        # w_crop = int(w_crop / 8)
        # h_crop = int(h_crop / 8)
        # beliefs = beliefs[:, h_crop:h_crop + int(img_size[1] / 8), w_crop:w_crop + int(img_size[0] / 8)]
        return {
                    'img':img,
                    'beliefs':beliefs,
                    'point_beliefs':pointsBelief
                }

"""
Some simple vector math functions to find the angle
between two points, used by affinity fields. 
"""
def length(v):
    return sqrt(v[0]**2+v[1]**2)

def dot_product(v,w):
   return v[0]*w[0]+v[1]*w[1]

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]

def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v)*length(w))
   rad=acos(cosx) # in radians
   return rad*180/pi # returns degrees

def py_ang(A, B=(1,0)):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner

def CreateBeliefMap(img,pointsBelief,nbpoints,sigma=16):
    """
    Args: 
        img: image
        pointsBelief: list of points in the form of 
                      [nb object, nb points, 2 (x,y)] 
        nbpoints: (int) number of points, DOPE uses 8 points here
        sigma: (int) size of the belief map point
    return: 
        return an array of PIL black and white images representing the 
        belief maps         
    """
    beliefsImg = []
    sigma = int(sigma)
    for numb_point in range(nbpoints):    
        array = np.zeros(img.size)
        for point in pointsBelief:
            p = point[numb_point]
            w = int(sigma*2)
            if p[0] - w >= 0 and p[0] + w < img.size[0] and p[1] - w >= 0 and p[1] + w < img.size[1]:
                for i in range(int(p[0])-w, int(p[0])+w):
                    for j in range(int(p[1])-w, int(p[1])+w):
                        array[i, j] = np.exp(-(((i - p[0])**2 + (j - p[1])**2)/(2*(sigma**2))))

        stack = np.stack([array,array,array],axis=0).transpose(2, 1, 0)
        beliefsImg.append(Image.fromarray((stack*255).astype('uint8')))
    return beliefsImg


def crop(img, i, j, h, w):
    """
    Crop the given PIL.Image.
    
    Args:
        img (PIL.Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL.Image: Cropped image.
    """
    return img.crop((j, i, j + w, i + h))
    
class AddRandomContrast(object):
    """
    Apply some random contrast from PIL
    """
    
    def __init__(self,sigma=0.1):
        self.sigma = sigma

    def __call__(self, im):
        contrast = ImageEnhance.Contrast(im)
        im = contrast.enhance( np.random.normal(1,self.sigma) )        
        return im


class AddRandomBrightness(object):
    """
    Apply some random brightness from PIL
    """

    def __init__(self,sigma=0.1):
        self.sigma = sigma

    def __call__(self, im):
        bright = ImageEnhance.Brightness(im)
        im = bright.enhance( np.random.normal(1,self.sigma) )
        return im        

class AddNoise(object):
    """
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """
    
    def __init__(self,std=0.1):
        self.std = std

    def __call__(self, tensor):
        # TODO: make efficient
        # t = torch.FloatTensor(tensor.size()).uniform_(self.min,self.max)
        t = torch.FloatTensor(tensor.size()).normal_(0,self.std)

        t = tensor.add(t)
        t = torch.clamp(t,-1,1) #this is expansive
        return t


def save_image(tensor, filename, nrow=4, padding=2,mean=None, std=None):
    """
    Saves a given Tensor into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    from PIL import Image
    
    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=10,pad_value=1)
    if not mean is None:
        ndarr = grid.mul(std).add(mean).mul(255).byte().transpose(0,2).transpose(0,1).numpy()
    else:      
        ndarr = grid.mul(0.5).add(0.5).mul(255).byte().transpose(0,2).transpose(0,1).numpy()
    im = Image.fromarray(ndarr)
    im.save(filename)

##################################################
# TRAINING CODE MAIN STARTING HERE
##################################################

print ("start:" , datetime.datetime.now().time())

conf_parser = argparse.ArgumentParser(
    description=__doc__, # printed with -h/--help
    # Don't mess with format of description
    formatter_class=argparse.RawDescriptionHelpFormatter,
    # Turn off help, so we print all options in response to -h
    add_help=False
    )
conf_parser.add_argument("-c", "--config",
                        help="Specify config file", metavar="FILE")

parser = argparse.ArgumentParser()

parser.add_argument('--data',  
    default = "/media/teera/ROGESD/dataset/dope/output/chessboard_color_emptyboard",
    help='path to training data')

parser.add_argument('--datatest', 
    default="", 
    help='path to data testing set')

parser.add_argument('--object', 
    default='chessboard',
    help='In the dataset which objet of interest')

parser.add_argument('--workers', 
    type=int, 
    default=2,
    help='number of data loading workers')

parser.add_argument('--batchsize', 
    type=int, 
    default=32, # VRAM 11.3 GB
    help='input batch size')

parser.add_argument('--imagesize', 
    type=int, 
    default=480,
    help='the height / width of the input image to network')

parser.add_argument('--lr', 
    type=float, 
    default=0.00003,
    help='learning rate, default=0.001')

parser.add_argument('--noise', 
    type=float, 
    default=2.0, 
    help='gaussian noise added to the image')

parser.add_argument('--net', 
    default='', 
    help="path to net (to continue training)")

parser.add_argument('--namefile', 
    default='epoch', 
    help="name to put on the file of the save weights")

parser.add_argument('--manualseed',
    type=int,
    help='manual seed')

parser.add_argument('--epochs',
    type=int,
    default=120,
    help="number of epochs to train")

parser.add_argument('--loginterval',
    type=int,
    default=1000)

parser.add_argument('--gpuids',
    nargs='+',
    type=int,
    default=[0], 
    help='GPUs to use')

parser.add_argument('--outf', 
    default='/media/teera/ROGESD/model/belief/chessboard_mono_6_stage_lr_0.00003',
    help='folder to output images and model checkpoints, it will \
    add a train_ in front of the name')

parser.add_argument('--sigma', 
    default=5,
    help='keypoint creation size for sigma')

parser.add_argument('--save', 
    action="store_true", 
    help='save a visual batch and quit, this is for\
    debugging purposes')

parser.add_argument("--pretrained",
    default=True,
    help='do you want to use vgg imagenet pretrained weights')

parser.add_argument('--nbupdates', 
    default=None, 
    help='nb max update to network, overwrites the epoch number\
    otherwise uses the number of epochs')

parser.add_argument('--datasize', 
    default=None, 
    help='randomly sample that number of entries in the dataset folder') 

# Read the config but do not overwrite the args written 
args, remaining_argv = conf_parser.parse_known_args()
defaults = { "option":"default" }

if args.config:
    config = ConfigParser.SafeConfigParser()
    config.read([args.config])
    defaults.update(dict(config.items("defaults")))

parser.set_defaults(**defaults)
parser.add_argument("--option")
opt = parser.parse_args(remaining_argv)

if opt.pretrained in ['false', 'False']:
    opt.pretrained = False

if not "/" in opt.outf:
    opt.outf = "train_{}".format(opt.outf)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualseed is None:
    opt.manualseed = random.randint(1, 10000)

# save the hyper parameters passed
with open (opt.outf+'/header.txt','w') as file: 
    file.write(str(opt)+"\n")

with open (opt.outf+'/header.txt','w') as file: 
    file.write(str(opt))
    file.write("seed: "+ str(opt.manualseed)+'\n')
    with open (opt.outf+'/test_metric.csv','w') as file:
        file.write("epoch, passed,total \n")

# set the manual seed. 
random.seed(opt.manualseed)
torch.manual_seed(opt.manualseed)
torch.cuda.manual_seed_all(opt.manualseed)

# save 
if not opt.save:
    contrast = 0.2
    brightness = 0.2
    noise = 0.1
    normal_imgs = [0.59,0.25]
    transform = transforms.Compose([
                               AddRandomContrast(contrast),
                               AddRandomBrightness(brightness),
                               transforms.Resize(opt.imagesize),
                               ])
else:
    contrast = 0.00001
    brightness = 0.00001
    noise = 0.00001
    normal_imgs = None
    transform = transforms.Compose([
                           transforms.Resize(opt.imagesize),
                           transforms.ToTensor()])

print ("load data")
#load the dataset using the loader in utils_pose
trainingdata = None
if not opt.data == "":
    train_dataset = MultipleVertexJson(
        root = opt.data,
        objectofinterest=opt.object,
        keep_orientation = True,
        noise = opt.noise,
        sigma = opt.sigma,
        data_size = opt.datasize,
        save = opt.save,
        transform = transform,
        normal = normal_imgs,
        target_transform = transforms.Compose([
                               transforms.Resize(opt.imagesize//8),
            ]),
        )
    trainingdata = torch.utils.data.DataLoader(train_dataset,
        batch_size = opt.batchsize, 
        shuffle = True,
        num_workers = opt.workers, 
        pin_memory = True
        )

if opt.save:
    for i in range(2):
        images = iter(trainingdata).next()
        if normal_imgs is None:
            normal_imgs = [0,1]
        save_image(images['img'],'{}/train_{}.png'.format( opt.outf,str(i).zfill(5)),mean=normal_imgs[0],std=normal_imgs[1])

        print (i)        

    print ('things are saved in {}'.format(opt.outf))
    quit()

testingdata = None
if not opt.datatest == "": 
    testingdata = torch.utils.data.DataLoader(
        MultipleVertexJson(
            root = opt.datatest,
            objectofinterest=opt.object,
            keep_orientation = True,
            noise = opt.noise,
            sigma = opt.sigma,
            data_size = opt.datasize,
            save = opt.save,
            transform = transform,
            normal = normal_imgs,
            target_transform = transforms.Compose([
                                   transforms.Resize(opt.imagesize//8),
                ]),
            ),
        batch_size = opt.batchsize, 
        shuffle = True,
        num_workers = opt.workers, 
        pin_memory = True)

if not trainingdata is None:
    print('training data: {} batches'.format(len(trainingdata)))
if not testingdata is None:
    print ("testing data: {} batches".format(len(testingdata)))
print('load models')

net = DopeNetwork(pretrained=opt.pretrained).cuda()
net = torch.nn.DataParallel(net,device_ids=opt.gpuids).cuda()

if opt.net != '':
    net.load_state_dict(torch.load(opt.net))

parameters = filter(lambda p: p.requires_grad, net.parameters())
optimizer = optim.Adam(parameters,lr=opt.lr)

# with open (opt.outf+'/loss_train.csv','w') as file:
#     file.write('epoch,batchid,loss\n')
#
# with open (opt.outf+'/loss_test.csv','w') as file:
#     file.write('epoch,batchid,loss\n')

loss_train_file = open(opt.outf + '/loss_train.csv', 'w')
loss_train_file.write('epoch,batchid,loss\n')
loss_test_file = open(opt.outf + '/loss_test.csv', 'w')
loss_test_file.write('epoch,batchid,loss\n')

nb_update_network = 0

def _runnetwork(epoch, loader, train=True):
    global nb_update_network
    # net
    if train:
        net.train()
    else:
        net.eval()

    with tqdm(total=len(loader.dataset)) as pbar:   # Progressbar with number of dataset
        pbar.set_description("Epoch %d" % epoch)
        for batch_idx, targets in enumerate(loader):

            data = Variable(targets['img'].cuda())

            output_belief = net(data)

            if train:
                optimizer.zero_grad()
            target_belief = Variable(targets['beliefs'].cuda())

            loss = None

            ## Belief maps loss ##
            for l in output_belief:  # output, each belief map layers.
                # print(l.shape)                      # torch.Size([16, 4, 60, 80])
                # print(l[0][0].shape, data.shape)    # torch.Size([60, 80]) torch.Size([16, 3, 480, 640])
                if loss is None:
                    loss = ((l - target_belief) * (l - target_belief)).mean()
                else:
                    loss_tmp = ((l - target_belief) * (l - target_belief)).mean()
                    loss += loss_tmp

            ## Affinities loss ##
            # for l in output_affinities:  # output, each belief map layers.
            #     loss_tmp = ((l - target_affinity) * (l - target_affinity)).mean()
            #     loss += loss_tmp

            if train:
                loss.backward()
                optimizer.step()
                nb_update_network += 1

            if train:
                namefile = '/loss_train.csv'
            else:
                namefile = '/loss_test.csv'

            with open(opt.outf + namefile, 'a') as file:
                s = '{}, {},{:.15f}\n'.format(
                    epoch, batch_idx, loss.data.item())
                # print (s)
                file.write(s)

            if train:
                if batch_idx % opt.loginterval == 0:
                    # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.15f}'.format(
                    #     epoch, batch_idx * len(data), len(loader.dataset),
                    #            100. * batch_idx / len(loader), loss.data.item()))
                    loss_train_file.write('{},{},{:.15f}'.format(epoch, batch_idx * len(data), loss.data.item()))
            else:
                if batch_idx % opt.loginterval == 0:
                    # print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.15f}'.format(
                    #     epoch, batch_idx * len(data), len(loader.dataset),
                    #            100. * batch_idx / len(loader), loss.data.item()))
                    loss_test_file.write('{},{},{:.15f}'.format(epoch, batch_idx * len(data), loss.data.item()))
            # break
            pbar.update(len(data))  # Update Progressbar (+len(data))
            if not opt.nbupdates is None and nb_update_network > int(opt.nbupdates):
                torch.save(net.state_dict(), '{}/net_{}.pth'.format(opt.outf, opt.namefile))
                break


for epoch in range(1, opt.epochs + 1):

    if not trainingdata is None:
        _runnetwork(epoch,trainingdata)

    if not opt.datatest == "":
        _runnetwork(epoch,testingdata,train = False)
        if opt.data == "":
            break # lets get out of this if we are only testing
    try:
        torch.save(net.state_dict(), '{}/net_{}_{}.pth'.format(opt.outf, opt.namefile ,epoch))
    except:
        pass

    if not opt.nbupdates is None and nb_update_network > int(opt.nbupdates):
        break

print ("end:" , datetime.datetime.now().time())
