import argparse, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from model_mbnet import ModelSpatial
# from model import ModelSpatial
from utils import imutils, evaluation
from config import *
import easydict

import torchvision.models as models
import torch.autograd.profiler as profiler

# import torch.cuda.profiler as profiler
# import pyprof
# pyprof.init()

def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)

def run():

    args = easydict.EasyDict({
        # "model_weights": "model_demo.pt",
        "model_weights": "mbnet_weights_6.pt",
        # "model_weights": "resnet_weights_3.pt",
        "image_dir": "data/demo/frames",
        "head": "data/demo/person1.txt",
        "vis_mode": "heatmap",
        "out_threshold": 100
    })

    column_names = ['frame', 'left', 'top', 'right', 'bottom']
    df = pd.read_csv(args.head, names=column_names, index_col=0)
    df['left'] -= (df['right']-df['left'])*0.1
    df['right'] += (df['right']-df['left'])*0.1
    df['top'] -= (df['bottom']-df['top'])*0.1
    df['bottom'] += (df['bottom']-df['top'])*0.1

    # set up data transformation
    test_transforms = _get_transform()

    model = ModelSpatial()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.model_weights)
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.cuda()
    model.train(False)

    print(df.index)

    hx = None

    with torch.no_grad():
        for i in df.index:
            frame_raw = Image.open(os.path.join(args.image_dir, i))
            frame_raw = frame_raw.convert('RGB')
            width, height = frame_raw.size
            # print ("frame_raw width: ", width, "height: ", height)

            head_box = [df.loc[i,'left'], df.loc[i,'top'], df.loc[i,'right'], df.loc[i,'bottom']]

            head = frame_raw.crop((head_box)) # head crop

            head = test_transforms(head) # transform inputs
            frame = test_transforms(frame_raw)
            head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], width, height,
                                                        resolution=input_resolution).unsqueeze(0)

            head = head.unsqueeze(0).cuda()
            frame = frame.unsqueeze(0).cuda()
            head_channel = head_channel.unsqueeze(0).cuda()

            # print("frame.shape: ", frame.shape)
            # print("head.shape: ", head.shape)
            # print("head_channel.shape: ", head_channel.shape)

            # forward pass
            raw_hm, _, inout = model(frame, head_channel, head)
            print("inout: ", inout)

            # heatmap modulation
            raw_hm = raw_hm.cpu().detach().numpy() * 255
            raw_hm = raw_hm.squeeze()
            # print("raw_hm shape: ", raw_hm.shape)
            # print ("raw_hm: ", raw_hm)

            inout = inout.cpu().detach().numpy()
            inout = 1 / (1 + np.exp(-inout))
            inout = (1 - inout) * 255

            # norm_map = imresize(raw_hm, (height, width)) - inout
            norm_map = np.array(Image.fromarray(raw_hm).resize((width, height))) - inout
            # print("norm_map shape: ", norm_map.shape)
            # print (norm_map)

            # vis
            # plt.close()
            # fig = plt.figure()
            # plt.axis('off')
            # plt.imshow(frame_raw)

            # ax = plt.gca()
            # rect = patches.Rectangle((head_box[0], head_box[1]), head_box[2]-head_box[0], head_box[3]-head_box[1], linewidth=2, edgecolor=(0,1,0), facecolor='none')
            # ax.add_patch(rect)

            # print ("inout: ", inout)
            # print ("out_threshold: ", args.out_threshold)

            # plt.imshow(norm_map, cmap = 'jet', alpha=0.2, vmin=0, vmax=255)

            # plt.show(block=False)

        print('DONE!')

    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=20))

if __name__ == "__main__":
    run()
