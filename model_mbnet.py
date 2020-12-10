import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
import math
from lib.pytorch_convolutional_rnn import convolutional_rnn
import numpy as np
from mobilenetv2 import MobileNetV2

# pytorch profiler
import torchvision.models as models
import torch.autograd.profiler as profiler


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def dwise_conv(ch_in, stride=1):
    return (
        nn.Sequential(
            #depthwise
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=stride, padding=1, groups=ch_in, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU6(inplace=True),
        )
    )

def conv1x1(ch_in, ch_out):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )

def conv1x1_nonlinear(ch_in, ch_out):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ch_out)
        )
    )

def conv3x3(ch_in, ch_out, stride):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )

def conv7x7_nonlinear(ch_in, ch_out, stride):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm2d(ch_out)
        )
    )

class InvertedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride):
        super(InvertedBlock, self).__init__()

        self.stride = stride
        assert stride in [1,2]
        hidden_dim = ch_in * expand_ratio
        self.use_res_connect = self.stride == 1 and ch_in == ch_out

        layers = []
        if expand_ratio != 1:
            layers.append(conv1x1(ch_in, hidden_dim))
        layers.extend([
            dwise_conv(hidden_dim, stride=stride), #dw
            conv1x1_nonlinear(hidden_dim, ch_out) #pw
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)

class MobileNetV2(nn.Module):
    def __init__(self, ch_in=3, n_classes=1000):
        super(MobileNetV2, self).__init__()

        self.configs=[
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        self.stem_conv = conv7x7_nonlinear(ch_in, 64, stride=2)
        # self.stem_conv = conv3x3(ch_in, 32, stride=2)

        layers = []
        input_channel = 64
        # input_channel = 32
        for t, c, n, s in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, stride=stride))
                input_channel = c

        self.layers = nn.Sequential(*layers)
        self.last_conv = conv1x1(input_channel, 1024)
        self._initialize_weights()

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ModelSpatial(nn.Module):
    # Define a ResNet 50-ish arch
    def __init__(self, block = Bottleneck, layers_scene = [3, 4, 6, 3, 2], layers_face = [3, 4, 6, 3, 2]):
        # Resnet Feature Extractor
        self.inplanes_scene = 64
        self.inplanes_face = 64
        super(ModelSpatial, self).__init__()
        # common
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        mbnet_head_layers = []
        mbnet_head_layers.append(MobileNetV2(ch_in=3))
        self.mbnet_head_conv = nn.Sequential(*mbnet_head_layers)

        mbnet_scene_layers = []
        mbnet_scene_layers.append(MobileNetV2(ch_in=4))
        self.mbnet_scene_conv = nn.Sequential(*mbnet_scene_layers)

        # attention
        self.attn = nn.Linear(1808, 1*7*7)

        # In Frame?
        self.compress_conv1_inout = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1_inout = nn.BatchNorm2d(1024)
        self.compress_conv2_inout = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2_inout = nn.BatchNorm2d(512)
        self.compress_conv3_inout = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn3_inout = nn.BatchNorm2d(256)
        self.compress_conv4_inout = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn4_inout = nn.BatchNorm2d(1)
        self.fc_inout = nn.Linear(49, 1)

        # Encode: saliency
        self.compress_conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)

        # Deconv
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.deconv_bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.deconv_bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2)
        self.deconv_bn3 = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, kernel_size=1, stride=1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, images, head, face):
        # Head Conv mbnet
        with profiler.profile(use_cuda=True) as prof:
            face_feat = self.mbnet_head_conv(face)
        print(prof.key_averages().table(sort_by="cpu_time_total"))

        with profiler.profile(use_cuda=True) as prof:
            # reduce face feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
            face_feat_reduced = self.avgpool(face_feat).view(-1, 1024)

            # reduce head channel size by max pooling: (N, 1, 224, 224) -> (N, 1, 28, 28)
            head_reduced = self.maxpool(self.maxpool(self.maxpool(head))).view(-1, 784)

            # get and reshape attention weights such that it can be multiplied with scene feature map
            attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))
            attn_weights = attn_weights.view(-1, 1, 49)
            attn_weights = F.softmax(attn_weights, dim=2) # soft attention weights single-channel
            attn_weights = attn_weights.view(-1, 1, 7, 7)

            # Scene Conv
            im = torch.cat((images, head), dim=1)
        print(prof.key_averages().table(sort_by="cpu_time_total"))

        with profiler.profile(use_cuda=True) as prof:
            scene_feat = self.mbnet_scene_conv(im) # torch.Size([48, 1024, 7, 7])
        print(prof.key_averages().table(sort_by="cpu_time_total"))

        with profiler.profile(use_cuda=True) as prof:
            attn_applied_scene_feat = torch.mul(attn_weights, scene_feat) # (N, 1, 7, 7) # applying attention weights on scene feat

            scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)

            # In Frame?: scene + face feat -> in/out
            encoding_inout = self.compress_conv1_inout(scene_face_feat)
            encoding_inout = self.compress_bn1_inout(encoding_inout)
            encoding_inout = self.relu(encoding_inout)
            encoding_inout = self.compress_conv2_inout(encoding_inout)
            encoding_inout = self.compress_bn2_inout(encoding_inout)
            encoding_inout = self.relu(encoding_inout)
            encoding_inout = self.compress_conv3_inout(encoding_inout)
            encoding_inout = self.compress_bn3_inout(encoding_inout)
            encoding_inout = self.relu(encoding_inout)
            encoding_inout = self.compress_conv4_inout(encoding_inout)
            encoding_inout = self.compress_bn4_inout(encoding_inout)
            encoding_inout = self.relu(encoding_inout)
            encoding_inout = encoding_inout.view(-1, 49)
            encoding_inout = self.fc_inout(encoding_inout)

            # Encode: scene + face feat -> encoding -> decoding
            encoding = self.compress_conv1(scene_face_feat)
            encoding = self.compress_bn1(encoding)
            encoding = self.relu(encoding)
            encoding = self.compress_conv2(encoding)
            encoding = self.compress_bn2(encoding)
            encoding = self.relu(encoding)

            # Decode
            gaze_heatmap_pred = []
            gaze_heatmap_pred = self.deconv1(encoding)
            gaze_heatmap_pred = self.deconv_bn1(gaze_heatmap_pred)
            gaze_heatmap_pred = self.relu(gaze_heatmap_pred)
            gaze_heatmap_pred = self.deconv2(gaze_heatmap_pred)
            gaze_heatmap_pred = self.deconv_bn2(gaze_heatmap_pred)
            gaze_heatmap_pred = self.relu(gaze_heatmap_pred)
            gaze_heatmap_pred = self.deconv3(gaze_heatmap_pred)
            gaze_heatmap_pred = self.deconv_bn3(gaze_heatmap_pred)
            gaze_heatmap_pred = self.relu(gaze_heatmap_pred)
            gaze_heatmap_pred = self.conv4(gaze_heatmap_pred)
        print(prof.key_averages().table(sort_by="cpu_time_total"))

        return gaze_heatmap_pred, torch.mean(attn_weights, 1, keepdim=True), encoding_inout
