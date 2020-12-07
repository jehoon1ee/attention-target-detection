import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
import math
from lib.pytorch_convolutional_rnn import convolutional_rnn
import numpy as np

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

##################################################################
##################################################################
##################################################################

def dwise_conv(ch_in, stride=1):
    return (
        nn.Sequential(
            #depthwise
            nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, stride=stride, groups=ch_in, bias=False),
            nn.BatchNorm2d(ch_in),
            nn.ReLU6(inplace=True),
        )
    )

def conv1x1(ch_in, ch_out):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )

def conv3x3(ch_in, ch_out, stride):
    return (
        nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU6(inplace=True)
        )
    )

class InvertedBlock(nn.Module):
    def __init__(self, ch_in, ch_out, expand_ratio, stride):
        super(InvertedBlock, self).__init__()

        self.stride = stride
        assert stride in [1,2]

        hidden_dim = ch_in * expand_ratio

        self.use_res_connect = self.stride==1 and ch_in==ch_out

        layers = []
        if expand_ratio != 1:
            layers.append(conv1x1(ch_in, hidden_dim))
        layers.extend([
            dwise_conv(hidden_dim, stride=stride), #dw
            conv1x1(hidden_dim, ch_out) #pw
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

        self.stem_conv = conv3x3(ch_in, 32, stride=2)

        layers = []
        input_channel = 32
        for t, c, n, s in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedBlock(ch_in=input_channel, ch_out=c, expand_ratio=t, stride=stride))
                input_channel = c

        self.layers = nn.Sequential(*layers)
        self.last_conv = conv1x1(input_channel, 1024)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        return x

##################################################################
##################################################################
##################################################################

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

        # scene pathway
        self.conv1_scene = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_scene = nn.BatchNorm2d(64)
        self.layer1_scene = self._make_layer_scene(block, 64, layers_scene[0])
        self.layer2_scene = self._make_layer_scene(block, 128, layers_scene[1], stride=2)
        self.layer3_scene = self._make_layer_scene(block, 256, layers_scene[2], stride=2)
        self.layer4_scene = self._make_layer_scene(block, 512, layers_scene[3], stride=2)
        self.layer5_scene = self._make_layer_scene(block, 256, layers_scene[4], stride=1) # additional to resnet50

        # face pathway
        self.conv1_face = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1_face = nn.BatchNorm2d(64)
        self.layer1_face = self._make_layer_face(block, 64, layers_face[0])
        self.layer2_face = self._make_layer_face(block, 128, layers_face[1], stride=2)
        self.layer3_face = self._make_layer_face(block, 256, layers_face[2], stride=2)
        self.layer4_face = self._make_layer_face(block, 512, layers_face[3], stride=2)
        self.layer5_face = self._make_layer_face(block, 256, layers_face[4], stride=1) # additional to resnet50

        # mobilenetv2
        # mbnet_layers = []
        # mbnet_layers.append(MobileNetV2(ch_in=3))
        # self.mbnet = nn.Sequential(*mbnet_layers)
        #
        # mbnet2_layers = []
        # mbnet2_layers.append(MobileNetV2(ch_in=4))
        # self.mbnet2 = nn.Sequential(*mbnet2_layers)

        # attention
        self.attn = nn.Linear(1808, 1*7*7)

        # encoding for saliency
        self.compress_conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)

        # encoding for in/out
        self.compress_conv1_inout = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1_inout = nn.BatchNorm2d(512)
        self.compress_conv2_inout = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2_inout = nn.BatchNorm2d(1)
        self.fc_inout = nn.Linear(49, 1)

        # decoding
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

    def _make_layer_scene(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_scene != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_scene, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_scene, planes, stride, downsample))
        self.inplanes_scene = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_scene, planes))

        return nn.Sequential(*layers)

    def _make_layer_face(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_face != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_face, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_face, planes, stride, downsample))
        self.inplanes_face = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_face, planes))

        return nn.Sequential(*layers)

    def forward(self, images, head, face):
        # print("images.shape: ", images.shape) # [48, 3, 224 ,244]
        # print("head.shape: ", head.shape) # [48, 1, 224, 224]
        # print("face.shape: ", face.shape) # [48, 3, 224, 224]

        # Head Conv mbnet
        # face_feat = self.mbnet(face)

        # Head Conv
        with profiler.profile(use_cuda=True) as prof:
            face = self.conv1_face(face)
            face = self.bn1_face(face)
            face = self.relu(face)
            face = self.maxpool(face)
            face = self.layer1_face(face)
            face = self.layer2_face(face)
            face = self.layer3_face(face)
            face = self.layer4_face(face)
            face_feat = self.layer5_face(face)
        print(prof.key_averages().table(sort_by="cpu_time_total"))
        # print("face_feat.shape: ", face_feat.shape) # [48, 1024, 7, 7]

        with profiler.profile(use_cuda=True) as prof:
            # reduce head channel size by max pooling: (N, 1, 224, 224) -> (N, 1, 28, 28)
            head_reduced = self.maxpool(self.maxpool(self.maxpool(head))).view(-1, 784)
            # print("head_reduced.shape: ", head_reduced.shape) # [48, 784]

            # reduce face feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
            face_feat_reduced = self.avgpool(face_feat).view(-1, 1024)

            # get and reshape attention weights such that it can be multiplied with scene feature map
            attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))
            attn_weights = attn_weights.view(-1, 1, 49)
            attn_weights = F.softmax(attn_weights, dim=2) # soft attention weights single-channel
            attn_weights = attn_weights.view(-1, 1, 7, 7)

            # Scene Conv
            im = torch.cat((images, head), dim=1)
            # scene_feat = self.mbnet2(im)
        print(prof.key_averages().table(sort_by="cpu_time_total"))

        with profiler.profile(use_cuda=True) as prof:
            im = self.conv1_scene(im)
            im = self.bn1_scene(im)
            im = self.relu(im)
            im = self.maxpool(im)
            im = self.layer1_scene(im)
            im = self.layer2_scene(im)
            im = self.layer3_scene(im)
            im = self.layer4_scene(im)
            scene_feat = self.layer5_scene(im)
        print(prof.key_averages().table(sort_by="cpu_time_total"))
        # print("scene_feat.shape: ", scene_feat.shape) # [48, 1024, 7, 7]
        # attn_weights = torch.ones(attn_weights.shape)/49.0

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





class BottleneckConvLSTM(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckConvLSTM, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.bn_ds = nn.BatchNorm2d(planes * self.expansion)

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # RW edit: handles batch_size==1
        if out.shape[0] > 1:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # RW edit: handles batch_size==1
        if out.shape[0] > 1:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # RW edit: handles batch_size==1
        if out.shape[0] > 1:
            out = self.bn3(out)

        if self.downsample is not None:
            # RW edit: handles batch_size==1
            if out.shape[0] > 1:
                residual = self.downsample(x)
                residual = self.bn_ds(residual)
            else:
                residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class ModelSpatioTemporal(nn.Module):
    # Define a ResNet 50-ish arch
    def __init__(self, block=BottleneckConvLSTM, num_lstm_layers = 1, bidirectional = False, layers_scene = [3, 4, 6, 3, 2], layers_face = [3, 4, 6, 3, 2]):
        # Resnet Feature Extractor
        self.inplanes_scene = 64
        self.inplanes_face = 64
        super(ModelSpatioTemporal, self).__init__()
        # common
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        # scene pathway
        self.conv1_scene = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_scene = nn.BatchNorm2d(64)
        self.layer1_scene = self._make_layer_scene(block, 64, layers_scene[0])
        self.layer2_scene = self._make_layer_scene(block, 128, layers_scene[1], stride=2)
        self.layer3_scene = self._make_layer_scene(block, 256, layers_scene[2], stride=2)
        self.layer4_scene = self._make_layer_scene(block, 512, layers_scene[3], stride=2)
        self.layer5_scene = self._make_layer_scene(block, 256, layers_scene[4], stride=1) # additional to resnet50

        # face pathway
        self.conv1_face = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1_face = nn.BatchNorm2d(64)
        self.layer1_face = self._make_layer_face(block, 64, layers_face[0])
        self.layer2_face = self._make_layer_face(block, 128, layers_face[1], stride=2)
        self.layer3_face = self._make_layer_face(block, 256, layers_face[2], stride=2)
        self.layer4_face = self._make_layer_face(block, 512, layers_face[3], stride=2)
        self.layer5_face = self._make_layer_face(block, 256, layers_face[4], stride=1) # additional to resnet50

        # attention
        self.attn = nn.Linear(1808, 1*7*7)

        # encoding for saliency
        self.compress_conv1 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1 = nn.BatchNorm2d(1024)
        self.compress_conv2 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2 = nn.BatchNorm2d(512)

        # encoding for in/out
        self.compress_conv1_inout = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn1_inout = nn.BatchNorm2d(512)
        self.compress_conv2_inout = nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.compress_bn2_inout = nn.BatchNorm2d(1)
        self.fc_inout = nn.Linear(49, 1)

        self.convlstm_scene = convolutional_rnn.Conv2dLSTM(in_channels=512,
                                                     out_channels=512,
                                                     kernel_size=3,
                                                     num_layers=num_lstm_layers,
                                                     bidirectional=bidirectional,
                                                     batch_first=True,
                                                     stride=1,
                                                     dropout=0.5)

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

    def _make_layer_scene(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_scene != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_scene, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_scene, planes, stride, downsample))
        self.inplanes_scene = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_scene, planes))

        return nn.Sequential(*layers)

    def _make_layer_face(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes_face != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes_face, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_face, planes, stride, downsample))
        self.inplanes_face = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_face, planes))

        return nn.Sequential(*layers)

    def forward(self, images, head, face, hidden_scene: tuple = None, batch_sizes: list = None):
        print("images.shape: ", images.shape)
        print("head.shape: ", head.shape)
        print("face.shape: ", face.shape)
        print("batch_sizes: ", batch_sizes)
        face = self.conv1_face(face)
        face = self.bn1_face(face)
        face = self.relu(face)
        face = self.maxpool(face)
        face = self.layer1_face(face)
        face = self.layer2_face(face)
        face = self.layer3_face(face)
        face = self.layer4_face(face)
        face_feat = self.layer5_face(face)

        # reduce head channel size by max pooling: (N, 1, 224, 224) -> (N, 1, 28, 28)
        head_reduced = self.maxpool(self.maxpool(self.maxpool(head))).view(-1, 784)
        # reduce face feature size by avg pooling: (N, 1024, 7, 7) -> (N, 1024, 1, 1)
        face_feat_reduced = self.avgpool(face_feat).view(-1, 1024)
        # get and reshape attention weights such that it can be multiplied with scene feature map
        attn_weights = self.attn(torch.cat((head_reduced, face_feat_reduced), 1))
        attn_weights = attn_weights.view(-1, 1, 49)
        attn_weights = F.softmax(attn_weights, dim=2) # soft attention weights single-channel
        attn_weights = attn_weights.view(-1, 1, 7, 7)

        im = torch.cat((images, head), dim=1)
        im = self.conv1_scene(im)
        im = self.bn1_scene(im)
        im = self.relu(im)
        im = self.maxpool(im)
        im = self.layer1_scene(im)
        im = self.layer2_scene(im)
        im = self.layer3_scene(im)
        im = self.layer4_scene(im)
        scene_feat = self.layer5_scene(im)
        attn_applied_scene_feat = torch.mul(attn_weights, scene_feat) # (N, 1, 7, 7) # applying attention weights on scene feat

        scene_face_feat = torch.cat((attn_applied_scene_feat, face_feat), 1)

        # scene + face feat -> in/out
        encoding_inout = self.compress_conv1_inout(scene_face_feat)
        encoding_inout = self.compress_bn1_inout(encoding_inout)
        encoding_inout = self.relu(encoding_inout)
        encoding_inout = self.compress_conv2_inout(encoding_inout)
        encoding_inout = self.compress_bn2_inout(encoding_inout)
        encoding_inout = self.relu(encoding_inout)

        # scene + face feat -> encoding -> decoding
        encoding = self.compress_conv1(scene_face_feat)
        encoding = self.compress_bn1(encoding)
        encoding = self.relu(encoding)
        encoding = self.compress_conv2(encoding)
        encoding = self.compress_bn2(encoding)
        encoding = self.relu(encoding)
        print("encoding shape: ", encoding.shape)

        # RW edit: x should be of shape (size, channel, width, height)
        # x_pad = PackedSequence(encoding, batch_sizes)
        # x_pad = pack_padded_sequence(encoding, bach_sizes.cpu(), batch_first=True)
        # print("len(x_pad): ", len(x_pad))
        # print("type(x_pad): ", type(x_pad))

        row_1 = 0
        row_2 = 0
        batch_sizes_len = len(batch_sizes)
        print("batch_sizes_len: ", batch_sizes_len)

        if (batch_sizes[0] > 0):
            tmp_1 = encoding[0:0+batch_sizes[0]]

            if (batch_sizes[0] == 1):
                row_1 = row_1 + 1
            elif (batch_sizes[0] == 2):
                row_1 = row_1 + 1
                row_2 = row_2 + 1

            if (batch_sizes_len >= 2) and (batch_sizes[1] > 0):
                tmp_2 = encoding[0+batch_sizes[0]:0+batch_sizes[0]+batch_sizes[1]]
                if (batch_sizes[1] == 1):
                    row_1 = row_1 + 1
                elif (batch_sizes[1] == 2):
                    row_1 = row_1 + 1
                    row_2 = row_2 + 1

                if (batch_sizes_len >= 3) and (batch_sizes[2] > 0):
                    tmp_3 = encoding[0+batch_sizes[0]+batch_sizes[1]:]
                    if (batch_sizes[2] == 1):
                        row_1 = row_1 + 1
                    elif (batch_sizes[2] == 2):
                        row_1 = row_1 + 1
                        row_2 = row_2 + 1

        # print("tmp_1.shape: ", tmp_1.shape)
        # print("tmp_2.shape: ", tmp_2.shape)
        # print("tmp_3.shape: ", tmp_3.shape)

        if (batch_sizes_len == 1):
            tmp_4 = tmp_1
        elif (batch_sizes_len == 2):
            tmp_4 = torch.stack([tmp_1, tmp_2], dim = 1)
        elif (batch_sizes_len == 3):
            tmp_4 = torch.stack([tmp_1, tmp_2, tmp_3], dim = 1)

        print("tmp_4.shape: ", tmp_4.shape)
        # tmp_5 = torch.transpose(tmp_4, 0, 1)
        tmp_5 = tmp_4
        print("tmp_5.shape: ", tmp_5.shape)
        x_pad = pack_padded_sequence(tmp_5, [row_1, row_2], batch_first=True)

        y, hx = self.convlstm_scene(x_pad, hx=hidden_scene)
        deconv = y.data

        inout_val = encoding_inout.view(-1, 49)
        inout_val = self.fc_inout(inout_val)

        deconv = self.deconv1(deconv)
        if encoding.shape[0] > 1:
            deconv = self.deconv_bn1(deconv)
        deconv = self.relu(deconv)
        deconv = self.deconv2(deconv)
        if encoding.shape[0] > 1:
            deconv = self.deconv_bn2(deconv)
        deconv = self.relu(deconv)
        deconv = self.deconv3(deconv)
        if encoding.shape[0] > 1:
            deconv = self.deconv_bn3(deconv)
        deconv = self.relu(deconv)
        deconv = self.conv4(deconv)

        return deconv, inout_val, hx
