import torch
from torchvision import transforms
import torch.nn as nn

from model_mbnet import ModelSpatial
from mobilenetv2 import mobilenetv2
from dataset import GazeFollow
from config import *
from utils import imutils, evaluation

import argparse
import os
from datetime import datetime
import shutil
import numpy as np
# from scipy.misc import imresize
from PIL import Image
from tensorboardX import SummaryWriter
import warnings
import time

warnings.simplefilter(action='ignore', category=FutureWarning)


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="gpu id")
parser.add_argument("--init_weights", type=str, default="mbnet_weights_13.pt", help="initial weights")
parser.add_argument("--lr", type=float, default=2.5e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=48, help="batch size")
parser.add_argument("--epochs", type=int, default=70, help="number of epochs")
parser.add_argument("--print_every", type=int, default=25, help="print every ___ iterations")
parser.add_argument("--eval_every", type=int, default=1000, help="evaluate every ___ iterations")
parser.add_argument("--save_every", type=int, default=1, help="save every ___ epochs")
parser.add_argument("--log_dir", type=str, default="logs", help="directory to save log files")
args = parser.parse_args()


def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)


def train():
    transform = _get_transform()

    # Prepare data
    print("Loading Data")
    train_dataset = GazeFollow(gazefollow_train_data, gazefollow_train_label,
                      transform, input_size=input_resolution, output_size=output_resolution)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=0)

    val_dataset = GazeFollow(gazefollow_val_data, gazefollow_val_label,
                      transform, input_size=input_resolution, output_size=output_resolution, test=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=0)

    # Set up log dir
    logdir = os.path.join(args.log_dir, 'train_13')
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)

    writer = SummaryWriter(logdir)
    np.random.seed(1)

    # Define device
    device = torch.device('cuda', args.device)

    # Load model
    print("Constructing model")
    print("Loading init_weights ", args.init_weights)
    model = ModelSpatial()
    model.cuda().to(device)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.init_weights)
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # Loss functions
    mse_loss = nn.MSELoss(reduce=False) # not reducing in order to ignore outside cases
    bcelogit_loss = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    step = 0
    loss_amp_factor = 10000 # multiplied to the loss to prevent underflow
    max_steps = len(train_loader)
    optimizer.zero_grad()

    print("Training in progress ...")
    for ep in range(args.epochs):
        for batch, (img, face, head_channel, gaze_heatmap, name, gaze_inside) in enumerate(train_loader):
            model.train(True)
            images = img.cuda().to(device)
            head = head_channel.cuda().to(device)
            faces = face.cuda().to(device)
            gaze_heatmap = gaze_heatmap.cuda().to(device)

            gaze_heatmap_pred, attmap, inout_pred = model(images, head, faces)
            # gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)

            # [1] L2 loss computed only for inside case
            l2_loss = 0

            gaze_inside = gaze_inside.cuda(device).to(torch.float)

            # [2] cross entropy loss for in vs out
            Xent_loss = bcelogit_loss(inout_pred.squeeze(), gaze_inside.squeeze()) * 100

            total_loss = Xent_loss
            # NOTE: summed loss is used to train the main model.
            # l2_loss is used to get SOTA on GazeFollow benchmark.
            total_loss.backward() # loss accumulation

            optimizer.step()
            optimizer.zero_grad()

            step += 1

            if batch % args.print_every == 0:
                print(time.strftime('%c', time.localtime(time.time())))
                print("Epoch:{:04d}\tstep:{:06d}/{:06d}\ttraining loss: (l2){:.4f} (Xent){:.4f}".format(ep, batch+1, max_steps, l2_loss, Xent_loss))

        if ep % args.save_every == 0:
            # save the model
            checkpoint = {'model': model.state_dict()}
            torch.save(checkpoint, os.path.join(logdir, 'epoch_%02d_weights.pt' % (ep+1)))


if __name__ == "__main__":
    train()
