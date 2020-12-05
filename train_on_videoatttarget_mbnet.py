import torch
from torchvision import transforms
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence

from model_mbnet import ModelSpatial
from dataset import VideoAttTarget_video
from config import *
from utils import imutils, evaluation
from lib.pytorch_convolutional_rnn import convolutional_rnn

import argparse
import os
from datetime import datetime
import shutil
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="gpu id")
parser.add_argument("--init_weights", type=str, default='mbnet_weights_12.pt', help="initial weights")
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--chunk_size", type=int, default=3, help="update every ___ frames")
parser.add_argument("--epochs", type=int, default=3, help="max number of epochs")
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
    train_dataset = VideoAttTarget_video(videoattentiontarget_train_data, videoattentiontarget_train_label,
                                          transform=transform, test=False, seq_len_limit=50)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=0)

    # Set up log dir
    logdir = os.path.join(args.log_dir,
                          datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)

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
    optimizer = torch.optim.Adam([
                        {'params': model.deconv1.parameters(), 'lr': args.lr},
                        {'params': model.deconv2.parameters(), 'lr': args.lr},
                        {'params': model.deconv3.parameters(), 'lr': args.lr},
                        {'params': model.conv4.parameters(), 'lr': args.lr},
                        {'params': model.fc_inout.parameters(), 'lr': args.lr*5},
                        ], lr = 0)

    step = 0
    loss_amp_factor = 10000  # multiplied to the loss to prevent underflow
    max_steps = len(train_loader)
    optimizer.zero_grad()

    print("Training in progress ...")
    for ep in range(args.epochs):
        for batch, (img, face, head_channel, gaze_heatmap, inout_label, lengths) in enumerate(train_loader):
            model.train(True)

            # freeze batchnorm layers
            for module in model.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    # print("BachNorm1d eval..")
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    # print("BachNorm2d eval..")
                    module.eval()
                if isinstance(module, torch.nn.modules.BatchNorm3d):
                    # print("BachNorm3d eval..")
                    module.eval()

            print("img.shape: ", img.shape)
            print("face.shape: ", face.shape)
            print("head_channel.shape: ", head_channel.shape)
            print("gaze_heatmap.shape: ", gaze_heatmap.shape)
            print("inout_label.shape: ", inout_label.shape)
            print("lengths: ", lengths)

            tmp_1 = pack_padded_sequence(img, lengths, batch_first=True)
            X_pad_data_img = tmp_1.data
            X_pad_sizes = tmp_1.batch_sizes

            tmp_2 = pack_padded_sequence(head_channel, lengths, batch_first=True)
            X_pad_data_head = tmp_2.data

            tmp_3 = pack_padded_sequence(face, lengths, batch_first=True)
            X_pad_data_face = tmp_3.data

            tmp_4 = pack_padded_sequence(gaze_heatmap, lengths, batch_first=True)
            Y_pad_data_heatmap = tmp_4.data

            tmp_5 = pack_padded_sequence(inout_label, lengths, batch_first=True)
            Y_pad_data_inout = tmp_5.data

            print("tmp_1.batch_sizes: ", tmp_1.batch_sizes)
            print("tmp_2.batch_sizes: ", tmp_2.batch_sizes)
            print("tmp_3.batch_sizes: ", tmp_3.batch_sizes)
            print("tmp_4.batch_sizes: ", tmp_4.batch_sizes)
            print("tmp_5.batch_sizes: ", tmp_5.batch_sizes)

            hx = (torch.zeros((num_lstm_layers, args.batch_size, 512, 7, 7)).cuda(device),
                  torch.zeros((num_lstm_layers, args.batch_size, 512, 7, 7)).cuda(device)) # (num_layers, batch_size, feature dims)
            print("hx[0].shape: ", hx[1].shape)
            print("hx[1].shape: ", hx[1].shape)
            last_index = 0
            previous_hx_size = args.batch_size

            for i in range(0, lengths[0], args.chunk_size):
                # In this for loop, we read batched images across the time dimension
                # we step forward N = chunk_size frames
                X_pad_sizes_slice = X_pad_sizes[i:i + args.chunk_size].cuda(device)
                curr_length = np.sum(X_pad_sizes_slice.cpu().detach().numpy())
                # curr_length = len(X_pad_sizes_slice.cpu().detach().numpy())

                print("X_pad_sizes_slice: ", X_pad_sizes_slice)
                print("curr_length: ", curr_length)

                # slice padded data
                X_pad_data_slice_img = X_pad_data_img[last_index:last_index + curr_length].cuda(device)
                X_pad_data_slice_head = X_pad_data_head[last_index:last_index + curr_length].cuda(device)
                X_pad_data_slice_face = X_pad_data_face[last_index:last_index + curr_length].cuda(device)
                Y_pad_data_slice_heatmap = Y_pad_data_heatmap[last_index:last_index + curr_length].cuda(device)
                Y_pad_data_slice_inout = Y_pad_data_inout[last_index:last_index + curr_length].cuda(device)
                last_index += curr_length

                # detach previous hidden states to stop gradient flow
                prev_hx = (hx[0][:, :min(X_pad_sizes_slice[0], previous_hx_size), :, :, :].detach(),
                           hx[1][:, :min(X_pad_sizes_slice[0], previous_hx_size), :, :, :].detach())

                # forward pass
                deconv, inout_val, hx = model(X_pad_data_slice_img, X_pad_data_slice_head, X_pad_data_slice_face, \
                                                         hidden_scene=prev_hx, batch_sizes=X_pad_sizes_slice)

                # [1] L2 loss computed only for inside case
                l2_loss = mse_loss(deconv.squeeze(1), Y_pad_data_slice_heatmap) * loss_amp_factor
                l2_loss = torch.mean(l2_loss, dim=1)
                l2_loss = torch.mean(l2_loss, dim=1)
                Y_pad_data_slice_inout = Y_pad_data_slice_inout.cuda(device).to(torch.float).squeeze()
                l2_loss = torch.mul(l2_loss, Y_pad_data_slice_inout) # zero out loss when it's outside gaze case
                l2_loss = torch.sum(l2_loss)/torch.sum(Y_pad_data_slice_inout)

                # [2] cross entropy loss for in vs out
                Xent_loss = bcelogit_loss(inout_val.squeeze(), Y_pad_data_slice_inout.squeeze())*100

                total_loss = l2_loss + Xent_loss
                total_loss.backward() # loss accumulation

                # update model parameters
                optimizer.step()
                optimizer.zero_grad()

                previous_hx_size = X_pad_sizes_slice[-1]

                step += 1

        if ep % args.save_every == 0:
            # save the model
            checkpoint = {'model': model.state_dict()}
            torch.save(checkpoint, os.path.join(logdir, 'epoch_%02d_weights.pt' % (ep+1)))


if __name__ == "__main__":
    train()
