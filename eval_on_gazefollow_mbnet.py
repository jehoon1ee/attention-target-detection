import torch
from torchvision import transforms
import torch.nn as nn

from model_mbnet import ModelSpatial
from dataset import GazeFollow
from config import *
from utils import imutils, evaluation

import argparse
import os
import numpy as np
from PIL import Image
import warnings
import sys

warnings.simplefilter(action='ignore', category=FutureWarning)


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="gpu id")
parser.add_argument("--model_weights", type=str, default="mbnet_weights.pt", help="model weights")
parser.add_argument("--batch_size", type=int, default=48, help="batch size")
args = parser.parse_args()


def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)


def test():
    transform = _get_transform()

    # Prepare data
    print("Loading Data")
    val_dataset = GazeFollow(gazefollow_val_data, gazefollow_val_label,
                      transform, input_size=input_resolution, output_size=output_resolution, test=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             num_workers=0)

    # Define device
    device = torch.device('cuda', args.device)

    # Load model
    print("Constructing model")
    print("Loading init_weights ", args.model_weights)
    model = ModelSpatial()
    model.cuda().to(device)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.model_weights)
    pretrained_dict = pretrained_dict['model']
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print('Evaluation in progress ...')
    model.train(False)
    AUC = []; min_dist = []; avg_dist = []

    np.set_printoptions(threshold=sys.maxsize)

    with torch.no_grad():
        for val_batch, (val_img, val_face, val_head_channel, val_gaze_heatmap, cont_gaze, imsize, _) in enumerate(val_loader):
            print("val_batch: ", val_batch)

            val_images = val_img.cuda().to(device)
            val_faces = val_face.cuda().to(device)
            val_head = val_head_channel.cuda().to(device)
            val_gaze_heatmap = val_gaze_heatmap.cuda().to(device)

            val_gaze_heatmap_pred, val_attmap, val_inout_pred = model(val_images, val_head, val_faces)
            val_gaze_heatmap_pred = val_gaze_heatmap_pred.squeeze(1)

            if (val_batch == 0):
                print("val_images.shape: ", val_images.shape)
                print("val_head.shape: ", val_head.shape)
                print("val_faces.shape: ", val_faces.shape)
                print("val_gaze_heatmap.shape: ", val_gaze_heatmap.shape)
                print("cont_gaze.shape: ", cont_gaze.shape)
                print("imsize.shape: ", imsize.shape)
                print("val_gaze_heatmap_pred.shape: ", val_gaze_heatmap_pred.shape)
                print("val_attmap.shape: ", val_attmap.shape)
                print("val_inout_pred.shape: ", val_inout_pred.shape)

            # go through each data point and record AUC, min dist, avg dist
            for b_i in range(len(cont_gaze)):

                # remove padding and recover valid ground truth points
                valid_gaze = cont_gaze[b_i]
                if (val_batch == 0):
                    print("before view() valid_gaze.shape: ", valid_gaze.shape)
                valid_gaze = valid_gaze[valid_gaze != -1].view(-1,2)
                if (val_batch == 0):
                    print("after view() valid_gaze.shape: ", valid_gaze.shape)
                # AUC: area under curve of ROC
                multi_hot = imutils.multi_hot_targets(cont_gaze[b_i], imsize[b_i])
                if (val_batch == 0):
                    print("multi_hot.shape: ", multi_hot.shape)
                    print("imsize[b_i]: ", imsize[b_i])

                # [1] auc
                tmp1 = imsize[b_i][0].item()
                tmp2 = imsize[b_i][1].item()
                scaled_heatmap = np.array(Image.fromarray(val_gaze_heatmap_pred[b_i].cpu().detach().numpy()).resize((tmp1, tmp2), Image.BILINEAR))
                if (val_batch == 0):
                    print("scaled_heatmap.shape: ", scaled_heatmap.shape)
                auc_score = evaluation.auc(scaled_heatmap, multi_hot)
                AUC.append(auc_score)

                # [2] min distance: minimum among all possible pairs of <ground truth point, predicted point>
                pred_x, pred_y = evaluation.argmax_pts(val_gaze_heatmap_pred[b_i].cpu().detach().numpy())
                norm_p = [pred_x/float(output_resolution), pred_y/float(output_resolution)]
                all_distances = []
                for gt_gaze in valid_gaze:
                    all_distances.append(evaluation.L2_dist(gt_gaze, norm_p))
                min_dist.append(min(all_distances))

                # [3] average distance: distance between the predicted point and human average point
                mean_gt_gaze = torch.mean(valid_gaze, 0)
                avg_distance = evaluation.L2_dist(mean_gt_gaze, norm_p)
                avg_dist.append(avg_distance)

    print("\tAUC:{:.4f}\tmin dist:{:.4f}\tavg dist:{:.4f}".format(
          torch.mean(torch.tensor(AUC)),
          torch.mean(torch.tensor(min_dist)),
          torch.mean(torch.tensor(avg_dist))))


if __name__ == "__main__":
    test()
