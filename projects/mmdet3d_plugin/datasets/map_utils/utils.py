"""
Utility functions and default settings

Author: Yuliang Guo (33yuliangguo@gmail.com)
Date: March, 2020
"""

import argparse
import errno
import os
import sys

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.init as init
import torch.optim
from torch.optim import lr_scheduler
import os.path as ops
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
plt.rcParams['figure.figsize'] = (35, 30)


def define_args():
    parser = argparse.ArgumentParser(description='Lane_detection_all_objectives')
    # Paths settings
    parser.add_argument('--dataset_name', type=str, help='the dataset name to be used in saving model names')
    parser.add_argument('--data_dir', type=str, help='The path saving train.json and val.json files')
    parser.add_argument('--dataset_dir', type=str, help='The path saving actual data')
    parser.add_argument('--save_path', type=str, default='data_splits/', help='directory to save output')
    # Dataset settings
    parser.add_argument('--org_h', type=int, default=1080, help='height of the original image')
    parser.add_argument('--org_w', type=int, default=1920, help='width of the original image')
    parser.add_argument('--crop_y', type=int, default=0, help='crop from image')
    parser.add_argument('--cam_height', type=float, default=1.55, help='height of camera in meters')
    parser.add_argument('--pitch', type=float, default=3, help='pitch angle of camera to ground in centi degree')
    parser.add_argument('--fix_cam', type=str2bool, nargs='?', const=True, default=False, help='if to use fix camera')
    parser.add_argument('--no_3d', action='store_true', help='if a dataset include laneline 3D attributes')
    parser.add_argument('--no_centerline', action='store_true', help='if a dataset include centerline')
    # 3DLaneNet settings
    parser.add_argument('--mod', type=str, default='3DLaneNet', help='model to train')
    parser.add_argument("--pretrained", type=str2bool, nargs='?', const=True, default=True, help="use pretrained vgg model")
    parser.add_argument("--batch_norm", type=str2bool, nargs='?', const=True, default=True, help="apply batch norm")
    parser.add_argument("--pred_cam", type=str2bool, nargs='?', const=True, default=False, help="use network to predict camera online?")
    parser.add_argument('--ipm_h', type=int, default=208, help='height of inverse projective map (IPM)')
    parser.add_argument('--ipm_w', type=int, default=128, help='width of inverse projective map (IPM)')
    parser.add_argument('--resize_h', type=int, default=360, help='height of the original image')
    parser.add_argument('--resize_w', type=int, default=480, help='width of the original image')
    parser.add_argument('--y_ref', type=float, default=20.0, help='the reference Y distance in meters from where lane association is determined')
    parser.add_argument('--prob_th', type=float, default=0.5, help='probability threshold for selecting output lanes')
    # General model settings
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--nepochs', type=int, default=30, help='total numbers of epochs')
    parser.add_argument('--learning_rate', type=float, default=5*1e-4, help='learning rate')
    parser.add_argument('--no_cuda', action='store_true', help='if gpu available')
    parser.add_argument('--nworkers', type=int, default=0, help='num of threads')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout in network')
    parser.add_argument('--pretrain_epochs', type=int, default=20, help='Number of epochs to perform segmentation pretraining')
    parser.add_argument('--channels_in', type=int, default=3, help='num channels of input image')
    parser.add_argument('--flip_on', action='store_true', help='Random flip input images on?')
    parser.add_argument('--test_mode', action='store_true', help='prevents loading latest saved model')
    parser.add_argument('--start_epoch', type=int, default=0, help='prevents loading latest saved model')
    parser.add_argument('--evaluate', action='store_true', help='only perform evaluation')
    parser.add_argument('--resume', type=str, default='', help='resume latest saved run')
    parser.add_argument('--vgg_mean', type=float, default=[0.485, 0.456, 0.406], help='Mean of rgb used in pretrained model on ImageNet')
    parser.add_argument('--vgg_std', type=float, default=[0.229, 0.224, 0.225], help='Std of rgb used in pretrained model on ImageNet')
    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='adam', help='adam or sgd')
    parser.add_argument('--weight_init', type=str, default='normal', help='normal, xavier, kaiming, orhtogonal weights initialisation')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 weight decay/regularisation on?')
    parser.add_argument('--lr_decay', action='store_true', help='decay learning rate with rule')
    parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=400, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--lr_policy', default=None, help='learning rate policy: lambda|step|plateau')
    parser.add_argument('--lr_decay_iters', type=int, default=30, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--clip_grad_norm', type=int, default=0, help='performs gradient clipping')
    # CUDNN usage
    parser.add_argument("--cudnn", type=str2bool, nargs='?', const=True, default=True, help="cudnn optimization active")
    # Tensorboard settings
    parser.add_argument("--no_tb", type=str2bool, nargs='?', const=True, default=False, help="Use tensorboard logging by tensorflow")
    # Print settings
    parser.add_argument('--print_freq', type=int, default=500, help='padding')
    parser.add_argument('--save_freq', type=int, default=500, help='padding')
    # Skip batch
    parser.add_argument('--list', type=int, nargs='+', default=[954, 2789], help='Images you want to skip')

    return parser


def tusimple_config(args):

    # set dataset parameters
    args.org_h = 720
    args.org_w = 1280
    args.crop_y = 80
    args.no_centerline = True
    args.no_3d = True
    args.fix_cam = True
    args.pred_cam = False

    # set camera parameters for the test dataset
    args.K = np.array([[1000, 0, 640],
                       [0, 1000, 400],
                       [0, 0, 1]])
    args.cam_height = 1.6
    args.pitch = 9

    # specify model settings
    """
    paper presented params:
        args.top_view_region = np.array([[-10, 85], [10, 85], [-10, 5], [10, 5]])
        args.anchor_y_steps = np.array([5, 20, 40, 60, 80, 100])
    """
    # args.top_view_region = np.array([[-10, 82], [10, 82], [-10, 2], [10, 2]])
    # args.anchor_y_steps = np.array([2, 3, 5, 10, 15, 20, 30, 40, 60, 80])
    args.top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
    args.anchor_y_steps = np.array([5, 10, 15, 20, 30, 40, 50, 60, 80, 100])
    args.num_y_steps = len(args.anchor_y_steps)

    # initialize with pre-trained vgg weights
    args.pretrained = False
    # apply batch norm in network
    args.batch_norm = True


def sim3d_config(args):

    # set dataset parameters
    args.org_h = 1080
    args.org_w = 1920
    args.crop_y = 0
    args.no_centerline = False
    args.no_3d = False
    args.fix_cam = False
    args.pred_cam = False

    # set camera parameters for the test datasets
    args.K = np.array([[2015., 0., 960.],
                       [0., 2015., 540.],
                       [0., 0., 1.]])

    # specify model settings
    """
    paper presented params:
        args.top_view_region = np.array([[-10, 85], [10, 85], [-10, 5], [10, 5]])
        args.anchor_y_steps = np.array([5, 20, 40, 60, 80, 100])
    """
    # args.top_view_region = np.array([[-10, 83], [10, 83], [-10, 3], [10, 3]])
    # args.anchor_y_steps = np.array([3, 5, 10, 20, 40, 60, 80, 100])
    args.top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
    args.anchor_y_steps = np.array([5, 10, 15, 20, 30, 40, 50, 60, 80, 100])
    args.num_y_steps = len(args.anchor_y_steps)

    # initialize with pre-trained vgg weights
    args.pretrained = False
    # apply batch norm in network
    args.batch_norm = True

def draw_laneline(img, ax1, ax2, ax3, lanelines, color, P_g2im, vis_2d=False, vis=None, interp=False, scatter=False):
    # lanelines: list, [N, 3]
    # draw 2d
    # rgb_color = [color[0] * 255, color[1] * 255, color[2] * 255]
    rgb_color = [int(0.8 * 255), int(0.2 * 255), int(0.1 * 255)]
    for idx, lane in enumerate(lanelines):
        if vis is not None:
            lane[:] = lane[:, vis[idx]>0.5]
        x = lane[:, 0]
        y = lane[:, 1]
        z = lane[:, 2]
        if x.shape[0] == 0 or y.shape[0] == 0 or z.shape[0] == 0:
            continue
        fit1 = np.polyfit(y, x, 2)
        fit2 = np.polyfit(y, z, 2)
        f_xy = np.poly1d(fit1)
        f_zy = np.poly1d(fit2)
        if interp:
            y = np.linspace(min(y), max(y), 2 * len(y))
            x = f_xy(y)
            z = f_zy(y)
        x_2d, y_2d = projective_transformation(P_g2im, x, y, z)
        if x_2d.shape[0] == 0 or y_2d.shape[0] == 0:
            continue
        fit_2d = np.polyfit(y_2d, x_2d, 2)
        f_xy_2d = np.poly1d(fit_2d)
        y_2d = np.linspace(min(y_2d), max(y_2d), 2 * len(y_2d))
        x_2d = f_xy_2d(y_2d)
        x_2d = x_2d.astype(np.int)
        y_2d = y_2d.astype(np.int)
        if vis_2d:
            for k in range(1, x_2d.shape[0]):
                img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), rgb_color[-1::-1], 2, lineType=cv2.LINE_AA)
        # x = x[y > 40]
        # z = z[y > 40]
        # y = y[y > 40]
        x = x * 2
        ax2.plot(x, y, color=color, linewidth=3)
        if scatter:
            ax2.scatter(x[::2], y[::2], color=color,linewidths=3)
        ax3.plot(x, y, z, color=color, linewidth=5)
    ax1.imshow(img[:, :, [2, 1, 0]])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xlabel('x/m')
    ax2.set_ylabel('y/m')
    bottom, top = ax2.get_ylim()
    left, right = ax3.get_xlim()
    ax2.set_xlim(left, right)
    ax2.set_ylim(15, 100)
    ax2.locator_params(nbins=25, axis='x')
    ax2.locator_params(nbins=10, axis='y')
    ax2.tick_params(labelsize=10)
    bottom, top = ax3.get_zlim()
    left, right = ax3.get_xlim()
    # print("bottom, top, left, right", bottom, top, left, right)
    ax3.set_zlim(min(bottom, -0.1), max(top, 0.1))
    ax3.set_xlim(left, right)
    ax3.set_ylim(0, 100)
    ax3.locator_params(nbins=5, axis='x')
    ax3.locator_params(nbins=10, axis='z')
    ax3.tick_params(pad=18, labelsize=15)
    ax3.set_xlabel('x axis')
    ax3.set_ylabel('y axis')
    ax3.set_zlabel('z axis')
    return img

def draw_laneline_ipm(ax, lanelines, color, H_g2ipm, y_range=(3, 103), x_range=(-10, 10), ipm_h=416, ipm_w=256, vis=None):
    # lanelines: list, [N, 3]
    # draw 2d
    rgb_color = [color[0] * 255, color[1] * 255, color[2] * 255]
    for idx, lane in enumerate(lanelines):
        if vis is not None:
            lane[:] = lane[:, vis[idx]>0.5]
        x = lane[:, 0]
        x = (x - x_range[0]) * ipm_w / (x_range[1] - x_range[0])
        y = lane[:, 1]
        y = ipm_h - ((y - y_range[0]) * ipm_h / (y_range[1] - y_range[0]))
        mask = np.logical_and(np.logical_and(np.logical_and(x >= 0, x < ipm_w), y >= 0), y < ipm_h) 
        x = x[mask]
        y = y[mask]
        if len(x) < 5:
            continue
        z = lane[:, 2]
        ax.plot(x, y, color=color, linewidth=5)
        ax.locator_params(nbins=50, axis='x')
        ax.locator_params(nbins=10, axis='y')
        # ax.scatter(x, y, color=color,linewidths=7)
        # x_ipm, y_ipm = homographic_transformation(H_g2ipm, x, y)
        # x_ipm = x_ipm.astype(np.int)
        # y_ipm = y_ipm.astype(np.int)
        # for k in range(1, x_ipm.shape[0]):
        #     img_ipm = cv2.line(img_ipm, (x_ipm[k - 1], y_ipm[k - 1]), (x_ipm[k], y_ipm[k]), rgb_color[-1::-1], 1, lineType=cv2.LINE_AA)
    # return 

def draw_laneline_nopro(img, ax1, ax2, ax3, lanelines, lanelines_2d, color, vis=None):
    # lanelines: list, [N, 3]
    # draw 2d
    rgb_color = [color[0] * 255, color[1] * 255, color[2] * 255]
    for idx, lane in enumerate(lanelines):
        if vis is not None:
            lane[:] = lane[:, vis[idx]>0.5]
        x = lane[:, 0]
        y = lane[:, 1]
        z = lane[:, 2]
        x_2d, y_2d = lanelines_2d[idx][:, 0], lanelines_2d[idx][:, 1]
        x_2d = x_2d.astype(np.int)
        y_2d = y_2d.astype(np.int)
        for k in range(1, x_2d.shape[0]):
            img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), rgb_color[-1::-1], 1)
        ax2.plot(x, y, color=color, linewidth=2)
        ax2.scatter(x, y, color=color,linewidths=1)
        ax3.plot(x, y, z, color=color, linewidth=2)
    ax1.imshow(img[:, :, [2, 1, 0]])
    ax2.set_xlabel('x axis')
    ax2.set_ylabel('y axis')
    bottom, top = ax2.get_ylim()
    left, right = ax3.get_xlim()
    ax2.set_xlim(left, right)
    ax2.set_ylim(0, 100)
    ax2.locator_params(nbins=50, axis='x')
    ax2.locator_params(nbins=10, axis='y')
    ax2.tick_params(labelsize=15)
    bottom, top = ax3.get_zlim()
    left, right = ax3.get_xlim()
    # print("bottom, top, left, right", bottom, top, left, right)
    ax3.set_zlim(min(bottom, -0.1), max(top, 0.1))
    ax3.set_xlim(left, right)
    ax3.set_ylim(0, 100)
    ax3.locator_params(nbins=5, axis='x')
    ax3.locator_params(nbins=10, axis='z')
    ax3.tick_params(pad=18, labelsize=15)
    ax3.set_xlabel('x axis')
    ax3.set_ylabel('y axis')
    ax3.set_zlabel('z axis')
    return img
        

class Visualizer:
    def __init__(self, args, vis_folder='val_vis'):
        self.save_path = args.save_path
        self.vis_folder = vis_folder
        self.no_3d = args.no_3d
        self.no_centerline = args.no_centerline
        self.vgg_mean = args.vgg_mean
        self.vgg_std = args.vgg_std
        self.ipm_w = args.ipm_w
        self.ipm_h = args.ipm_h
        self.num_y_steps = args.num_y_steps

        if args.no_3d:
            self.anchor_dim = args.num_y_steps + 1
        else:
            if 'ext' in args.mod:
                self.anchor_dim = 3 * args.num_y_steps + 1
            else:
                self.anchor_dim = 2 * args.num_y_steps + 1

        x_min = args.top_view_region[0, 0]
        x_max = args.top_view_region[1, 0]
        self.anchor_x_steps = np.linspace(x_min, x_max, np.int(args.ipm_w / 8), endpoint=True)
        self.anchor_y_steps = args.anchor_y_steps

        # transformation from ipm to ground region
        H_ipm2g = cv2.getPerspectiveTransform(np.float32([[0, 0],
                                                          [self.ipm_w-1, 0],
                                                          [0, self.ipm_h-1],
                                                          [self.ipm_w-1, self.ipm_h-1]]),
                                              np.float32(args.top_view_region))
        self.H_g2ipm = np.linalg.inv(H_ipm2g)

        # probability threshold for choosing visualize lanes
        self.prob_th = args.prob_th

    def draw_on_img(self, img, lane_anchor, P_g2im, draw_type='laneline', color=[0, 0, 1]):
        """
        :param img: image in numpy array, each pixel in [0, 1] range
        :param lane_anchor: lane anchor in N X C numpy ndarray, dimension in agree with dataloader
        :param P_g2im: projection from ground 3D coordinates to image 2D coordinates
        :param draw_type: 'laneline' or 'centerline' deciding which to draw
        :param color: [r, g, b] color for line,  each range in [0, 1]
        :return:
        """

        for j in range(lane_anchor.shape[0]):
            # draw laneline
            if draw_type == 'laneline' and lane_anchor[j, self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, :self.num_y_steps]
                x_3d = x_offsets + self.anchor_x_steps[j]
                if P_g2im.shape[1] == 3:
                    x_2d, y_2d = homographic_transformation(P_g2im, x_3d, self.anchor_y_steps)
                else:
                    z_3d = lane_anchor[j, self.num_y_steps:self.anchor_dim - 1]
                    x_2d, y_2d = projective_transformation(P_g2im, x_3d, self.anchor_y_steps, z_3d)
                x_2d = x_2d.astype(np.int)
                y_2d = y_2d.astype(np.int)
                for k in range(1, x_2d.shape[0]):
                    img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color, 2)

            # draw centerline
            if draw_type == 'centerline' and lane_anchor[j, 2 * self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, self.anchor_dim:self.anchor_dim + self.num_y_steps]
                x_3d = x_offsets + self.anchor_x_steps[j]
                if P_g2im.shape[1] == 3:
                    x_2d, y_2d = homographic_transformation(P_g2im, x_3d, self.anchor_y_steps)
                else:
                    z_3d = lane_anchor[j, self.anchor_dim + self.num_y_steps:2 * self.anchor_dim - 1]
                    x_2d, y_2d = projective_transformation(P_g2im, x_3d, self.anchor_y_steps, z_3d)
                x_2d = x_2d.astype(np.int)
                y_2d = y_2d.astype(np.int)
                for k in range(1, x_2d.shape[0]):
                    img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color, 2)

            # draw the additional centerline for the merging case
            if draw_type == 'centerline' and lane_anchor[j, 3 * self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, 2 * self.anchor_dim:2 * self.anchor_dim + self.num_y_steps]
                x_3d = x_offsets + self.anchor_x_steps[j]
                if P_g2im.shape[1] == 3:
                    x_2d, y_2d = homographic_transformation(P_g2im, x_3d, self.anchor_y_steps)
                else:
                    z_3d = lane_anchor[j, 2 * self.anchor_dim + self.num_y_steps:3 * self.anchor_dim - 1]
                    x_2d, y_2d = projective_transformation(P_g2im, x_3d, self.anchor_y_steps, z_3d)
                x_2d = x_2d.astype(np.int)
                y_2d = y_2d.astype(np.int)
                for k in range(1, x_2d.shape[0]):
                    img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color, 2)
        return img

    def draw_on_img_new(self, img, lane_anchor, P_g2im, draw_type='laneline', color=[0, 0, 1]):
        """
        :param img: image in numpy array, each pixel in [0, 1] range
        :param lane_anchor: lane anchor in N X C numpy ndarray, dimension in agree with dataloader
        :param P_g2im: projection from ground 3D coordinates to image 2D coordinates
        :param draw_type: 'laneline' or 'centerline' deciding which to draw
        :param color: [r, g, b] color for line,  each range in [0, 1]
        :return:
        """
        for j in range(lane_anchor.shape[0]):
            # draw laneline
            if draw_type == 'laneline' and lane_anchor[j, self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, :self.num_y_steps]
                x_3d = x_offsets + self.anchor_x_steps[j]
                # print(P_g2im.shape);exit()
                if P_g2im.shape[1] == 3:
                    x_2d, y_2d = homographic_transformation(P_g2im, x_3d, self.anchor_y_steps)
                    visibility = np.ones_like(x_2d)
                else:
                    z_3d = lane_anchor[j, self.num_y_steps:2*self.num_y_steps]
                    x_2d, y_2d = projective_transformation(P_g2im, x_3d, self.anchor_y_steps, z_3d)
                    visibility = lane_anchor[j, 2 * self.num_y_steps:3 * self.num_y_steps]
                x_2d = x_2d.astype(np.int)
                y_2d = y_2d.astype(np.int)
                for k in range(1, x_2d.shape[0]):
                    if visibility[k] > self.prob_th:
                        img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color, 2)
                    else:
                        img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), [0, 0, 0], 2)

            # draw centerline
            if draw_type == 'centerline' and lane_anchor[j, 2 * self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, self.anchor_dim:self.anchor_dim + self.num_y_steps]
                x_3d = x_offsets + self.anchor_x_steps[j]
                if P_g2im.shape[1] == 3:
                    x_2d, y_2d = homographic_transformation(P_g2im, x_3d, self.anchor_y_steps)
                    visibility = np.ones_like(x_2d)
                else:
                    z_3d = lane_anchor[j, self.anchor_dim + self.num_y_steps:self.anchor_dim + 2*self.num_y_steps]
                    x_2d, y_2d = projective_transformation(P_g2im, x_3d, self.anchor_y_steps, z_3d)
                    visibility = lane_anchor[j, self.anchor_dim + 2*self.num_y_steps:self.anchor_dim + 3*self.num_y_steps]
                x_2d = x_2d.astype(np.int)
                y_2d = y_2d.astype(np.int)
                for k in range(1, x_2d.shape[0]):
                    if visibility[k] > self.prob_th:
                        img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color, 2)
                    else:
                        img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), [0, 0, 0], 2)

            # draw the additional centerline for the merging case
            if draw_type == 'centerline' and lane_anchor[j, 3*self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, 2*self.anchor_dim:2*self.anchor_dim + self.num_y_steps]
                x_3d = x_offsets + self.anchor_x_steps[j]
                if P_g2im.shape[1] == 3:
                    x_2d, y_2d = homographic_transformation(P_g2im, x_3d, self.anchor_y_steps)
                    visibility = np.ones_like(x_2d)
                else:
                    z_3d = lane_anchor[j, 2*self.anchor_dim + self.num_y_steps:2*self.anchor_dim + 2*self.num_y_steps]
                    x_2d, y_2d = projective_transformation(P_g2im, x_3d, self.anchor_y_steps, z_3d)
                    visibility = lane_anchor[j,
                                 2 * self.anchor_dim + 2 * self.num_y_steps:2 * self.anchor_dim + 3 * self.num_y_steps]
                x_2d = x_2d.astype(np.int)
                y_2d = y_2d.astype(np.int)
                for k in range(1, x_2d.shape[0]):
                    if visibility[k] > self.prob_th:
                        img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), color, 2)
                    else:
                        img = cv2.line(img, (x_2d[k - 1], y_2d[k - 1]), (x_2d[k], y_2d[k]), [0, 0, 0], 2)
        return img

    def draw_on_ipm(self, im_ipm, lane_anchor, draw_type='laneline', color=[0, 0, 1]):
        for j in range(lane_anchor.shape[0]):
            # draw laneline
            if draw_type == 'laneline' and lane_anchor[j, self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, :self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]

                # compute lanelines in ipm view
                x_ipm, y_ipm = homographic_transformation(self.H_g2ipm, x_g, self.anchor_y_steps)
                x_ipm = x_ipm.astype(np.int)
                y_ipm = y_ipm.astype(np.int)
                for k in range(1, x_g.shape[0]):
                    im_ipm = cv2.line(im_ipm, (x_ipm[k - 1], y_ipm[k - 1]),
                                      (x_ipm[k], y_ipm[k]), color, 1)

            # draw centerline
            if draw_type == 'centerline' and lane_anchor[j, 2 * self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, self.anchor_dim:self.anchor_dim + self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]

                # compute lanelines in ipm view
                x_ipm, y_ipm = homographic_transformation(self.H_g2ipm, x_g, self.anchor_y_steps)
                x_ipm = x_ipm.astype(np.int)
                y_ipm = y_ipm.astype(np.int)
                for k in range(1, x_g.shape[0]):
                    im_ipm = cv2.line(im_ipm, (x_ipm[k - 1], y_ipm[k - 1]),
                                      (x_ipm[k], y_ipm[k]), color, 1)

            # draw the additional centerline for the merging case
            if draw_type == 'centerline' and lane_anchor[j, 3 * self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, 2 * self.anchor_dim:2 * self.anchor_dim + self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]

                # compute lanelines in ipm view
                x_ipm, y_ipm = homographic_transformation(self.H_g2ipm, x_g, self.anchor_y_steps)
                x_ipm = x_ipm.astype(np.int)
                y_ipm = y_ipm.astype(np.int)
                for k in range(1, x_g.shape[0]):
                    im_ipm = cv2.line(im_ipm, (x_ipm[k - 1], y_ipm[k - 1]),
                                      (x_ipm[k], y_ipm[k]), color, 1)
        return im_ipm

    def draw_on_ipm_new(self, im_ipm, lane_anchor, draw_type='laneline', color=[0, 0, 1], width=1):
        for j in range(lane_anchor.shape[0]):
            # draw laneline
            if draw_type == 'laneline' and lane_anchor[j, self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, :self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    visibility = np.ones_like(x_g)
                else:
                    visibility = lane_anchor[j, 2*self.num_y_steps:3*self.num_y_steps]

                # print(x_g.shape);print(x_g);exit()
                # compute lanelines in ipm view
                x_ipm, y_ipm = homographic_transformation(self.H_g2ipm, x_g, self.anchor_y_steps)
                x_ipm = x_ipm.astype(np.int)
                y_ipm = y_ipm.astype(np.int)
                for k in range(1, x_g.shape[0]):
                    if visibility[k] > self.prob_th:
                        im_ipm = cv2.line(im_ipm, (x_ipm[k - 1], y_ipm[k - 1]),
                                          (x_ipm[k], y_ipm[k]), color, width)
                    else:
                        im_ipm = cv2.line(im_ipm, (x_ipm[k - 1], y_ipm[k - 1]),
                                          (x_ipm[k], y_ipm[k]), [0, 0, 0], width)

            # draw centerline
            if draw_type == 'centerline' and lane_anchor[j, 2*self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, self.anchor_dim:self.anchor_dim + self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    visibility = np.ones_like(x_g)
                else:
                    visibility = lane_anchor[j, self.anchor_dim + 2*self.num_y_steps:self.anchor_dim + 3*self.num_y_steps]

                # compute lanelines in ipm view
                x_ipm, y_ipm = homographic_transformation(self.H_g2ipm, x_g, self.anchor_y_steps)
                x_ipm = x_ipm.astype(np.int)
                y_ipm = y_ipm.astype(np.int)
                for k in range(1, x_g.shape[0]):
                    if visibility[k] > self.prob_th:
                        im_ipm = cv2.line(im_ipm, (x_ipm[k - 1], y_ipm[k - 1]),
                                          (x_ipm[k], y_ipm[k]), color, width)
                    else:
                        im_ipm = cv2.line(im_ipm, (x_ipm[k - 1], y_ipm[k - 1]),
                                          (x_ipm[k], y_ipm[k]), [0, 0, 0], width)

            # draw the additional centerline for the merging case
            if draw_type == 'centerline' and lane_anchor[j, 3*self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, 2*self.anchor_dim:2*self.anchor_dim + self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    visibility = np.ones_like(x_g)
                else:
                    visibility = lane_anchor[j, 2*self.anchor_dim + 2*self.num_y_steps:2*self.anchor_dim + 3*self.num_y_steps]

                # compute lanelines in ipm view
                x_ipm, y_ipm = homographic_transformation(self.H_g2ipm, x_g, self.anchor_y_steps)
                x_ipm = x_ipm.astype(np.int)
                y_ipm = y_ipm.astype(np.int)
                for k in range(1, x_g.shape[0]):
                    if visibility[k] > self.prob_th:
                        im_ipm = cv2.line(im_ipm, (x_ipm[k - 1], y_ipm[k - 1]),
                                          (x_ipm[k], y_ipm[k]), color, width)
                    else:
                        im_ipm = cv2.line(im_ipm, (x_ipm[k - 1], y_ipm[k - 1]),
                                          (x_ipm[k], y_ipm[k]), [0, 0, 0], width)
        return im_ipm

    def draw_3d_curves(self, ax, lane_anchor, draw_type='laneline', color=[0, 0, 1]):
        for j in range(lane_anchor.shape[0]):
            # draw laneline
            if draw_type == 'laneline' and lane_anchor[j, self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, :self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    z_g = np.zeros_like(x_g)
                else:
                    z_g = lane_anchor[j, self.num_y_steps:2*self.num_y_steps]
                ax.plot(x_g, self.anchor_y_steps, z_g, color=color)

            # draw centerline
            if draw_type == 'centerline' and lane_anchor[j, 2*self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, self.anchor_dim:self.anchor_dim + self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    z_g = np.zeros_like(x_g)
                else:
                    z_g = lane_anchor[j, self.anchor_dim + self.num_y_steps:self.anchor_dim + 2*self.num_y_steps]
                ax.plot(x_g, self.anchor_y_steps, z_g, color=color)

            # draw the additional centerline for the merging case
            if draw_type == 'centerline' and lane_anchor[j, 3*self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, 2*self.anchor_dim:2*self.anchor_dim + self.num_y_steps]
                x_g = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    z_g = np.zeros_like(x_g)
                else:
                    z_g = lane_anchor[j, 2*self.anchor_dim + self.num_y_steps:2*self.anchor_dim + 2*self.num_y_steps]
                ax.plot(x_g, self.anchor_y_steps, z_g, color=color)

    def draw_3d_curves_new(self, ax, lane_anchor, h_cam, draw_type='laneline', color=[0, 0, 1]):
        for j in range(lane_anchor.shape[0]):
            # draw laneline
            if draw_type == 'laneline' and lane_anchor[j, self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, :self.num_y_steps]
                x_gflat = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    z_g = np.zeros_like(x_gflat)
                    visibility = np.ones_like(x_gflat)
                else:
                    z_g = lane_anchor[j, self.num_y_steps:2*self.num_y_steps]
                    visibility = lane_anchor[j, 2*self.num_y_steps:3*self.num_y_steps]
                x_gflat = x_gflat[np.where(visibility > self.prob_th)]
                z_g = z_g[np.where(visibility > self.prob_th)]
                if len(x_gflat) > 0:
                    # transform lane detected in flat ground space to 3d ground space
                    x_g, y_g = transform_lane_gflat2g(h_cam,
                                                      x_gflat,
                                                      self.anchor_y_steps[np.where(visibility > self.prob_th)],
                                                      z_g)
                    ax.plot(x_g, y_g, z_g, color=color)

            # draw centerline
            if draw_type == 'centerline' and lane_anchor[j, 2*self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, self.anchor_dim:self.anchor_dim + self.num_y_steps]
                x_gflat = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    z_g = np.zeros_like(x_gflat)
                    visibility = np.ones_like(x_gflat)
                else:
                    z_g = lane_anchor[j, self.anchor_dim + self.num_y_steps:self.anchor_dim + 2*self.num_y_steps]
                    visibility = lane_anchor[j, self.anchor_dim + 2*self.num_y_steps:self.anchor_dim + 3*self.num_y_steps]
                x_gflat = x_gflat[np.where(visibility > self.prob_th)]
                z_g = z_g[np.where(visibility > self.prob_th)]
                if len(x_gflat) > 0:
                    # transform lane detected in flat ground space to 3d ground space
                    x_g, y_g = transform_lane_gflat2g(h_cam,
                                                      x_gflat,
                                                      self.anchor_y_steps[np.where(visibility > self.prob_th)],
                                                      z_g)
                    ax.plot(x_g, y_g, z_g, color=color)

            # draw the additional centerline for the merging case
            if draw_type == 'centerline' and lane_anchor[j, 3*self.anchor_dim - 1] > self.prob_th:
                x_offsets = lane_anchor[j, 2*self.anchor_dim:2*self.anchor_dim + self.num_y_steps]
                x_gflat = x_offsets + self.anchor_x_steps[j]
                if self.no_3d:
                    z_g = np.zeros_like(x_gflat)
                    visibility = np.ones_like(x_gflat)
                else:
                    z_g = lane_anchor[j, 2*self.anchor_dim + self.num_y_steps:2*self.anchor_dim + 2*self.num_y_steps]
                    visibility = lane_anchor[j, 2*self.anchor_dim + 2*self.num_y_steps:2*self.anchor_dim + 3*self.num_y_steps]
                x_gflat = x_gflat[np.where(visibility > self.prob_th)]
                z_g = z_g[np.where(visibility > self.prob_th)]
                if len(x_gflat) > 0:
                    # transform lane detected in flat ground space to 3d ground space
                    x_g, y_g = transform_lane_gflat2g(h_cam,
                                                      x_gflat,
                                                      self.anchor_y_steps[np.where(visibility > self.prob_th)],
                                                      z_g)
                    ax.plot(x_g, y_g, z_g, color=color)

    def save_result(self, dataset, train_or_val, epoch, batch_i, idx, images, gt, pred, pred_cam_pitch, pred_cam_height, aug_mat=np.identity(3, dtype=np.float), evaluate=False):
        if not dataset.data_aug:
            aug_mat = np.repeat(np.expand_dims(aug_mat, axis=0), idx.shape[0], axis=0)

        for i in range(idx.shape[0]):
            # during training, only visualize the first sample of this batch
            if i > 0 and not evaluate:
                break
            im = images.permute(0, 2, 3, 1).data.cpu().numpy()[i]
            # the vgg_std and vgg_mean are for images in [0, 1] range
            im = im * np.array(self.vgg_std)
            im = im + np.array(self.vgg_mean)
            im = np.clip(im, 0, 1)

            gt_anchors = gt[i]
            pred_anchors = pred[i]

            # apply nms to avoid output directly neighbored lanes
            # consider w/o centerline cases
            if self.no_centerline:
                pred_anchors[:, -1] = nms_1d(pred_anchors[:, -1])
            else:
                pred_anchors[:, self.anchor_dim - 1] = nms_1d(pred_anchors[:, self.anchor_dim - 1])
                pred_anchors[:, 2 * self.anchor_dim - 1] = nms_1d(pred_anchors[:, 2 * self.anchor_dim - 1])
                pred_anchors[:, 3 * self.anchor_dim - 1] = nms_1d(pred_anchors[:, 3 * self.anchor_dim - 1])

            H_g2im, P_g2im, H_crop, H_im2ipm = dataset.transform_mats(idx[i])
            if self.no_3d:
                P_gt = np.matmul(H_crop, H_g2im)
                H_g2im_pred = homograpthy_g2im(pred_cam_pitch[i],
                                               pred_cam_height[i], dataset.K)
                P_pred = np.matmul(H_crop, H_g2im_pred)

                # consider data augmentation
                P_gt = np.matmul(aug_mat[i, :, :], P_gt)
                P_pred = np.matmul(aug_mat[i, :, :], P_pred)
            else:
                P_gt = np.matmul(H_crop, P_g2im)
                P_g2im_pred = projection_g2im(pred_cam_pitch[i],
                                              pred_cam_height[i], dataset.K)
                P_pred = np.matmul(H_crop, P_g2im_pred)

                # consider data augmentation
                P_gt = np.matmul(aug_mat[i, :, :], P_gt)
                P_pred = np.matmul(aug_mat[i, :, :], P_pred)

            # update transformation with image augmentation
            H_im2ipm = np.matmul(H_im2ipm, np.linalg.inv(aug_mat[i, :, :]))
            im_ipm = cv2.warpPerspective(im, H_im2ipm, (self.ipm_w, self.ipm_h))
            im_ipm = np.clip(im_ipm, 0, 1)

            # draw lanes on image
            im_laneline = im.copy()
            im_laneline = self.draw_on_img(im_laneline, gt_anchors, P_gt, 'laneline', [0, 0, 1])
            im_laneline = self.draw_on_img(im_laneline, pred_anchors, P_pred, 'laneline', [1, 0, 0])
            if not self.no_centerline:
                im_centerline = im.copy()
                im_centerline = self.draw_on_img(im_centerline, gt_anchors, P_gt, 'centerline', [0, 0, 1])
                im_centerline = self.draw_on_img(im_centerline, pred_anchors, P_pred, 'centerline', [1, 0, 0])

            # draw lanes on ipm
            ipm_laneline = im_ipm.copy()
            ipm_laneline = self.draw_on_ipm(ipm_laneline, gt_anchors, 'laneline', [0, 0, 1])
            ipm_laneline = self.draw_on_ipm(ipm_laneline, pred_anchors, 'laneline', [1, 0, 0])
            if not self.no_centerline:
                ipm_centerline = im_ipm.copy()
                ipm_centerline = self.draw_on_ipm(ipm_centerline, gt_anchors, 'centerline', [0, 0, 1])
                ipm_centerline = self.draw_on_ipm(ipm_centerline, pred_anchors, 'centerline', [1, 0, 0])

            # plot on a single figure
            if self.no_centerline and self.no_3d:
                fig = plt.figure()
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                ax1.imshow(im_laneline)
                ax2.imshow(ipm_laneline)
            elif not self.no_centerline and self.no_3d:
                fig = plt.figure()
                ax1 = fig.add_subplot(221)
                ax2 = fig.add_subplot(222)
                ax3 = fig.add_subplot(223)
                ax4 = fig.add_subplot(224)
                ax1.imshow(im_laneline)
                ax2.imshow(ipm_laneline)
                ax3.imshow(im_centerline)
                ax4.imshow(ipm_centerline)
            elif not self.no_centerline and not self.no_3d:
                fig = plt.figure()
                ax1 = fig.add_subplot(231)
                ax2 = fig.add_subplot(232)
                ax3 = fig.add_subplot(233, projection='3d')
                ax4 = fig.add_subplot(234)
                ax5 = fig.add_subplot(235)
                ax6 = fig.add_subplot(236, projection='3d')
                ax1.imshow(im_laneline)
                ax2.imshow(ipm_laneline)
                self.draw_3d_curves(ax3, gt_anchors, 'laneline', [0, 0, 1])
                self.draw_3d_curves(ax3, pred_anchors, 'laneline', [1, 0, 0])
                ax3.set_xlabel('x axis')
                ax3.set_ylabel('y axis')
                ax3.set_zlabel('z axis')
                bottom, top = ax3.get_zlim()
                ax3.set_zlim(min(bottom, -1), max(top, 1))
                ax3.set_xlim(-20, 20)
                ax3.set_ylim(0, 100)
                ax4.imshow(im_centerline)
                ax5.imshow(ipm_centerline)
                self.draw_3d_curves(ax6, gt_anchors, 'centerline', [0, 0, 1])
                self.draw_3d_curves(ax6, pred_anchors, 'centerline', [1, 0, 0])
                ax6.set_xlabel('x axis')
                ax6.set_ylabel('y axis')
                ax6.set_zlabel('z axis')
                bottom, top = ax6.get_zlim()
                ax6.set_zlim(min(bottom, -1), max(top, 1))
                ax6.set_xlim(-20, 20)
                ax6.set_ylim(0, 100)

            if evaluate:
                fig.savefig(self.save_path + '/example/' + self.vis_folder + '/infer_{}'.format(idx[i]))
            else:
                fig.savefig(self.save_path + '/example/{}/epoch-{}_batch-{}_idx-{}'.format(train_or_val,
                                                                                           epoch, batch_i, idx[i]))
            plt.clf()
            plt.close(fig)

    def save_result_new(self, dataset, train_or_val, epoch, batch_i, idx, images, gt, pred, pred_cam_pitch, pred_cam_height, aug_mat=np.identity(3, dtype=np.float), evaluate=False):
        # print(dataset.data_aug);exit()
        if not dataset.data_aug:
            aug_mat = np.repeat(np.expand_dims(aug_mat, axis=0), idx.shape[0], axis=0)

        for i in range(idx.shape[0]):
            # during training, only visualize the first sample of this batch
            if i > 0 and not evaluate:
                break
            im = images.permute(0, 2, 3, 1).data.cpu().numpy()[i]
            # the vgg_std and vgg_mean are for images in [0, 1] range
            im = im * np.array(self.vgg_std)
            im = im + np.array(self.vgg_mean)
            im = np.clip(im, 0, 1)
            # im2show = (im * 255).astype(np.uint8);print(im2show.shape)  # 360 480 3
            # cv2.imshow('imimg',im2show)
            # cv2.waitKey(0)
            # exit()

            gt_anchors = gt[i]
            pred_anchors = pred[i]

            # apply nms to avoid output directly neighbored lanes
            # consider w/o centerline cases
            if self.no_centerline:
                pred_anchors[:, -1] = nms_1d(pred_anchors[:, -1])
            else:
                pred_anchors[:, self.anchor_dim - 1] = nms_1d(pred_anchors[:, self.anchor_dim - 1])
                pred_anchors[:, 2 * self.anchor_dim - 1] = nms_1d(pred_anchors[:, 2 * self.anchor_dim - 1])
                pred_anchors[:, 3 * self.anchor_dim - 1] = nms_1d(pred_anchors[:, 3 * self.anchor_dim - 1])

            H_g2im, P_g2im, H_crop, H_im2ipm = dataset.transform_mats(idx[i])
            P_gt = np.matmul(H_crop, H_g2im)
            H_g2im_pred = homograpthy_g2im(pred_cam_pitch[i],
                                           pred_cam_height[i], dataset.K)
            P_pred = np.matmul(H_crop, H_g2im_pred)

            # consider data augmentation
            P_gt = np.matmul(aug_mat[i, :, :], P_gt)
            P_pred = np.matmul(aug_mat[i, :, :], P_pred)

            # update transformation with image augmentation
            H_im2ipm = np.matmul(H_im2ipm, np.linalg.inv(aug_mat[i, :, :]))
            im_ipm = cv2.warpPerspective(im, H_im2ipm, (self.ipm_w, self.ipm_h))
            im_ipm = np.clip(im_ipm, 0, 1)
            # im_ipm2show = (im_ipm * 255).astype(np.uint8);print(im_ipm.shape)  # 208 128 3
            # cv2.imshow('im_ipmimg',im_ipm2show)
            # cv2.waitKey(0)
            # exit()

            # draw lanes on image
            im_laneline = im.copy()
            im_laneline = self.draw_on_img_new(im_laneline, gt_anchors, P_gt, 'laneline', [0, 0, 1])
            im_laneline = self.draw_on_img_new(im_laneline, pred_anchors, P_pred, 'laneline', [1, 0, 0])
            if not self.no_centerline:
                im_centerline = im.copy()
                im_centerline = self.draw_on_img_new(im_centerline, gt_anchors, P_gt, 'centerline', [0, 0, 1])
                im_centerline = self.draw_on_img_new(im_centerline, pred_anchors, P_pred, 'centerline', [1, 0, 0])

            # draw lanes on ipm
            ipm_laneline = im_ipm.copy()
            ipm_laneline = self.draw_on_ipm_new(ipm_laneline, gt_anchors, 'laneline', [0, 0, 1])
            ipm_laneline = self.draw_on_ipm_new(ipm_laneline, pred_anchors, 'laneline', [1, 0, 0])
            if not self.no_centerline:
                ipm_centerline = im_ipm.copy()
                ipm_centerline = self.draw_on_ipm_new(ipm_centerline, gt_anchors, 'centerline', [0, 0, 1])
                ipm_centerline = self.draw_on_ipm_new(ipm_centerline, pred_anchors, 'centerline', [1, 0, 0])

            # plot on a single figure
            if self.no_centerline and self.no_3d:
                fig = plt.figure()
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                ax1.imshow(im_laneline)
                ax2.imshow(ipm_laneline)
            elif not self.no_centerline and self.no_3d:
                fig = plt.figure()
                ax1 = fig.add_subplot(221)
                ax2 = fig.add_subplot(222)
                ax3 = fig.add_subplot(223)
                ax4 = fig.add_subplot(224)
                ax1.imshow(im_laneline)
                ax2.imshow(ipm_laneline)
                ax3.imshow(im_centerline)
                ax4.imshow(ipm_centerline)
            elif not self.no_centerline and not self.no_3d:
                fig = plt.figure()
                ax1 = fig.add_subplot(231)
                ax2 = fig.add_subplot(232)
                ax3 = fig.add_subplot(233, projection='3d')
                ax4 = fig.add_subplot(234)
                ax5 = fig.add_subplot(235)
                ax6 = fig.add_subplot(236, projection='3d')
                ax1.imshow(im_laneline)
                ax2.imshow(ipm_laneline)
                # TODO:use separate gt_cam_height when ready
                self.draw_3d_curves_new(ax3, gt_anchors, pred_cam_height[i], 'laneline', [0, 0, 1])
                self.draw_3d_curves_new(ax3, pred_anchors, pred_cam_height[i], 'laneline', [1, 0, 0])
                ax3.set_xlabel('x axis')
                ax3.set_ylabel('y axis')
                ax3.set_zlabel('z axis')
                bottom, top = ax3.get_zlim()
                ax3.set_xlim(-20, 20)
                ax3.set_ylim(0, 100)
                ax3.set_zlim(min(bottom, -1), max(top, 1))
                ax4.imshow(im_centerline)
                ax5.imshow(ipm_centerline)
                # TODO:use separate gt_cam_height when ready
                self.draw_3d_curves_new(ax6, gt_anchors, pred_cam_height[i], 'centerline', [0, 0, 1])
                self.draw_3d_curves_new(ax6, pred_anchors, pred_cam_height[i], 'centerline', [1, 0, 0])
                ax6.set_xlabel('x axis')
                ax6.set_ylabel('y axis')
                ax6.set_zlabel('z axis')
                bottom, top = ax6.get_zlim()
                ax6.set_xlim(-20, 20)
                ax6.set_ylim(0, 100)
                ax6.set_zlim(min(bottom, -1), max(top, 1))

            if evaluate:
                fig.savefig(self.save_path + '/example/' + self.vis_folder + '/infer_{}'.format(idx[i]))
            else:
                fig.savefig(self.save_path + '/example/{}/epoch-{}_batch-{}_idx-{}'.format(train_or_val,
                                                                                           epoch, batch_i, idx[i]))
            plt.clf()
            plt.close(fig)


def prune_3d_lane_by_visibility(lane_3d, visibility):
    lane_3d = lane_3d[visibility > 0, ...]
    return lane_3d


def prune_3d_lane_by_range(lane_3d, x_min, x_max):
    # TODO: solve hard coded range later
    # remove points with y out of range
    # 3D label may miss super long straight-line with only two points: Not have to be 200, gt need a min-step
    # 2D dataset requires this to rule out those points projected to ground, but out of meaningful range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 1] > 0, lane_3d[:, 1] < 200), ...]

    # remove lane points out of x range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 0] > x_min,
                                     lane_3d[:, 0] < x_max), ...]
    return lane_3d

def resample_laneline_in_y_lrj(input_lane, y_steps, out_vis=False):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """

    # at least two points are included
    assert(input_lane.shape[0] >= 2)

    y_min = np.min(input_lane[:, 1])-5
    y_max = np.max(input_lane[:, 1])+5

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

    f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")

    x_values = f_x(y_steps)
    z_values = f_z(y_steps)

    if out_vis:
        output_visibility = np.logical_and(y_steps >= y_min, y_steps <= y_max)
        return x_values, z_values, output_visibility
    return x_values, z_values

def resample_laneline_in_y(input_lane, y_steps, out_vis=False):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """

    # at least two points are included
    assert(input_lane.shape[0] >= 2)

    y_min = np.min(input_lane[:, 1])-5
    y_max = np.max(input_lane[:, 1])+5
    # y_min = np.min(input_lane[:, 1])
    # y_max = np.max(input_lane[:, 1])

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

    f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")

    x_values = f_x(y_steps)
    z_values = f_z(y_steps)

    if out_vis:
        output_visibility = np.logical_and(y_steps >= y_min, y_steps <= y_max)
        return x_values, z_values, output_visibility.astype(np.float32) + 1e-9
    return x_values, z_values

def resample_laneline_in_y_with_vis(input_lane, y_steps, vis_vec):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """

    # at least two points are included
    assert(input_lane.shape[0] >= 2)

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

    f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")
    f_vis = interp1d(input_lane[:, 1], vis_vec, fill_value="extrapolate")

    x_values = f_x(y_steps)
    z_values = f_z(y_steps)
    vis_values = f_vis(y_steps)

    x_values = x_values[vis_values > 0.5]
    y_values = y_steps[vis_values > 0.5]
    z_values = z_values[vis_values > 0.5]
    return np.array([x_values, y_values, z_values]).T


def homography_im2ipm_norm(top_view_region, org_img_size, crop_y, resize_img_size, cam_pitch, cam_height, K):
    """
        Compute the normalized transformation such that image region are mapped to top_view region maps to
        the top view image's 4 corners
        Ground coordinates: x-right, y-forward, z-up
        The purpose of applying normalized transformation: 1. invariance in scale change
                                                           2.Torch grid sample is based on normalized grids

    :param top_view_region: a 4 X 2 list of (X, Y) indicating the top-view region corners in order:
                            top-left, top-right, bottom-left, bottom-right
    :param org_img_size: the size of original image size: [h, w]
    :param crop_y: pixels croped from original img
    :param resize_img_size: the size of image as network input: [h, w]
    :param cam_pitch: camera pitch angle wrt ground plane
    :param cam_height: camera height wrt ground plane in meters
    :param K: camera intrinsic parameters
    :return: H_im2ipm_norm: the normalized transformation from image to IPM image
    """

    # compute homography transformation from ground to image (only this depends on cam_pitch and cam height)
    H_g2im = homograpthy_g2im(cam_pitch, cam_height, K)
    # transform original image region to network input region
    H_c = homography_crop_resize(org_img_size, crop_y, resize_img_size)
    H_g2im = np.matmul(H_c, H_g2im)

    # compute top-view corners' coordinates in image
    x_2d, y_2d = homographic_transformation(H_g2im, top_view_region[:, 0], top_view_region[:, 1])
    border_im = np.concatenate([x_2d.reshape(-1, 1), y_2d.reshape(-1, 1)], axis=1)

    # compute the normalized transformation
    border_im[:, 0] = border_im[:, 0] / resize_img_size[1]
    border_im[:, 1] = border_im[:, 1] / resize_img_size[0]
    border_im = np.float32(border_im)
    dst = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
    # img to ipm
    H_im2ipm_norm = cv2.getPerspectiveTransform(border_im, dst)
    # ipm to im
    H_ipm2im_norm = cv2.getPerspectiveTransform(dst, border_im)
    return H_im2ipm_norm, H_ipm2im_norm


def homography_im2ipm_norm_extrinsic(top_view_region, org_img_size, crop_y, resize_img_size, E, K):
    """
        Compute the normalized transformation such that image region are mapped to top_view region maps to
        the top view image's 4 corners
        Ground coordinates: x-right, y-forward, z-up
        The purpose of applying normalized transformation: 1. invariance in scale change
                                                           2.Torch grid sample is based on normalized grids

    :param top_view_region: a 4 X 2 list of (X, Y) indicating the top-view region corners in order:
                            top-left, top-right, bottom-left, bottom-right
    :param org_img_size: the size of original image size: [h, w]
    :param crop_y: pixels croped from original img
    :param resize_img_size: the size of image as network input: [h, w]
    :param cam_pitch: camera pitch angle wrt ground plane
    :param cam_height: camera height wrt ground plane in meters
    :param K: camera intrinsic parameters
    :return: H_im2ipm_norm: the normalized transformation from image to IPM image
    """

    # compute homography transformation from ground to image (only this depends on cam_pitch and cam height)
    H_g2im = homography_g2im_extrinsic(E, K)
    # transform original image region to network input region
    H_c = homography_crop_resize(org_img_size, crop_y, resize_img_size)
    H_g2im = np.matmul(H_c, H_g2im)

    # compute top-view corners' coordinates in image
    x_2d, y_2d = homographic_transformation(H_g2im, top_view_region[:, 0], top_view_region[:, 1])
    border_im = np.concatenate([x_2d.reshape(-1, 1), y_2d.reshape(-1, 1)], axis=1)

    # compute the normalized transformation
    border_im[:, 0] = border_im[:, 0] / resize_img_size[1]
    border_im[:, 1] = border_im[:, 1] / resize_img_size[0]
    border_im = np.float32(border_im)
    dst = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
    # img to ipm
    H_im2ipm_norm = cv2.getPerspectiveTransform(border_im, dst)
    # ipm to im
    H_ipm2im_norm = cv2.getPerspectiveTransform(dst, border_im)
    return H_im2ipm_norm, H_ipm2im_norm

def homography_ipmnorm2g(top_view_region):
    src = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])
    H_ipmnorm2g = cv2.getPerspectiveTransform(src, np.float32(top_view_region))
    return H_ipmnorm2g


def homograpthy_g2im(cam_pitch, cam_height, K):
    # transform top-view region to original image region
    R_g2c = np.array([[1, 0, 0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch)],
                      [0, np.sin(np.pi / 2 + cam_pitch), np.cos(np.pi / 2 + cam_pitch)]])
    H_g2im = np.matmul(K, np.concatenate([R_g2c[:, 0:2], [[0], [cam_height], [0]]], 1))
    return H_g2im

def homography_g2im_extrinsic(E, K):
    """E: extrinsic matrix, 4*4"""
    E_inv = np.linalg.inv(E)[0:3, :]
    H_g2c = E_inv[:, [0,1,3]]
    H_g2im = np.matmul(K, H_g2c)
    return H_g2im

def projection_g2im(cam_pitch, cam_height, K):
    P_g2c = np.array([[1,                             0,                              0,          0],
                      [0, np.cos(np.pi / 2 + cam_pitch), -np.sin(np.pi / 2 + cam_pitch), cam_height],
                      [0, np.sin(np.pi / 2 + cam_pitch),  np.cos(np.pi / 2 + cam_pitch),          0]])
    P_g2im = np.matmul(K, P_g2c)
    return P_g2im

def projection_g2im_extrinsic(E, K):
    E_inv = np.linalg.inv(E)[0:3, :]
    P_g2im = np.matmul(K, E_inv)
    return P_g2im


def homography_crop_resize(org_img_size, crop_y, resize_img_size):
    """
        compute the homography matrix transform original image to cropped and resized image
    :param org_img_size: [org_h, org_w]
    :param crop_y:
    :param resize_img_size: [resize_h, resize_w]
    :return:
    """
    # transform original image region to network input region
    ratio_x = resize_img_size[1] / org_img_size[1]
    ratio_y = resize_img_size[0] / (org_img_size[0] - crop_y)
    H_c = np.array([[ratio_x, 0, 0],
                    [0, ratio_y, -ratio_y*crop_y],
                    [0, 0, 1]])
    return H_c


def homographic_transformation(Matrix, x, y):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x3 homography matrix
            x (array): original x coordinates
            y (array): original y coordinates
    """
    ones = np.ones((1, len(y)))
    coordinates = np.vstack((x, y, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :]/trans[2, :]
    y_vals = trans[1, :]/trans[2, :]
    return x_vals, y_vals


def projective_transformation(Matrix, x, y, z):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x4 projection matrix
            x (array): original x coordinates
            y (array): original y coordinates
            z (array): original z coordinates
    """
    ones = np.ones((1, len(z)))
    coordinates = np.vstack((x, y, z, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :]/(trans[2, :] + 1e-8)
    y_vals = trans[1, :]/(trans[2, :] + 1e-8)
    return x_vals, y_vals


def transform_lane_gflat2g(h_cam, X_gflat, Y_gflat, Z_g):
    """
        Given X coordinates in flat ground space, Y coordinates in flat ground space, and Z coordinates in real 3D ground space
        with projection matrix from 3D ground to flat ground, compute real 3D coordinates X, Y in 3D ground space.

    :param P_g2gflat: a 3 X 4 matrix transforms lane form 3d ground x,y,z to flat ground x, y
    :param X_gflat: X coordinates in flat ground space
    :param Y_gflat: Y coordinates in flat ground space
    :param Z_g: Z coordinates in real 3D ground space
    :return:
    """

    X_g = X_gflat - X_gflat * Z_g / h_cam
    Y_g = Y_gflat - Y_gflat * Z_g / h_cam

    return X_g, Y_g


def transform_lane_g2gflat(h_cam, X_g, Y_g, Z_g):
    """
        Given X coordinates in flat ground space, Y coordinates in flat ground space, and Z coordinates in real 3D ground space
        with projection matrix from 3D ground to flat ground, compute real 3D coordinates X, Y in 3D ground space.

    :param P_g2gflat: a 3 X 4 matrix transforms lane form 3d ground x,y,z to flat ground x, y
    :param X_gflat: X coordinates in flat ground space
    :param Y_gflat: Y coordinates in flat ground space
    :param Z_g: Z coordinates in real 3D ground space
    :return:
    """

    X_gflat = X_g * h_cam / (h_cam - Z_g)
    Y_gflat = Y_g * h_cam / (h_cam - Z_g)

    return X_gflat, Y_gflat

def convert_lanes_3d_to_gflat(lanes, P_g2gflat):
    """
        Convert a set of lanes from 3D ground coordinates [X, Y, Z], to IPM-based
        flat ground coordinates [x_gflat, y_gflat, Z]
    :param lanes: a list of N x 3 numpy arrays recording a set of 3d lanes
    :param P_g2gflat: projection matrix from 3D ground coordinates to frat ground coordinates
    :return:
    """
    # TODO: this function can be simplified with the derived formula
    for lane in lanes:
        # convert gt label to anchor label
        lane_gflat_x, lane_gflat_y = projective_transformation(P_g2gflat, lane[:, 0], lane[:, 1], lane[:, 2])
        lane[:, 0] = lane_gflat_x
        lane[:, 1] = lane_gflat_y
        
def nms_1d(v):
    """

    :param v: a 1D numpy array
    :return:
    """
    v_out = v.copy()
    len = v.shape[0]
    if len < 2:
        return v
    for i in range(len):
        if i != 0 and v[i - 1] > v[i]:
            v_out[i] = 0.
        elif i != len-1 and v[i+1] > v[i]:
            v_out[i] = 0.
    return v_out


def first_run(save_path):
    txt_file = os.path.join(save_path,'first_run.txt')
    if not os.path.exists(txt_file):
        open(txt_file, 'w').close()
    else:
        saved_epoch = open(txt_file).read()
        if saved_epoch is None:
            print('You forgot to delete [first run file]')
            return '' 
        return saved_epoch
    return ''


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


# trick from stackoverflow
def str2bool(argument):
    if argument.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif argument.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Wrong argument in argparse, should be a boolean')


class Logger(object):
    """
    Source https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        self.fpath = fpath
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def define_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("The requested optimizer: {} is not implemented".format(optim))
    return optimizer


def define_scheduler(optimizer, args):
    if args.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - args.niter) / float(args.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=args.lr_decay_iters, gamma=args.gamma)
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                   factor=args.gamma,
                                                   threshold=0.0001,
                                                   patience=args.lr_decay_iters)
    elif args.lr_policy == 'none':
        scheduler = None
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def define_init_weights(model, init_w='normal', activation='relu'):
    print('Init weights in network with [{}]'.format(init_w))
    if init_w == 'normal':
        model.apply(weights_init_normal)
    elif init_w == 'xavier':
        model.apply(weights_init_xavier)
    elif init_w == 'kaiming':
        model.apply(weights_init_kaiming)
    elif init_w == 'orthogonal':
        model.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{}] is not implemented'.format(init_w))


def weights_init_normal(m):
    classname = m.__class__.__name__
#    print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
#    print(classname)
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
