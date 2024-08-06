from re import L
import os
import json
import pickle

import tqdm
import pdb

import cv2
import numpy as np
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from copy import deepcopy
from scipy.interpolate import interp1d

from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose

from projects.mmdet3d_plugin.datasets.map_utils.utils import *
from projects.mmdet3d_plugin.datasets.map_utils import eval_openlane

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

DARK_GREEN = (115, 181, 34)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
PINK = (180, 105, 255)
CYAN = (255, 128, 0)

CHOCOLATE = (30, 105, 210)
PEACHPUFF = (185, 218, 255)
STATEGRAY = (255, 226, 198)

GT_COLOR = [PINK, CYAN, ORANGE, YELLOW, BLUE]
PRED_COLOR = [RED, GREEN, DARK_GREEN, PURPLE, CHOCOLATE, PEACHPUFF, STATEGRAY]
PRED_HIT_COLOR = GREEN
PRED_MISS_COLOR = RED
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

class LaneEval(object):
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def bench(pred, gt, y_samples, running_time):
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception('Format of lanes error.')
        # if running_time > 200 or len(gt) + 2 < len(pred):
        #     return 0., 0., 1.
        angles = [
            LaneEval.get_angle(np.array(x_gts), np.array(y_samples))
            for x_gts in gt
        ]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        for x_gts, thresh in zip(gt, threshs):
            accs = [
                LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts),
                                       thresh) for x_preds in pred
            ]
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(pred) - matched
        # if len(gt) > 4 and fn > 0:
        #     fn -= 1
        s = sum(line_accs)
        # if len(gt) > 4:
        #     s -= min(line_accs)
        # return s / max(min(4.0, len(gt)), 1.), \
        #         fp / len(pred) if len(pred) > 0 else 0., \
        #         fn / max(min(len(gt), 4.), 1.)
        return s / max(len(gt), 1.), \
                fp / len(pred) if len(pred) > 0 else 0., \
                fn / max(len(gt), 1.)

    @staticmethod
    def bench_one_submit(pred_file, gt_file):
        try:
            json_pred = [
                json.loads(line) for line in open(pred_file).readlines()
            ]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            raise Exception(
                'We do not get the predictions of all the test tasks')
        gts = {l['raw_file']: l for l in json_gt}
        accuracy, fp, fn = 0., 0., 0.
        for pred in json_pred:
            if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
                raise Exception(
                    'raw_file or lanes or run_time not in some predictions.')
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']
            run_time = pred['run_time']
            if raw_file not in gts:
                raise Exception(
                    'Some raw_file from your predictions do not exist in the test tasks.'
                )
            gt = gts[raw_file]
            gt_lanes = gt['lanes']
            y_samples = gt['h_samples']

            # [TODO]multiprocess resample here
            pred_lanes_resample = []
            for idx, lane in enumerate(pred_lanes):
                x_values, _, visibility_vec = resample_laneline_in_y(np.array(lane), np.array(y_samples), out_vis=True)
                x_values[visibility_vec < 0.5] = -2
                x_values[x_values < 0] = -2
                x_values[x_values > 1920] = -2
                if sum(visibility_vec) >= 1:
                    pred_lanes_resample.append(x_values.tolist())

            # import pdb; pdb.set_trace()
            try:
                a, p, n = LaneEval.bench(pred_lanes_resample, gt_lanes, y_samples,
                                         run_time)
            except BaseException as e:
                raise Exception('Format of lanes error.')
            accuracy += a
            fp += p
            fn += n
        num = len(gts)
        # the first return parameter is the default ranking parameter

        fp = fp / num
        fn = fn / num
        tp = 1 - fp
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        return {
            'Accuracy': accuracy / num,
            'F1_score': f1,
            'FP': fp,
            'FN': fn,
        }

@DATASETS.register_module()
class OpenlaneDataset(Dataset):
    CLASSES = ('divider',)
    def __init__(self, 
                 pipeline,
                 data_root,
                 img_dir='images', 
                 img_suffix='.jpg',
                 data_list='training.txt',
                 test_list=None,
                 test_mode=False,
                 dataset_config=None,
                 sample_method='3d',
                 output_vis=True,
                 flip_2d=True,
                 num_pts_per_gt_vec=20,
                 map_classes=None,
                 y_steps = [  5,  10,  15,  20,  30,  40,  50,  60,  80,  100],
                 is_resample=True, 
                 visibility=False,
                 no_cls=False):
        # import pdb; pdb.set_trace()
        self.pipeline = Compose(pipeline)
        self.data_root = data_root
        self.img_dir = os.path.join(data_root, img_dir)
        self.img_suffix = img_suffix
        self.test_mode = test_mode
        self.metric = 'default'
        self.is_resample = is_resample
        self.dataset_config = dataset_config
        self.sample_method = sample_method
        self.output_vis = output_vis
        self.flip_2d = flip_2d
        self.data_list = os.path.join(data_root, 'data_lists', data_list)
        self.cache_dir = os.path.join(data_root, 'cache_dense')
        self.eval_file = os.path.join(data_root, 'data_splits', 'validation.json')  
        self.visibility = visibility
        self.no_cls = no_cls
        
        print('is_resample: {}'.format(is_resample))
        inp_h, inp_w = dataset_config['input_size']

        self.h_org  = 1280
        self.w_org  = 1920
        self.org_h  = 1280
        self.org_w  = 1920
        self.h_crop = 0
        self.crop_y = 0

        # parameters related to service network
        self.h_net = inp_h
        self.w_net = inp_w
        self.resize_h = inp_h
        self.resize_w = inp_w
        self.ipm_h = 208  # 26
        self.ipm_w = 128  # 16

        self.top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
        self.H_crop_ipm = homography_crop_resize([self.h_org, self.w_org], self.h_crop, [self.h_net, self.w_net])
        self.H_crop_im  = homography_crop_resize([self.h_org, self.w_org], self.h_crop, [self.h_org, self.w_org])
        self.H_crop_resize_im = homography_crop_resize([self.h_org, self.w_org], self.h_crop, [self.resize_h, self.resize_w])
        self.H_g2side = cv2.getPerspectiveTransform(
            np.float32([[0, -10], [0, 10], [100, -10], [100, 10]]),
            np.float32([[0, 300], [0, 0], [300, 300], [300, 0]]))
        # org2resized+cropped
        self.H_ipm2g = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [self.ipm_w - 1, 0], [0, self.ipm_h - 1], [self.ipm_w - 1, self.ipm_h - 1]]),
            np.float32(self.top_view_region))

        x_min = self.top_view_region[0, 0]  # -10
        x_max = self.top_view_region[1, 0]  # 10
        self.x_min = x_min  # -10
        self.x_max = x_max  # 10
        # self.anchor_y_steps = np.array(y_steps, dtype=np.float)
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        if num_pts_per_gt_vec == 20:
            # 5, 10, 15, ..., 100 (105)
            self.anchor_y_steps = np.arange(5, 105, 5, dtype=np.float32) # 20
        elif num_pts_per_gt_vec == 21:
            # 3, 8, 13, ..., 98, 103 (108)
            self.anchor_y_steps = np.arange(3, 108, 5, dtype=np.float32) # 21
        self.anchor_len = len(self.anchor_y_steps)

        self.y_min = self.top_view_region[2, 1]
        self.y_max = self.top_view_region[0, 1]
        if self.is_resample:
            self.gflatYnorm = self.anchor_y_steps[-1]
            self.gflatZnorm = 10
            self.gflatXnorm = 30
        else:
            self.gflatYnorm = 200
            self.gflatZnorm = 1
            self.gflatXnorm = 20

        self.num_types = 1
        self.num_categories = 21
        if self.is_resample:
            self.sample_hz = 1
        else:
            self.sample_hz = 4

        self.img_w, self.img_h = self.h_org, self.w_org
        self.max_lanes = 25
        self.normalize = True
        self.to_tensor = ToTensor()

        self.MAPCLASSES = map_classes
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)

        if test_list is not None:
            self.test_list =  os.path.join(self.data_root, test_list)
        else:
            self.test_list = None

        self.load_annotations()
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

        self.gt_lane_start = 5
        self.gt_lane_len = 200
        if not os.path.exists(self.img_dir.replace('images', 'gtmask')):
            # [NOTE] preprocess gt mask
            SEG_WIDTH = 45
            ratio_h, ratio_w = self.h_org / self.h_net, self.w_org / self.w_net
            thickness = int(SEG_WIDTH / ratio_w) # 45 // 1.875 = 24
            for idx in tqdm.tqdm(range(len(self.img_infos))):
                results = self.img_infos[idx].copy()
                with open(results['anno_file'], 'rb') as f:
                    obj = pickle.load(f)
                    results.update(obj)
                results['gt_project_matrix'] = projection_g2im_extrinsic(results['gt_camera_extrinsic'], results['gt_camera_intrinsic'])
                results['gt_homography_matrix'] = homography_g2im_extrinsic(results['gt_camera_extrinsic'], results['gt_camera_intrinsic'])

                valid_lanes = np.where(results['gt_3dlanes'][:, 1] > 0)
                gt_pts = results['gt_3dlanes'][valid_lanes] # N, 605
                x_target = gt_pts[:, self.gt_lane_start : self.gt_lane_start + self.gt_lane_len]
                y_target = np.expand_dims(np.linspace(1, self.gt_lane_len, self.gt_lane_len), axis=0).repeat(gt_pts.shape[0], axis=0)
                z_target = gt_pts[:, self.gt_lane_start + self.gt_lane_len : self.gt_lane_start + self.gt_lane_len * 2]
                vis_target = gt_pts[:, self.gt_lane_start + self.gt_lane_len * 2 : ]
                
                # seg idx has the same order as gt_lanes
                seg_idx_label = np.zeros((gt_pts.shape[0], self.h_net, self.w_net), dtype=np.int8)
                for i in range(gt_pts.shape[0]):
                    valid_pts = np.where(vis_target[i] > 0.5)[0]
                    x_target_i, y_target_i, z_target_i = x_target[i][valid_pts], y_target[i][valid_pts], z_target[i][valid_pts]
                    u_vals, v_vals = self.projective_transformation(results['gt_project_matrix'], 
                                    x_target_i, y_target_i, z_target_i, only_in_img=True)
                    if len(u_vals) < 2:
                        continue
                    u_vals, v_vals = self.resample_laneline_in_y_2d(u_vals, v_vals, 
                                    self.num_pts_per_gt_vec)
                    for j in range(len(u_vals) - 1):
                        seg_idx_label[i] = cv2.line(
                            seg_idx_label[i],
                            (int(u_vals[j] / ratio_w), int(v_vals[j] / ratio_h)), 
                            (int(u_vals[j + 1] / ratio_w), int(v_vals[j + 1] / ratio_h)),
                            color=np.array([1]).item(), # 0: ground, 1: lane
                            thickness=thickness)
                filename = results['filename'].replace('images', 'gtmask').replace('jpg', 'npy')
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                np.save(filename, seg_idx_label)

    def load_annotations(self):
        print('Now loading annotations...')
        self.img_infos = []
        with open(self.data_list, 'r') as anno_obj:
            all_ids = [s.strip() for s in anno_obj.readlines()]
            for k, id in enumerate(all_ids):
                anno = {'filename': os.path.join(self.img_dir, id + self.img_suffix),
                        'anno_file': os.path.join(self.cache_dir, id + '.pkl')}
                self.img_infos.append(anno)
        print("after load annotation")
        print("find {} samples in {}.".format(len(self.img_infos), self.data_list))

    def projective_transformation(self, mat, x, y, z, only_in_img=True):
        ones = np.ones((len(z)))
        coordinates = np.vstack((x, y, z, ones))
        trans = np.matmul(mat, coordinates) # [3,4], [N,4]

        x_vals = trans[0, :] / (trans[2, :] + 1e-8)
        y_vals = trans[1, :] / (trans[2, :] + 1e-8)

        valid_flag = np.logical_and(trans[2, :] > 0,
                            np.logical_and(x_vals > 0, 
                                np.logical_and(x_vals < self.w_org,
                                    np.logical_and(y_vals > 0, y_vals < self.h_org))))
        
        if only_in_img:
            # return points that are visible on 2d
            valid_pts = np.where(valid_flag)
            
            x_vals = x_vals[valid_pts]
            y_vals = y_vals[valid_pts]

            return x_vals, y_vals
        else:
            # return all points and valid_flag
            return x_vals, y_vals, valid_flag

    def resample_laneline_in_y_2d(self, x, y, num_pts):
        # at least two points are included
        assert(x.shape[0] >= 2)
        y_min = np.min(y)
        y_max = np.max(y)
        y_steps = np.linspace(y_min, y_max, num_pts)

        f_x = interp1d(y, x, fill_value="extrapolate")
        x_values = f_x(y_steps)
        return x_values, y_steps

    def resample_laneline_in_y_3d(self, x, y, z, num_pts):
        # at least two points are included
        assert(x.shape[0] >= 2)
        y_min = np.min(y)
        y_max = np.max(y)
        y_steps = np.linspace(y_min, y_max, num_pts)
        f_x = interp1d(y, x, fill_value="extrapolate")
        f_z = interp1d(y, z, fill_value="extrapolate")
        x_values = f_x(y_steps)
        z_values = f_z(y_steps)
        return x_values, y_steps, z_values
    
    def transform_box(self, pts):
        # pts: 20, 2
        pts_x, pts_y = pts[:, 0], pts[:, 1]
        bbox = np.concatenate([
            np.min(pts_x, keepdims=True), 
            np.min(pts_y, keepdims=True), 
            np.max(pts_x, keepdims=True), 
            np.max(pts_y, keepdims=True)])
        return bbox
        
    def __getitem__(self, idx, transform=False):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """
        results = self.img_infos[idx].copy()
        results['img_info'] = {}
        results['img_info']['filename'] = results['filename']
        # print(results['filename'])
        # import pdb; pdb.set_trace()
        results['ori_filename'] = results['filename']
        results['ori_shape'] = (self.h_org, self.w_org)
        results['flip'] = False
        results['flip_direction'] = None
        with open(results['anno_file'], 'rb') as f:
            obj = pickle.load(f)
            results.update(obj)
        # if self.no_cls:
        #     results['gt_3dlanes'][:, 1] = results['gt_3dlanes'][:, 1] > 0     
        results['img_metas'] = {'ori_shape':results['ori_shape']}
        results['gt_project_matrix'] = projection_g2im_extrinsic(results['gt_camera_extrinsic'], results['gt_camera_intrinsic'])
        results['gt_homography_matrix'] = homography_g2im_extrinsic(results['gt_camera_extrinsic'], results['gt_camera_intrinsic'])
        if not self.test_mode:
            valid_lanes = np.where(results['gt_3dlanes'][:, 1] > 0)
            gt_pts = results['gt_3dlanes'][valid_lanes] # N, 605
            x_target = gt_pts[:, self.gt_lane_start : self.gt_lane_start + self.gt_lane_len]
            y_target = np.expand_dims(np.linspace(1, self.gt_lane_len, self.gt_lane_len), axis=0).repeat(gt_pts.shape[0], axis=0)
            z_target = gt_pts[:, self.gt_lane_start + self.gt_lane_len : self.gt_lane_start + self.gt_lane_len * 2]
            vis_target = gt_pts[:, self.gt_lane_start + self.gt_lane_len * 2 : ]
            # import pdb; pdb.set_trace()
            indices = np.arange(2, 103, 5)
            _x_target, _z_target, _vis_target = x_target[:, indices], z_target[:, indices], vis_target[:, indices]
            results['gt_3dlanes'] = np.concatenate((gt_pts[:, 1:2], _x_target, _z_target, _vis_target), axis=1)
            
            gt_2dlanes = []
            gt_2dboxes = []
            for i in range(gt_pts.shape[0]):
                # import pdb; pdb.set_trace()
                if self.sample_method == '2d':
                    valid_pts = np.where(vis_target[i] > 0.5)[0]
                    # 1. Select points that are visible on 3d
                    x_target_i, y_target_i, z_target_i = x_target[i][valid_pts], y_target[i][valid_pts], z_target[i][valid_pts]
                    # 2. Project 3d points onto 2d
                    # 3. Select points that are visible on 2d    
                    u_vals, v_vals = self.projective_transformation(results['gt_project_matrix'], 
                                    x_target_i, y_target_i, z_target_i, only_in_img=True)
                    # 4. Resample laneline in 2d
                    u_vals, v_vals = self.resample_laneline_in_y_2d(u_vals, v_vals, self.num_pts_per_gt_vec)
                elif self.sample_method == '3d':
                    valid_pts = np.where(vis_target[i] > 0.5)[0]
                    # 1. Select points that are visible on 3d
                    x_target_i, y_target_i, z_target_i = x_target[i][valid_pts], y_target[i][valid_pts], z_target[i][valid_pts]
                    # 2. Resample laneline in 3d
                    x_target_i, y_target_i, z_target_i = self.resample_laneline_in_y_3d(
                                    x_target_i, y_target_i, z_target_i, self.num_pts_per_gt_vec)
                    # 3. Project 3d points onto 2d
                    u_vals, v_vals, valid_flag = self.projective_transformation(results['gt_project_matrix'], 
                                    x_target_i, y_target_i, z_target_i, only_in_img=False)
                    # 4. Processing points not seen on 2d
                    u_vals = np.where(valid_flag, u_vals, -1e5 * np.ones_like(u_vals))
                    v_vals = np.where(valid_flag, v_vals, -1e5 * np.ones_like(v_vals))
                elif self.sample_method == '2dvis':
                    # 1. Resample laneline in 3d
                    # indices = np.arange(4, 100, 5) # 20
                    x_target_i, y_target_i, z_target_i = x_target[i][indices], y_target[i][indices], z_target[i][indices]
                    vis_target_i = vis_target[i][indices]
                    # 2. Project 3d points onto 2d
                    u_vals, v_vals, valid_flag = self.projective_transformation(results['gt_project_matrix'], 
                                    x_target_i, y_target_i, z_target_i, only_in_img=False)
                    # 3. Processing points not seen on 2d
                    valid_flag = np.logical_and(valid_flag, vis_target_i > 0.5)
                    u_vals = np.where(valid_flag, u_vals, -1e5 * np.ones_like(u_vals))
                    v_vals = np.where(valid_flag, v_vals, -1e5 * np.ones_like(v_vals))
                elif self.sample_method == '3dvis':
                    valid_pts = np.where(vis_target[i] > 0.5)[0]
                    # 1. Select points that are visible on 3d
                    x_target_i, y_target_i, z_target_i = x_target[i][valid_pts], y_target[i][valid_pts], z_target[i][valid_pts]
                    # 2. Select points that are visible on 2d
                    _, _, valid_flag = self.projective_transformation(results['gt_project_matrix'], 
                                    x_target_i, y_target_i, z_target_i, only_in_img=False)
                    valid_pts = np.where(valid_flag)
                    x_target_i, y_target_i, z_target_i = x_target_i[valid_pts], y_target_i[valid_pts], z_target_i[valid_pts]
                    # 3. Resample laneline in 3d
                    x_target_i, y_target_i, z_target_i = self.resample_laneline_in_y_3d(
                                    x_target_i, y_target_i, z_target_i, self.num_pts_per_gt_vec)
                    # 4. Project 3d points onto 2d
                    u_vals, v_vals = self.projective_transformation(results['gt_project_matrix'], 
                                    x_target_i, y_target_i, z_target_i, only_in_img=True)
                else:
                    raise NotImplementedError()

                lane = np.stack((u_vals, v_vals), axis=1).astype(np.float32) # [20, 2]
                if self.flip_2d:
                    gt_2dlanes.append(np.stack((lane, lane[::-1]), axis=0)) # [2, 20, 2]
                else:
                    gt_2dlanes.append(lane[None]) # [1, 20, 2]
                gt_2dboxes.append(self.transform_box(lane))
            results['gt_2dlanes'] = np.stack(gt_2dlanes, axis=0)
            results['gt_2dboxes'] = np.stack(gt_2dboxes, axis=0)
            results['gt_labels'] = np.zeros((len(gt_2dlanes),), dtype=np.int64)

            # import pdb; pdb.set_trace()
            # img = cv2.imread(results['filename'])
            # color = [255, 0, 0] # blue
            # for i in range(len(results['gt_2dlanes'])):
            #     lane = results['gt_2dlanes'][i][0]
            #     for k in range(1, len(lane)):
            #         if lane[k - 1][0] > 0 and lane[k][0] > 0:
            #             img = cv2.line(img, (int(lane[k - 1][0]), int(lane[k - 1][1])), 
            #                         (int(lane[k][0]), int(lane[k][1])), color, 2)
            # for i in range(len(results['gt_2dboxes'])):
            #     box = results['gt_2dboxes'][i]
            #     img = cv2.rectangle(img, (int(box[0]), int(box[1])), 
            #                     (int(box[2]), int(box[3])), color, 2)
            # cv2.imwrite('./test.jpg', img)
            seg_idx_label = np.load(results['filename'].replace('images', 'gtmask').replace('jpg', 'npy'))
            results['seg_idx_label'] = seg_idx_label.astype(np.float32) # n, h, w
        else:
            # useless when test
            results['gt_2dlanes'] = np.zeros((self.anchor_len,), dtype=np.float32)
            results['gt_2dboxes'] = np.zeros((self.anchor_len,), dtype=np.float32)
            results['gt_labels'] = np.zeros((self.anchor_len,), dtype=np.int64)
            results['seg_idx_label'] = np.zeros((self.anchor_len,), dtype=np.float32)
        
        # seg_idx_label = np.load(results['filename'].replace('images', 'gtmask').replace('jpg', 'npy'))
        # results['seg_idx_label'] = seg_idx_label.astype(np.float32) # n, h, w
        # # [NOTE] check gt mask
        # img = cv2.imread(results['filename'])
        # img = cv2.resize(img, (self.w_net, self.h_net))
        # color = [255, 0, 0] # blue
        # for ins in range(len(seg_idx_label)):
        #     for r in range(len(seg_idx_label[ins])):
        #         for c in range(len(seg_idx_label[ins, r])):
        #             if seg_idx_label[ins][r][c] > 0.5:
        #                 img = cv2.circle(img, (c, r), radius=1, color=color, thickness=1)
        # cv2.imwrite('./test.jpg', img)
        results = self.pipeline(results)
        return results

    def pred2lanes(self, pred):
        ys = np.array(self.anchor_y_steps, dtype=np.float32)
        lanes = []
        logits = []
        probs = []
        for lane in pred:
            lane_xs = lane[5:5 + self.anchor_len]
            lane_zs = lane[5 + self.anchor_len : 5 + 2 * self.anchor_len]
            lane_vis = (lane[5 + self.anchor_len * 2 : 5 + 3 * self.anchor_len]) > 0
            if (lane_vis).sum() < 2:
                continue
            lane_xs = lane_xs[lane_vis]
            lane_ys = ys[lane_vis]
            lane_zs = lane_zs[lane_vis]
            lanes.append(np.stack([lane_xs, lane_ys, lane_zs], axis=-1).tolist())
            logits.append(lane[5 + 3 * self.anchor_len:])
            probs.append(lane[1])
        return lanes, probs, logits

    def pred2apollosimformat(self, idx, pred):
        old_anno = self.img_infos[idx]
        filename = old_anno['filename']
        json_line = dict()
        pred_proposals = pred['proposals_list']
        pred_lanes, prob_lanes, logits_lanes = self.pred2lanes(pred_proposals)
        json_line['raw_file'] = '/'.join(filename.split('/')[-3:])
        json_line["laneLines"] = pred_lanes
        json_line["laneLines_prob"]  = prob_lanes
        json_line["laneLines_logit"] = logits_lanes
        return json_line

    def pred2openlaneformat(self, idx, pred):
        lanes = pred['pts_3d'].numpy() # 50, 20, 2
        scores = pred['scores_3d'].numpy() # 50
        vis = pred['vis_3d'].numpy() # 50, 20
        labels = pred['labels_3d'].numpy() # 50

        pred_lanes = []
        prob_lanes = []
        label_lanes = []
        for i, lane in enumerate(lanes):
            lane_xs = lane[..., 0]
            lane_zs = lane[..., 1]
            lane_vis = vis[i]
            if (lane_vis).sum() < 2:
                continue
            lane_xs = lane_xs[lane_vis]
            lane_ys = self.anchor_y_steps[lane_vis]
            lane_zs = lane_zs[lane_vis]
            pred_lanes.append(np.stack([lane_xs, lane_ys, lane_zs], axis=-1).tolist())
            label_lanes.append(labels[i])
            prob_lanes.append(scores[i])

        old_anno = self.img_infos[idx]
        filename = old_anno['filename']
        json_line = dict()
        json_line['raw_file'] = '/'.join(filename.split('/')[-3:])
        json_line["laneLines"] = pred_lanes
        json_line["laneLines_prob"]  = prob_lanes
        json_line["laneLines_label"] = label_lanes
        return json_line

    def format_results(self, predictions, filename):
        with open(filename, 'w') as jsonFile:
            for idx in tqdm.tqdm(range(len(predictions))):
                # result = self.pred2apollosimformat(idx, predictions[idx])
                result = self.pred2openlaneformat(idx, predictions[idx])
                save_result = {}
                save_result['file_path'] = result['raw_file']
                lane_lines = []
                for k in range(len(result['laneLines'])):
                    # NOTE: use sigmoid instead of softmax
                    # cate = int(np.argmax(result['laneLines_logit'][k][1:])) + 1
                    cate = int(result['laneLines_label'][k]) # +1 in bbox coder
                    prob = float(result['laneLines_prob'][k])
                    lane_lines.append({'xyz': result['laneLines'][k], 'category': cate, 'laneLines_prob': prob})
                save_result['lane_lines'] = lane_lines
                json.dump(save_result, jsonFile)
                jsonFile.write('\n')
        print("save results to ", filename)

    def eval(self,
            predictions, # 
            metric='openlane', # 
            logger=None,
            jsonfile_prefix=None, #
            result_names=['pts_bbox'],
            show=False, #
            show_interval=1, #
            out_dir=None, # 
            pipeline=None, #
            prob_th=0.5
           ):
        # format
        print("evaluating results...")
        os.makedirs(jsonfile_prefix, exist_ok=True)
        pred_filename = os.path.join(jsonfile_prefix, 'lane3d_prediction.json')
        predictions = [out[result_names[0]] for out in predictions]
        self.format_results(predictions, pred_filename)
        # evaluate
        evaluator = eval_openlane.OpenLaneEval(self)
        pred_lines = open(pred_filename).readlines()
        json_pred = [json.loads(line) for line in pred_lines]
        json_gt = [json.loads(line) for line in open(self.eval_file).readlines()]
        if len(json_gt) != len(json_pred):
            print("gt len:", len(json_gt))
            print("pred len:", len(json_pred))
            # raise Exception('We do not get the predictions of all the test tasks')
        if self.test_list is not None:
            test_list = [s.strip().split('.')[0] for s in open(self.test_list, 'r').readlines()]
            json_pred = [s for s in json_pred if s['file_path'][:-4] in test_list]
            json_gt = [s for s in json_gt if s['file_path'][:-4] in test_list]
        gts = {l['file_path']: l for l in json_gt}
        eval_stats = evaluator.bench_one_submit(json_pred, gts, prob_th=prob_th)
        eval_results = {}
        eval_results['F_score'] = eval_stats[0]
        eval_results['recall'] = eval_stats[1]
        eval_results['precision'] = eval_stats[2]
        eval_results['cate_acc'] = eval_stats[3]
        eval_results['x_error_close'] = eval_stats[4]
        eval_results['x_error_far'] = eval_stats[5]
        eval_results['z_error_close'] = eval_stats[6]
        eval_results['z_error_far'] = eval_stats[7]
        print("===> Evaluation on validation set: \n"   
                "laneline F-measure {:.4} \n"
                "laneline Recall  {:.4} \n"
                "laneline Precision  {:.4} \n"
                "laneline Category Accuracy  {:.4} \n"
                "laneline x error (close)  {:.4} m\n"
                "laneline x error (far)  {:.4} m\n"
                "laneline z error (close)  {:.4} m\n"
                "laneline z error (far)  {:.4} m\n".format(
                        eval_results['F_score'], eval_results['recall'],
                        eval_results['precision'], eval_results['cate_acc'],
                        eval_results['x_error_close'], eval_results['x_error_far'],
                        eval_results['z_error_close'], eval_results['z_error_far']))

        # visualize
        if show:
            save_dir = os.path.join(jsonfile_prefix, out_dir)
            os.makedirs(save_dir, exist_ok=True)
            print("visualizing results at", save_dir)
            from projects.mmdet3d_plugin.datasets.map_utils.vis_openlane import LaneVis
            visualizer = LaneVis(self)
            visualizer.visualize(pred_filename, gt_file = self.eval_file, img_dir = self.data_root, 
                                 test_file=self.test_list, save_dir = save_dir, prob_th=prob_th, vis_step=show_interval)
        return eval_results
    
    def _format_bbox(self, results, jsonfile_prefix=None):
        # jsonfile_prefix = 'test/{config}/{time}/pts_bbox'
        os.makedirs(jsonfile_prefix, exist_ok=True)
        res_path = os.path.join(jsonfile_prefix, "maptr_openlane_2dlanes.json")
        
        # the same sample interval as gt
        # h_samples = np.linspace(400, 1275, 176).astype(np.int32) # interval 5
        predictions = []

        # import pdb; pdb.set_trace()
        for i in range(len(results)):
            # results[0].keys()
            # dict_keys(['boxes_3d', 'scores_3d', 'labels_3d', 'pts_3d', 'vis_3d'])
            lanes = results[i]['pts_3d'].numpy() # 50, 20, 2
            scores = results[i]['scores_3d'].numpy() # 50
            vis = results[i]['vis_3d'].numpy() # 50, 20

            lanes_2d_sample = []
            for idx, lane in enumerate(lanes):
                if self.output_vis:
                    # prune lane by visibility
                    lane = lane[np.where(vis[idx])]
                    if len(lane) < 2: continue
                # rearrange y increases monotonically
                lane = lane[np.argsort(lane[:, 1])]
                idx2del = []
                max_y = lane[0, 1]
                end_y = lane[-1, 1]
                for j in range(1, lane.shape[0]):
                    if lane[j, 1] <= max_y or lane[j, 1] > end_y:
                        idx2del.append(j)
                    else:
                        max_y = lane[j, 1]
                lane = np.delete(lane, idx2del, 0)
                if len(lane) < 2: continue
                lanes_2d_sample.append(lane.tolist())

            predictions.append(
                {
                    # gt file format
                    'raw_file': self.img_infos[i]['filename'].replace("./data/OpenLane/images", "clips"),
                    'lanes': lanes_2d_sample,
                    'scores': scores.tolist(),
                    'run_time': 1.0
                }
            )

        print('Results writes to', res_path)
        with open(res_path, "w") as f:
            for l in predictions:
                json.dump(l, f)
                f.write('\n')
        return res_path

    def format_results_2d(self, results, jsonfile_prefix=None):
        # assert isinstance(results, list), 'results must be a list'
        # assert len(results) == len(self), (
        #     'The length of results is not equal to the dataset len: {} != {}'.
        #     format(len(results), len(self)))

        result_files = dict()
        for name in results[0]: # pts_bbox
            print(f'\nFormating bboxes of {name}')
            results_ = [out[name] for out in results]
            tmp_file_ = os.path.join(jsonfile_prefix, name)
            result_files.update({name: self._format_bbox(results_, tmp_file_)})
        return result_files

    def _evaluate_single(self,
                        result_path,
                        logger=None,
                        metric='chamfer', # clrnet
                        result_name='pts_bbox'):
        """Evaluation OpenLane 2d Lanes.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        
        print('-*' * 10 + f'use metric:{metric}' + '-*' * 10)
        if metric == "clrnet":
            test_json_file = './data/OpenLane/gt2d/test.json'
            result = LaneEval.bench_one_submit(result_path, test_json_file)
        elif metric == "chamfer":
            from projects.mmdet3d_plugin.datasets.map_utils.mean_ap import eval_map
            thresholds = [20., 30., 40.]
            num_pts = 100
            cls_aps = np.zeros((len(thresholds), self.NUM_MAPCLASSES))

            '''
            len(cls_gens) = 3(dict_keys(['divider', 'ped_crossing', 'boundary']))
            type(cls_gens['divider']) = 'tuple', len=len(dataset)
            cls_gens['divider'])[0].shape = [21, 201](lane num, 200 pts + score)
            '''
            cls_gens = {'divider': []}
            json_pred = [json.loads(line) for line in open(result_path).readlines()]
            for pred in json_pred:
                pred_lanes = pred['lanes']
                scores = pred['scores']
                pred_lanes_resample = []
                for lane in pred_lanes:
                    lane = np.array(lane)
                    x_values, y_values = self.resample_laneline_in_y_2d(lane[:, 0], lane[:, 1], num_pts)
                    pred_lanes_resample.append(np.stack((x_values, y_values), axis=1))
                if len(pred_lanes_resample):
                    pred_lanes_resample = np.stack(pred_lanes_resample, axis=0).reshape((len(pred_lanes), -1))
                    pred_lanes_resample = np.concatenate((pred_lanes_resample, np.array(scores)[:, np.newaxis]), axis=1)
                else:
                    pred_lanes_resample = np.zeros((0, num_pts * 2 + 1))
                cls_gens['divider'].append(pred_lanes_resample)
            cls_gens['divider'] = tuple(cls_gens['divider'])

            cls_gts = {'divider': []}
            for idx in range(len(self)):
                results = self.img_infos[idx].copy()
                with open(results['anno_file'], 'rb') as f:
                    obj = pickle.load(f)
                    results.update(obj)
                results['gt_project_matrix'] = projection_g2im_extrinsic(results['gt_camera_extrinsic'], results['gt_camera_intrinsic'])
                valid_lanes = np.where(results['gt_3dlanes'][:, 1] > 0)
                gt_pts = results['gt_3dlanes'][valid_lanes]
                x_target = gt_pts[:, self.gt_lane_start : self.gt_lane_start + self.gt_lane_len]
                y_target = np.expand_dims(np.linspace(1, self.gt_lane_len, self.gt_lane_len), axis=0).repeat(gt_pts.shape[0], axis=0)
                z_target = gt_pts[:, self.gt_lane_start + self.gt_lane_len : self.gt_lane_start + self.gt_lane_len * 2]
                vis_target = gt_pts[:, self.gt_lane_start + self.gt_lane_len * 2 : ]
                gt_2dlanes = []
                for i in range(gt_pts.shape[0]):
                    valid_pts = np.where(vis_target[i] > 0.5)[0]
                    x_target_i, y_target_i, z_target_i = x_target[i][valid_pts], y_target[i][valid_pts], z_target[i][valid_pts]
                    u_vals, v_vals = self.projective_transformation(results['gt_project_matrix'], 
                            x_target_i, y_target_i, z_target_i, only_in_img=False)
                    u_vals, v_vals = self.resample_laneline_in_y_2d(u_vals, v_vals, num_pts)
                    gt_2dlanes.append(np.stack((u_vals, v_vals), axis=1))
                if len(gt_2dlanes):
                    gt_2dlanes = np.stack(gt_2dlanes, axis=0).reshape((len(gt_pts), -1))
                else:
                    gt_2dlanes = np.zeros((0, num_pts * 2))
                cls_gts['divider'].append(gt_2dlanes)
            cls_gts['divider'] = tuple(cls_gts['divider'])

            # import pdb; pdb.set_trace()
            result = {}
            for i, thr in enumerate(thresholds):
                print('-*' * 10 + f'threshhold:{thr}' + '-*' * 10)
                mAP, cls_ap = eval_map(
                                gen_results=None,
                                annotations=None,
                                cls_gens=cls_gens,
                                cls_gts=cls_gts,
                                threshold=thr,
                                cls_names=self.MAPCLASSES,
                                logger=None,
                                num_pred_pts_per_instance=self.num_pts_per_gt_vec,
                                pc_range=None,
                                metric=metric)

                for j in range(self.NUM_MAPCLASSES):
                    cls_aps[i, j] = cls_ap[j]['ap']

            for i, name in enumerate(self.MAPCLASSES):
                print('{}: {}'.format(name, cls_aps.mean(0)[i]))
                result['OpenLane_{}/{}_AP'.format(metric, name)] = cls_aps.mean(0)[i]
            # print('map: {}'.format(cls_aps.mean(0).mean()))
            # result['OpenLane_{}/mAP'.format(metric)] = cls_aps.mean(0).mean()

            for i, name in enumerate(self.MAPCLASSES):
                for j, thr in enumerate(thresholds):
                    result['OpenLane_{}/{}_AP_thr_{}'.format(metric, name, thr)] = cls_aps[j][i]
                    
        else:
            raise NotImplementedError()

        return result

    def evaluate(self,
                 results, # 
                 metric='bbox', # chamfer
                 logger=None,
                 jsonfile_prefix=None, #
                 result_names=['pts_bbox'],
                 show=False, #
                 show_interval=1, #
                 out_dir=None, # 
                 pipeline=None #
                ):
        """Evaluation OpenLane 2d Lanes.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files = self.format_results_2d(results, jsonfile_prefix)

        # metric = ['clrnet', 'chamfer']
        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name], metric=metric[0])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files, metric=metric[0])
        print(results_dict)

        if show:
            print('Visualizing {} frames'.format(len(self) // show_interval))
            out_dir = os.path.join(jsonfile_prefix, out_dir)
            self.show(result_files[result_names[0]], out_dir, show_interval)
            # self.show(result_files, out_dir, show_interval)

        return results_dict


    def show(self, result_files, out_dir=None, show_interval=1):
        json_pred = [
            json.loads(line) for line in open(result_files).readlines()
        ]
        for pred in json_pred[::show_interval]:
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']

            lanes_xys = []
            for lane in pred_lanes:
                xys = []
                for x, y in lane:
                    if x <= 0 or y <= 0:
                        continue
                    x, y = int(x), int(y)
                    xys.append((x, y))
                if len(xys):
                    lanes_xys.append(xys)
            if len(lanes_xys):
                lanes_xys.sort(key=lambda xys : xys[0][0])

            img = cv2.imread(raw_file.replace("clips", "./data/OpenLane/images"))
            for idx, xys in enumerate(lanes_xys):
                for i in range(1, len(xys)):
                    cv2.line(img, xys[i - 1], xys[i], PRED_COLOR[idx % len(PRED_COLOR)], thickness=2)

            out_file = os.path.join(out_dir, raw_file.split("validation/")[1])
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            cv2.imwrite(out_file, img)

    def __len__(self):
        return len(self.img_infos)

    def _get_img_heigth(self, path):
        return 1280

    def _get_img_width(self, path):
        return 1920
