import os
import cv2
import torch
import numpy as np
import random
from mmcv.parallel import DataContainer as DC
from math import factorial
from pyquaternion import Quaternion
from shapely.geometry import LineString, Point, box

import mmcv
from mmdet.datasets import DATASETS
from mmdet3d.datasets import Custom3DDataset

from .openlanev2.centerline.dataset import Collection
from .openlanev2.centerline.evaluation import evaluate as openlanev2_evaluate
from .openlanev2.centerline.evaluation import evaluate_centerline as openlanev2_evaluate_centerline
from .openlanev2.centerline.visualization.utils import COLOR_DICT, interp_arc

COLOR_GT = (0, 255, 0)
COLOR_GT_TOPOLOGY = (0, 127, 0)
COLOR_PRED = (0, 0, 255)
COLOR_PRED_TOPOLOGY = (0, 0, 127)
COLOR_DICT = {k: (v[2], v[1], v[0]) for k, v in COLOR_DICT.items()}
COLOR_NUM = len(COLOR_DICT.keys())

def render_pv(images, lidar2imgs, 
              gt_lane, gt_label,
            #   gt_lc, pred_lc, gt_te, gt_te_attr, pred_te, pred_te_attr,
            ):

    results = []

    for idx, (image, lidar2img) in enumerate(zip(images, lidar2imgs)):

        for i, lane in enumerate(gt_lane):
            xyz1 = np.concatenate([lane, np.ones((lane.shape[0], 1))], axis=1)
            xyz1 = xyz1 @ lidar2img.T
            xyz1 = xyz1[xyz1[:, 2] > 1e-5]
            if xyz1.shape[0] == 0:
                continue
            points_2d = xyz1[:, :2] / xyz1[:, 2:3]

            points_2d = points_2d.astype(int)
            image = cv2.polylines(image, points_2d[None], False, COLOR_DICT[i % COLOR_NUM], 2)

        # if gt_lc is not None :
        #     for lc in gt_lc:
        #         xyz1 = np.concatenate([lc, np.ones((lc.shape[0], 1))], axis=1)
        #         xyz1 = xyz1 @ lidar2img.T
        #         xyz1 = xyz1[xyz1[:, 2] > 1e-5]
        #         if xyz1.shape[0] == 0:
        #             continue
        #         points_2d = xyz1[:, :2] / xyz1[:, 2:3]

        #         points_2d = points_2d.astype(int)
        #         image = cv2.polylines(image, points_2d[None], False, COLOR_GT, 2)

        # if pred_lc is not None:
        #     for lc in pred_lc:
        #         xyz1 = np.concatenate([lc, np.ones((lc.shape[0], 1))], axis=1)
        #         xyz1 = xyz1 @ lidar2img.T
        #         xyz1 = xyz1[xyz1[:, 2] > 1e-5]
        #         if xyz1.shape[0] == 0:
        #             continue
        #         points_2d = xyz1[:, :2] / xyz1[:, 2:3]

        #         points_2d = points_2d.astype(int)
        #         image = cv2.polylines(image, points_2d[None], False, COLOR_PRED, 2)

        # if idx == 0: # front view image
            
        #     if gt_te is not None:
        #         for bbox, attr in zip(gt_te, gt_te_attr):
        #             b = bbox.astype(np.int32)
        #             image = render_corner_rectangle(image, (b[0], b[1]), (b[2], b[3]), COLOR_DICT[attr], 3, 1)

        #     if pred_te is not None:
        #         for bbox, attr in zip(pred_te, pred_te_attr):
        #             b = bbox.astype(np.int32)
        #             image = cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), COLOR_DICT[attr], 3)

        results.append(image)

    return results

def render_corner_rectangle(img, pt1, pt2, color,
                            corner_thickness=3, edge_thickness=2,
                            centre_cross=False, lineType=cv2.LINE_8):

    corner_length = min(abs(pt1[0] - pt2[0]), abs(pt1[1] - pt2[1])) // 4
    e_args = [color, edge_thickness, lineType]
    c_args = [color, corner_thickness, lineType]

    # edges
    img = cv2.line(img, (pt1[0] + corner_length, pt1[1]), (pt2[0] - corner_length, pt1[1]), *e_args)
    img = cv2.line(img, (pt2[0], pt1[1] + corner_length), (pt2[0], pt2[1] - corner_length), *e_args)
    img = cv2.line(img, (pt1[0], pt1[1] + corner_length), (pt1[0], pt2[1] - corner_length), *e_args)
    img = cv2.line(img, (pt1[0] + corner_length, pt2[1]), (pt2[0] - corner_length, pt2[1]), *e_args)

    # corners
    img = cv2.line(img, pt1, (pt1[0] + corner_length, pt1[1]), *c_args)
    img = cv2.line(img, pt1, (pt1[0], pt1[1] + corner_length), *c_args)
    img = cv2.line(img, (pt2[0], pt1[1]), (pt2[0] - corner_length, pt1[1]), *c_args)
    img = cv2.line(img, (pt2[0], pt1[1]), (pt2[0], pt1[1] + corner_length), *c_args)
    img = cv2.line(img, (pt1[0], pt2[1]), (pt1[0] + corner_length, pt2[1]), *c_args)
    img = cv2.line(img, (pt1[0], pt2[1]), (pt1[0], pt2[1] - corner_length), *c_args)
    img = cv2.line(img, pt2, (pt2[0] - corner_length, pt2[1]), *c_args)
    img = cv2.line(img, pt2, (pt2[0], pt2[1] - corner_length), *c_args)

    if centre_cross:
        cx, cy = int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2)
        img = cv2.line(img, (cx - corner_length, cy), (cx + corner_length, cy), *e_args)
        img = cv2.line(img, (cx, cy - corner_length), (cx, cy + corner_length), *e_args)
    
    return img

def render_front_view(image, lidar2img, gt_lc, pred_lc, gt_te, pred_te, gt_topology_lcte, pred_topology_lcte):

    if gt_topology_lcte is not None:
        for lc_idx, lcte in enumerate(gt_topology_lcte):
            for te_idx, connected in enumerate(lcte):
                if connected:
                    lc = gt_lc[lc_idx]
                    lc = lc[len(lc) // 2][None, ...]
                    xyz1 = np.concatenate([lc, np.ones((lc.shape[0], 1))], axis=1)
                    xyz1 = xyz1 @ lidar2img.T
                    xyz1 = xyz1[xyz1[:, 2] > 1e-5]
                    if xyz1.shape[0] == 0:
                        continue
                    p1 = (xyz1[:, :2] / xyz1[:, 2:3])[0].astype(int)

                    te = gt_te[te_idx]
                    p2 = np.array([(te[0]+te[2])/2, te[3]]).astype(int)

                    image = cv2.arrowedLine(image, (p2[0], p2[1]), (p1[0], p1[1]), COLOR_GT_TOPOLOGY, tipLength=0.03)

    if pred_topology_lcte is not None:
        for lc_idx, lcte in enumerate(pred_topology_lcte):
            for te_idx, connected in enumerate(lcte):
                if connected:
                    lc = pred_lc[lc_idx]
                    lc = lc[len(lc) // 2][None, ...]
                    xyz1 = np.concatenate([lc, np.ones((lc.shape[0], 1))], axis=1)
                    xyz1 = xyz1 @ lidar2img.T
                    xyz1 = xyz1[xyz1[:, 2] > 1e-5]
                    if xyz1.shape[0] == 0:
                        continue
                    p1 = (xyz1[:, :2] / xyz1[:, 2:3])[0].astype(int)

                    te = pred_te[te_idx]
                    p2 = np.array([(te[0]+te[2])/2, te[3]]).astype(int)

                    image = cv2.arrowedLine(image, (p2[0], p2[1]), (p1[0], p1[1]), COLOR_PRED_TOPOLOGY, tipLength=0.03)

    return image
    
def render_bev(gt_lane, gt_label,
        # gt_lc=None, pred_lc=None, gt_topology_lclc=None, pred_topology_lclc=None, 
        map_size=[-52, 52, -27, 27], scale=20
    ):

    image = np.zeros((int(scale*(map_size[1]-map_size[0])), int(scale*(map_size[3] - map_size[2])), 3), dtype=np.uint8)

    for i, lane in enumerate(gt_lane):
        # color = COLOR_DICT[i % COLOR_NUM]
        color = COLOR_DICT[gt_label[i] % COLOR_NUM]
        draw_coor = (scale * (-lane[:, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
        image = cv2.polylines(image, [draw_coor[:, [1,0]]], False, color, max(round(scale * 0.2), 1))
        image = cv2.circle(image, (draw_coor[0, 1], draw_coor[0, 0]), max(round(scale * 0.5), 3), color, -1)
        image = cv2.circle(image, (draw_coor[-1, 1], draw_coor[-1, 0]), max(round(scale * 0.5), 3), color, -1)

    # if gt_lc is not None:
    #     for lc in gt_lc:
    #         draw_coor = (scale * (-lc[:, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
    #         image = cv2.polylines(image, [draw_coor[:, [1,0]]], False, COLOR_GT, max(round(scale * 0.2), 1))
    #         image = cv2.circle(image, (draw_coor[0, 1], draw_coor[0, 0]), max(round(scale * 0.5), 3), COLOR_GT, -1)
    #         image = cv2.circle(image, (draw_coor[-1, 1], draw_coor[-1, 0]), max(round(scale * 0.5), 3), COLOR_GT, -1)
    
    # if gt_topology_lclc is not None:
    #     for l1_idx, lclc in enumerate(gt_topology_lclc):
    #         for l2_idx, connected in enumerate(lclc):
    #             if connected:
    #                 l1 = gt_lc[l1_idx]
    #                 l2 = gt_lc[l2_idx]
    #                 l1_mid = len(l1) // 2
    #                 l2_mid = len(l2) // 2
    #                 p1 = (scale * (-l1[l1_mid, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
    #                 p2 = (scale * (-l2[l2_mid, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
    #                 image = cv2.arrowedLine(image, (p1[1], p1[0]), (p2[1], p2[0]), COLOR_GT_TOPOLOGY, max(round(scale * 0.1), 1), tipLength=0.03)

    # if pred_lc is not None:
    #     for lc in pred_lc:
    #         draw_coor = (scale * (-lc[:, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
    #         image = cv2.polylines(image, [draw_coor[:, [1,0]]], False, COLOR_PRED, max(round(scale * 0.2), 1))
    #         image = cv2.circle(image, (draw_coor[0, 1], draw_coor[0, 0]), max(round(scale * 0.5), 3), COLOR_PRED, -1)
    #         image = cv2.circle(image, (draw_coor[-1, 1], draw_coor[-1, 0]), max(round(scale * 0.5), 3), COLOR_PRED, -1)

    # if pred_topology_lclc is not None:
    #     for l1_idx, lclc in enumerate(pred_topology_lclc):
    #         for l2_idx, connected in enumerate(lclc):
    #             if connected:
    #                 l1 = pred_lc[l1_idx]
    #                 l2 = pred_lc[l2_idx]
    #                 l1_mid = len(l1) // 2
    #                 l2_mid = len(l2) // 2
    #                 p1 = (scale * (-l1[l1_mid, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
    #                 p2 = (scale * (-l2[l2_mid, :2] + np.array([map_size[1], map_size[3]]))).astype(np.int)
    #                 image = cv2.arrowedLine(image, (p1[1], p1[0]), (p2[1], p2[0]), COLOR_PRED_TOPOLOGY, max(round(scale * 0.1), 1), tipLength=0.03)

    return image

@DATASETS.register_module()
class OpenLaneV2SubsetADataset(Custom3DDataset):

    CLASSES = [None]

    def __init__(self,
                 data_root,
                 meta_root,
                 collection,
                 fixed_num,
                 code_size,
                 img_size,
                 pipeline,
                 test_mode,
                 flip_gt=False,
                 map_classes=[None],
                 input_size=None,
                 sample_interval=20,
                 crop_h=(0, 2048),
                 point_cloud_range=[-50.0, -25.0, -3.0, 50.0, 25.0, 2.0],
                 # sliding window mode
                 queue_length=1,
                 random_length=0,
                 num_frame_losses=1,
                 # streaming mode
                 seq_mode=False, 
                ):
        self.ann_file = f'{meta_root}/{collection}.pkl'
        self.fixed_num = fixed_num
        self.code_size = code_size
        self.img_size = img_size
        self.flip_gt = flip_gt
        self.map_classes = map_classes
        self.input_size = input_size if input_size else (1024, 1024)
        self.h_org, self.w_org  = 2048, 2048
        self.h_net, self.w_net = self.input_size
        self.sample_interval = sample_interval
        self.crop_h = crop_h
        self.point_cloud_range = point_cloud_range
        self.queue_length = queue_length
        self.random_length = random_length
        self.num_frame_losses = num_frame_losses
        self.seq_mode = seq_mode
        super().__init__(
            data_root=data_root, 
            ann_file=self.ann_file, 
            pipeline=pipeline, 
            test_mode=test_mode,
        )
        # [NOTE] preprocess gt mask
        # self.prepare_seg_label()
        self.prev_scene_token = None
        if seq_mode:
            # import pdb; pdb.set_trace()
            self.queue_length = 1
            self.random_length = 0
            self.num_frame_losses = 1
            self.seq_split_num = 4 # quene length 8
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        import math
        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and self.data_infos[idx][1] != self.data_infos[idx - 1][1]:
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                self.flag = np.array(new_flags, dtype=np.int64)

    def load_annotations(self, ann_file):
        # import pdb; pdb.set_trace()
        # ann_file = ann_file.name.split('.pkl')[0].split('/')
        ann_file = ann_file.split('.pkl')[0].split('/')
        self.collection = Collection(data_root=self.data_root, meta_root='/'.join(ann_file[:-1]), collection=ann_file[-1])
        return self.collection.keys

    def prepare_seg_label(self):
        from tqdm import tqdm
        for index in tqdm(range(len(self.data_infos))):
            split, segment_id, timestamp = self.data_infos[index]
            frame = self.collection.get_frame_via_identifier((split, segment_id, timestamp))

            img_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            rots = []
            trans = []
            cam2imgs = []
            for i, camera in enumerate(frame.get_camera_list()):

                assert camera == 'ring_front_center' if i == 0 else True, \
                    'the first image should be the front view'

                lidar2cam_r = np.linalg.inv(frame.get_extrinsic(camera)['rotation'])
                lidar2cam_t = frame.get_extrinsic(camera)['translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t

                intrinsic = frame.get_intrinsic(camera)['K']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)

                img_paths.append(frame.get_image_path(camera))
                lidar2cam_rts.append(lidar2cam_rt.T)
                cam_intrinsics.append(viewpad)
                lidar2img_rts.append(lidar2img_rt)
                rots.append(np.linalg.inv(frame.get_extrinsic(camera)['rotation']))
                trans.append(-frame.get_extrinsic(camera)['translation'])
                cam2imgs.append(frame.get_intrinsic(camera)['K'])

            can_bus = np.zeros(18)
            rotation = Quaternion._from_matrix(frame.get_pose()['rotation'])
            can_bus[:3] = frame.get_pose()['translation']
            can_bus[3:7] = rotation
            patch_angle = rotation.yaw_pitch_roll[0] / np.pi * 180
            if patch_angle < 0:
                patch_angle += 360
            can_bus[-2] = patch_angle / 180 * np.pi
            can_bus[-1] = patch_angle

            input_dict = {
                'scene_token': segment_id,
                'sample_idx': timestamp,
                'img_paths': img_paths,
                'lidar2cam': lidar2cam_rts,
                'cam_intrinsic': cam_intrinsics,
                'lidar2img': lidar2img_rts,
                'rots': rots,
                'trans': trans,
                'cam2imgs': cam2imgs,
                'can_bus': can_bus,
            }

            input_dict.update(self.get_ann_info(index, input_dict))

            # import pdb; pdb.set_trace()
            SEG_WIDTH = 40
            for cam_idx, filename in enumerate(input_dict['img_paths']):
                # if cam_idx > 0:
                #     ratio_h, ratio_w = self.h_org / self.h_net, self.w_org / self.w_net
                #     thickness = round(SEG_WIDTH / ratio_h) # 16
                # else:
                #     ratio_h, ratio_w = self.w_org / self.h_net, self.h_org / self.w_net
                #     thickness = round(SEG_WIDTH / ratio_h) # 12
                ratio_h, ratio_w = self.h_org / self.h_net, self.w_org / self.w_net
                thickness = round(SEG_WIDTH / ratio_h) # 20
                gt_pts = input_dict['gt_2dlanes'][cam_idx] # vec, 1, pts, 2
                seg_idx_label = np.zeros((gt_pts.shape[0], self.h_net, self.w_net), dtype=np.int8)
                for i in range(gt_pts.shape[0]):
                    u_vals = gt_pts[i, 0, :, 0]
                    v_vals = gt_pts[i, 0, :, 1]
                    for j in range(len(u_vals) - 1):
                        seg_idx_label[i] = cv2.line(
                            seg_idx_label[i],
                            (int(u_vals[j] / ratio_w), int(v_vals[j] / ratio_h)), 
                            (int(u_vals[j + 1] / ratio_w), int(v_vals[j + 1] / ratio_h)),
                            color=np.array([1]).item(), # 0: ground, 1: lane
                            thickness=thickness)
                filename = filename.replace('image', 'gtmask').replace('jpg', 'npy')
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                np.save(filename, seg_idx_label)
                # img = cv2.imread(filename)
                # img = mmcv.impad(img, shape=(self.w_org, self.h_org), pad_val=128)
                # img = cv2.resize(img, (self.w_net, self.h_net))
                # color = [255, 0, 0] # blue
                # for ins in range(len(seg_idx_label)):
                #     for r in range(len(seg_idx_label[ins])):
                #         for c in range(len(seg_idx_label[ins, r])):
                #             if seg_idx_label[ins][r][c] > 0.5:
                #                 img = cv2.circle(img, (c, r), radius=1, color=color, thickness=1)
                # cv2.imwrite('./test.jpg', img)

    def get_data_info(self, index):

        split, segment_id, timestamp = self.data_infos[index]
        frame = self.collection.get_frame_via_identifier((split, segment_id, timestamp))

        img_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        rots = []
        trans = []
        cam2imgs = []
        for i, camera in enumerate(frame.get_camera_list()):

            assert camera == 'ring_front_center' if i == 0 else True, \
                'the first image should be the front view'

            lidar2cam_r = np.linalg.inv(frame.get_extrinsic(camera)['rotation'])
            lidar2cam_t = frame.get_extrinsic(camera)['translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t

            intrinsic = frame.get_intrinsic(camera)['K']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)

            img_paths.append(frame.get_image_path(camera))
            lidar2cam_rts.append(lidar2cam_rt.T)
            cam_intrinsics.append(viewpad)
            lidar2img_rts.append(lidar2img_rt)
            rots.append(np.linalg.inv(frame.get_extrinsic(camera)['rotation']))
            trans.append(-frame.get_extrinsic(camera)['translation'])
            cam2imgs.append(frame.get_intrinsic(camera)['K'])

        can_bus = np.zeros(18)
        rotation = Quaternion._from_matrix(frame.get_pose()['rotation'])
        can_bus[:3] = frame.get_pose()['translation']
        can_bus[3:7] = rotation
        patch_angle = rotation.yaw_pitch_roll[0] / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        lidar2global = np.eye(4)
        lidar2global[:3, :3] = np.array(frame.get_pose()['rotation'])
        lidar2global[:3, 3] = np.array(frame.get_pose()['translation'])
        global2lidar = np.linalg.inv(lidar2global)
        input_dict = {
            'scene_token': segment_id,
            'sample_idx': timestamp,
            'img_paths': img_paths,
            'lidar2cam': lidar2cam_rts,
            'cam_intrinsic': cam_intrinsics,
            'lidar2img': lidar2img_rts,
            'rots': rots,
            'trans': trans,
            'cam2imgs': cam2imgs,
            'can_bus': can_bus,
            'ego_pose': lidar2global,
            'ego_pose_inv': global2lidar,
        }

        input_dict.update(self.get_ann_info(index, input_dict))

        return input_dict

    def get_ann_info(self, index, input_dict):
        split, segment_id, timestamp = self.data_infos[index]
        frame = self.collection.get_frame_via_identifier((split, segment_id, timestamp))

        # from IPython import embed; embed()
        # import pdb; pdb.set_trace()
        # annotations = frame.get_annotations()
        # gt_lane = []
        # gt_label = []
        lcs = frame.get_annotations_lane_centerlines()
        gt_topology_lclc = frame.get_annotations_topology_lclc()
        gt_lane = np.array([lc['points'] for lc in lcs], dtype=np.float32)
        # if not self.test_mode:
        #     # when train, sample 11 pts from 201 pts
        #     gt_lane = gt_lane[:, ::20, :] # N, 11, 3
        # import pdb; pdb.set_trace()
        assert (gt_lane.shape[1] - 1) % (self.fixed_num - 1) == 0
        interval = (gt_lane.shape[1] - 1) // (self.fixed_num - 1) # 201 -> 11 / 6
        gt_lane = gt_lane[:, ::interval, :]
        if self.map_classes and len(self.map_classes) > 1:
            gt_label = np.array([lc['is_intersection_or_connector'] for lc in lcs], dtype=np.int64)
        else:
            gt_label = np.zeros((len(gt_lane), ), dtype=np.int64)
        gt_te = np.array([element['points'].flatten() for element in frame.get_annotations_traffic_elements()], dtype=np.float32).reshape(-1, 4)
        gt_te_labels = np.array([element['attribute']for element in frame.get_annotations_traffic_elements()], dtype=np.int64)
        gt_topology_lcte = frame.get_annotations_topology_lcte()
        # crop front image
        # import pdb; pdb.set_trace()
        gt_te[:, 1] -= self.crop_h[0]
        gt_te[:, 3] -= self.crop_h[0]
        mask = gt_te[:, 3] > 0
        gt_te = gt_te[mask]
        gt_te_labels = gt_te_labels[mask]
        gt_topology_lcte = gt_topology_lcte[:, mask]
        # front image
        # image = mmcv.imread(input_dict['img_paths'][0])
        # for bbox, attr in zip(gt_te, gt_te_labels):
        #     b = bbox.astype(np.int32)
        #     image = render_corner_rectangle(image, (b[0], b[1]), (b[2], b[3]), COLOR_DICT[attr], 3, 1)
        # cv2.imwrite('./test.jpg', image)
        # import pdb; pdb.set_trace()

        # for lane_segment in annotations['lane_segment']:
        #     left_laneline = lane_segment['left_laneline']
        #     if lane_segment['left_laneline_type']:
        #         gt_lane.append(left_laneline) # [N, 3]
        #         gt_label.append(0) # 0: lane
        #     right_laneline = lane_segment['right_laneline']
        #     if lane_segment['right_laneline_type']:
        #         gt_lane.append(right_laneline) # [N, 3]
        #         gt_label.append(0) # 0: lane

        # for area in annotations['area']:
        #     points = area['points'] 
        #     linetype = area['category']
        #     gt_lane.append(points) # [N, 3]
        #     gt_label.append(linetype) # 1: ped, 2: boundary

        # gt_lane = set([tuple(t.flatten()) for t in gt_lane])
        # gt_lane = np.array(list(gt_lane), dtype=np.float32).reshape(-1, self.fixed_num, self.code_size)
        
        # render pv
        # images = [mmcv.imread(img_path) for img_path in input_dict['img_paths']]
        # images = render_pv(
        #     images, input_dict['lidar2img'], 
        #     gt_lane=gt_lane, gt_label=gt_label,
        # )
        # for cam_idx, image in enumerate(images):
        #     output_path = os.path.join(f'vis/pv_{frame.get_camera_list()[cam_idx]}.jpg')
        #     mmcv.imwrite(image, output_path)
        # img_pts = [
        #     (0, 3321, 2048, 4871),
        #     (356, 1273, 1906, 3321),
        #     (356, 4871, 1906, 6919),
        #     (2048, 4096, 3598, 6144),
        #     (2048, 2048, 3598, 4096),
        #     (2048, 6144, 3598, 8192),
        #     (2048, 0, 3598, 2048),
        # ]
        # multiview = np.zeros([3598, 8192, 3], dtype=np.uint8)
        # for idx, pts in enumerate(img_pts):
        #     multiview[pts[0]:pts[2], pts[1]:pts[3]] = images[idx]
        # multiview[2048:] = multiview[2048:, ::-1]
        # multiview = cv2.resize(multiview, None, fx=0.5, fy=0.5)
        # output_path = os.path.join(f'vis/pv_multiview.jpg')
        # mmcv.imwrite(multiview, output_path)

        # render bev
        # bev_lane = render_bev(
        #     gt_lane=gt_lane, gt_label=gt_label,
        #     map_size=[-52, 55, -27, 27], scale=20,
        # )
        # output_path = os.path.join(f'vis/bev_lane.jpg')
        # mmcv.imwrite(bev_lane, output_path)

        # import pdb; pdb.set_trace()
        gt_lane_2d = []
        gt_box_2d = []
        gt_label_2d = []
        for cam_idx, (image, lidar2img) in enumerate(zip(input_dict['img_paths'], input_dict['lidar2img'])):
            # image = mmcv.imread(image)
            if cam_idx:
                # img_size = self.img_size
                img_size = (2048, 1550)
            else:
                # img_size = (self.img_size[1], self.img_size[0])
                img_size = (1550, 2048)

            gt_lane_2d_per_image = []
            gt_label_2d_per_image = []
            _gt_lane = np.array([lc['points'] for lc in lcs], dtype=np.float32)
            _gt_lane = _gt_lane[:, ::self.sample_interval, :] # N, 201, 3
            for i, lane in enumerate(_gt_lane):
                xyz1 = np.concatenate([lane, np.ones((lane.shape[0], 1))], axis=1)
                xyz1 = xyz1 @ lidar2img.T
                xyz1 = xyz1[xyz1[:, 2] > 1e-5] # maybe empty
                
                points_2d = xyz1[:, :2] / xyz1[:, 2:3]
                if cam_idx:
                    valid_flag = np.logical_and(points_2d[:, 0] >= 0, 
                                    np.logical_and(points_2d[:, 0] <= img_size[0],
                                        np.logical_and(points_2d[:, 1] >= 0, 
                                            points_2d[:, 1] <= img_size[1])))
                    valid_pts = np.where(valid_flag)
                    points_2d = points_2d[valid_pts]
                else:
                    # crop front image
                    valid_flag = np.logical_and(points_2d[:, 0] >= 0, 
                                    np.logical_and(points_2d[:, 0] <= img_size[0],
                                        np.logical_and(points_2d[:, 1] >= self.crop_h[0],
                                            points_2d[:, 1] <= self.crop_h[1])))
                    valid_pts = np.where(valid_flag)
                    points_2d = points_2d[valid_pts]
                    points_2d[:, 1] -= self.crop_h[0] #
                # if len(points_2d) >= 2:
                points_2d = interp_arc(points_2d, self.fixed_num)
                if points_2d is not None:
                    gt_lane_2d_per_image.append(points_2d)
                    gt_label_2d_per_image.append(gt_label[i])
            #         image = cv2.polylines(image, 
            #                                 [points_2d.astype(int)], 
            #                                 False, # is closed
            #                                 COLOR_DICT[i % COLOR_NUM], 
            #                                 2, # thickness
            #                             )
            # mmcv.imwrite(image, f'vis/gt_{frame.get_camera_list()[cam_idx]}.jpg')
            if len(gt_lane_2d_per_image):
                gt_lane_2d_per_image = np.stack(gt_lane_2d_per_image, axis=0).astype(np.float32)
                # [NOTE] centerline has direction
                if self.flip_gt:
                    gt_lane_2d.append(np.stack((gt_lane_2d_per_image, np.flip(gt_lane_2d_per_image, axis=1)), axis=1)) # [N, 2, 20, 3] * 7
                else:
                    gt_lane_2d.append(gt_lane_2d_per_image[:, np.newaxis]) # [N, 1, 20, 3] * 7
                # gt_label_2d.append(np.zeros((len(gt_lane_2d_per_image)), dtype=np.int64)) # [N] * 7
                gt_label_2d.append(np.array(gt_label_2d_per_image, dtype=np.int64))
                gt_box_2d.append(np.zeros((len(gt_lane_2d_per_image), 4), dtype=np.float32)) # [N, 4] * 7
            else:
                gt_lane_2d.append(np.array([], dtype=np.float32))
                gt_label_2d.append(np.array([], dtype=np.int64))
                gt_box_2d.append(np.array([], dtype=np.float32))

        if self.flip_gt:
            gt_lane = np.stack((gt_lane, np.flip(gt_lane, axis=1)), axis=1) # [N, 2, 20, 3]
        else:
            gt_lane = gt_lane[:, np.newaxis] # N, 1, 20, 3
        ''' check bezier curve
        In [24]: from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import CustomParameterizeLane
        In [25]: para = CustomParameterizeLane(method='bezier_Endpointfixed', method_para=dict(n_control=4))
        In [29]: control_pts = para.fit_bezier_Endpointfixed(gt_lane, 4)
        In [33]: def vis(gt_line_instances):
        ...:     import matplotlib.pyplot as plt
        ...:     plt.figure(figsize=(2, 4))
        ...:     plt.ylim(-50, 50)
        ...:     plt.xlim(-25, 25)
        ...:     plt.axis('off')
        ...:     for gt_line_instance in gt_line_instances:
        ...:         y = gt_line_instance[:, 0]
        ...:         x = gt_line_instance[:, 1]
        ...:         plt.plot(x, y, color='orange', linewidth=1, alpha=0.8, zorder=-1)
        ...:         plt.scatter(x, y, color='orange', s=1, alpha=0.8, zorder=-1)
        ...:     plt.savefig('bev_lane.jpg', bbox_inches='tight', format='png', dpi=1200)
        ...:     plt.close()
        '''
        return {
            'gt_3dlanes': gt_lane,
            'gt_2dlanes': gt_lane_2d,
            'gt_2dboxes': gt_box_2d,
            'gt_labels': gt_label_2d,
            'gt_labels_3d': gt_label,
            'gt_topology_lclc': gt_topology_lclc,
            'gt_te': gt_te,
            'gt_te_labels': gt_te_labels,
            'gt_topology_lcte': gt_topology_lcte,
        }
    
    def pre_pipeline(self, results):
        pass

    def prepare_train_data(self, index):
        # import pdb; pdb.set_trace()
        # stream mode: self.seq_mode
        # sliding window: self.queue_length > 1
        if not self.seq_mode and self.queue_length == 1:
            input_dict = self.get_data_info(index)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            # 2d gt num
            if len(torch.cat(example['gt_labels'].data)):
                return example
            else:
                return None
        else:
            queue = []
            index_list = list(range(index - self.queue_length - self.random_length + 1, index))
            random.shuffle(index_list)
            index_list = sorted(index_list[self.random_length:])
            index_list.append(index) # quene = 8, random = 1
            prev_scene_token = None
            for i in index_list:
                i = max(0, i)
                input_dict = self.get_data_info(i)
                if input_dict is None:
                    return None
                
                if not self.seq_mode: # for sliding window only
                    if input_dict['scene_token'] != prev_scene_token:
                        input_dict.update(dict(prev_exists=False))
                        prev_scene_token = input_dict['scene_token']
                    else:
                        input_dict.update(dict(prev_exists=True))
                else:
                    # import pdb; pdb.set_trace()
                    assert len(index_list) == 1
                    prev_exists = not (i == 0 or self.flag[i - 1] != self.flag[i])
                    input_dict.update(dict(prev_exists=prev_exists))

                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                queue.append(example)

            # 2d gt num
            for k in range(self.num_frame_losses):
                if not len(torch.cat(queue[-k-1]['gt_labels'].data)):
                    return None
            return self.union2one(queue)

    def union2one(self, queue):
        collect_keys=[
            'img',
            'gt_3dlanes', 'gt_2dlanes', 'gt_2dboxes', 'gt_labels',
            'gt_camera_extrinsic', 'gt_camera_intrinsic',
            'gt_project_matrix', 'gt_homography_matrix',
            'gt_topology_lclc', 'gt_te', 'gt_te_labels', 'gt_topology_lcte',
        ]
        # meta_keys=[
        #     'scene_token', 'sample_idx', 'img_paths', 
        #     'img_shape', 'scale_factor', 'pad_shape',
        #     'lidar2img', 'can_bus', 'ori_shape', 'prev_exists',
        # ],
        # import pdb; pdb.set_trace()
        for key in collect_keys:
            if key in ['img', 'gt_camera_extrinsic', 'gt_camera_intrinsic', 'gt_project_matrix', 'gt_homography_matrix']:
                queue[-1][key] = DC(torch.stack([each[key].data for each in queue]), stack=True)
            else:
                queue[-1][key] = DC([each[key].data for each in queue]) # stack=False, cpu_only=False
        
        queue[-1]['img_metas'] = DC([each['img_metas'].data for each in queue], cpu_only=True)
        queue = queue[-1]
        return queue

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict['scene_token'] != self.prev_scene_token:
            input_dict.update(dict(prev_exists=False))
            self.prev_scene_token = input_dict['scene_token']
        else:
            input_dict.update(dict(prev_exists=True))
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def __getitem__(self, idx):
        # import pdb; pdb.set_trace()
        if isinstance(idx, list):
            idx = idx[0] # check please!
            # from mmcv.runner import get_dist_info
            # _rank, _num_replicas = get_dist_info()
            # print('{} {}'.format(_rank, idx))
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    # def evaluate(self, 
    #              results, 
    #              logger=None,
    #              dump=None,
    #              dump_dir=None,
    #              visualization=False, 
    #              visualization_dir=None,
    #              visualization_num=None,
    #              **kwargs):
        
    #     if logger:
    #         logger.info(f'Start formating...')
    #     pred_dict = self.format_preds(results)

    #     if dump:
    #         assert dump_dir is not None
    #         assert check_results(pred_dict), "Please fill the missing keys."
    #         output_path = os.path.join(dump_dir, 'result.pkl')
    #         mmcv.dump(pred_dict, output_path)

    #     if visualization:
    #         assert visualization_dir is not None
    #         self.visualize(pred_dict, visualization_dir, visualization_num, **kwargs)
        
    #     if logger:
    #         logger.info(f'Start evaluatation...')
    #     metric_results = {}
    #     for key, val in openlanev2_evaluate(ground_truth=self.ann_file, predictions=pred_dict).items():
    #         for k, v in val.items():
    #             metric_results[k if k != 'score' else key] = v
    #     return metric_results

    def format_openlanev2_gt(self, clip_range=None, fixed_num=11):
        gt_dict = {}
        for idx in range(len(self.data_infos)):
            split, segment_id, timestamp = self.data_infos[idx]
            key = (split, segment_id, str(timestamp))
            gt_dict[key] = dict(
                lane_centerline = [],
                traffic_element=[],
                topology_lclc = None,
                topology_lcte = None,
            )
            frame = self.collection.get_frame_via_identifier((split, segment_id, timestamp))
            gt_lane = np.array([lc['points'] for lc in frame.get_annotations_lane_centerlines()], dtype=np.float32)
            gt_lane = gt_lane[:, ::20, :] # 201 -> 11
            for gt_idx, lane in enumerate(gt_lane):
                gt_dict[key]['lane_centerline'].append(dict(
                    id = 10000 + gt_idx,
                    points = lane,
                    confidence = 1.
                ))
            # if clip_range is None:
            #     pass
            # else:
            #     for gt_idx, lane in enumerate(gt_lane):
            #         pts = LineString(lane).intersection(box(*clip_range))
            #         if pts.geom_type == 'MultiLineString':
            #             for new_pts_single in pts.geoms:
            #                 if new_pts_single.is_empty:
            #                     continue
            #                 line = np.array(new_pts_single.coords, dtype=np.float32)
            #                 line = interp_arc(line, fixed_num)
            #                 if line is not None:
            #                     gt_dict[key]['lane_centerline'].append(dict(
            #                         id = 10000 + gt_idx,
            #                         points = line,
            #                         confidence = 1.
            #                     ))
            #         elif not pts.is_empty:
            #             line = np.array(pts.coords, dtype=np.float32)
            #             line = interp_arc(line, fixed_num)
            #             if line is not None:
            #                 gt_dict[key]['lane_centerline'].append(dict(
            #                     id = 10000 + gt_idx,
            #                     points = line,
            #                     confidence = 1.
            #                 ))
            gt_te = np.array([element['points'] for element in frame.get_annotations_traffic_elements()], dtype=np.float32)
            gt_te_labels = np.array([element['attribute']for element in frame.get_annotations_traffic_elements()], dtype=np.int64)
            for gt_idx, (te, te_label) in enumerate(zip(gt_te, gt_te_labels)):
                gt_dict[key]['traffic_element'].append(dict(
                    id = 20000 + gt_idx,
                    points = te,
                    attribute = te_label,
                    confidence = 1.
                ))
            gt_topology_lclc = frame.get_annotations_topology_lclc()
            gt_dict[key]['topology_lclc'] = gt_topology_lclc
            gt_topology_lcte = frame.get_annotations_topology_lcte()
            gt_dict[key]['topology_lcte'] = gt_topology_lcte
        return gt_dict
    
    def format_results(self, results, jsonfile_prefix=None, clip_range=None, fixed_num=11):
        pred_dict = {}
        pred_dict['method'] = 'dummy'
        pred_dict['authors'] = []
        pred_dict['e-mail'] = 'dummy'
        pred_dict['institution / company'] = 'dummy'
        pred_dict['country / region'] = 'CN'
        pred_dict['results'] = {}
        # lane, lane_topo, traffic_topo
        eval_topo = 'lane'
        for idx, result in enumerate(results):
            split, segment_id, timestamp = self.data_infos[idx]
            key = (split, segment_id, str(timestamp))

            pred_info = dict(
                lane_centerline=[],
                traffic_element=[],
                topology_lclc=None,
                topology_lcte=None
            )

            result = result['pts_bbox']
            lanes = result['pts_3d'].numpy() # N, 11, 2
            scores = result['scores_3d'].numpy() # N
            for pred_idx, (lane, score) in enumerate(zip(lanes, scores)):
                if clip_range is None:
                    if len(lane) < 11:
                        lane = interp_arc(lane, 11)
                    lc_info = dict(
                        id = 10000 + pred_idx,
                        points = lane,
                        confidence = score
                    )
                    pred_info['lane_centerline'].append(lc_info)
                else:
                    pts = LineString(lane).intersection(box(*clip_range))
                    if pts.geom_type == 'MultiLineString':
                        for new_pts_single in pts.geoms:
                            if new_pts_single.is_empty:
                                continue
                            line = np.array(new_pts_single.coords, dtype=np.float32)
                            line = interp_arc(line, fixed_num)
                            if line is not None:
                                lc_info = dict(
                                    id = 10000 + pred_idx,
                                    points = line,
                                    confidence = score
                                )
                                pred_info['lane_centerline'].append(lc_info)
                    elif not pts.is_empty:
                        line = np.array(pts.coords, dtype=np.float32)
                        line = interp_arc(line, fixed_num)
                        if line is not None:
                            lc_info = dict(
                                id = 10000 + pred_idx,
                                points = line,
                                confidence = score
                            )
                            pred_info['lane_centerline'].append(lc_info)

            if 'lclc_topo' in result.keys():
                pred_info['topology_lclc'] = result['lclc_topo'].numpy()
                eval_topo = 'lane_topo'

            if 'lcte_topo' in result.keys():
                pred_info['topology_lcte'] = result['lcte_topo'].numpy()
                eval_topo = 'traffic_topo'

                # te
                pred_te = result['pred_te']
                for pred_idx, (bbox, confidence) in enumerate(zip(*pred_te)):
                    # crop front image
                    bbox[1] += self.crop_h[0]
                    bbox[3] += self.crop_h[0]
                    te_info = dict(
                        id = 20000 + pred_idx,
                        points = bbox[:-1].reshape(2, 2).astype(np.float32),
                        attribute = bbox[-1],
                        confidence = confidence,
                    )
                    pred_info['traffic_element'].append(te_info)

            pred_dict['results'][key] = dict(predictions=pred_info)

        return pred_dict, eval_topo

    def evaluate(self, results, logger=None, show=False, out_dir=None, **kwargs):
        """Evaluation in Openlane-V2 subset_A dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str): Metric to be performed.
            iou_thr (float): IoU threshold for evaluation.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            show (bool): Whether to visualize the results.
            out_dir (str): Path of directory to save the results.
            pipeline (list[dict]): Processing pipeline.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        # if show:
        #     assert out_dir, 'Expect out_dir when show is set.'
        #     logger.info(f'Visualizing results at {out_dir}...')
        #     self.show(results, out_dir)
        #     logger.info(f'Visualize done.')

        # import pdb; pdb.set_trace()
        print(f'Starting format results...')
        gt_dict = self.format_openlanev2_gt()
        pred_dict, eval_topo = self.format_results(results)

        print(f'Starting openlanev2 evaluate...')
        metric_results = openlanev2_evaluate_centerline(gt_dict, pred_dict, eval_topo)
        # if self.point_cloud_range[3] - self.point_cloud_range[0] == 200:
        #     print(f'---------- MAP_RANGE = [-50, -25, 50, 25] ----------')
        #     print(f'Starting format results...')
        #     gt_dict = self.format_openlanev2_gt(clip_range=[-50, -25, 50, 25], fixed_num=11)
        #     pred_dict, eval_topo = self.format_results(results, clip_range=[-50, -25, 50, 25], fixed_num=11)
        #     print(f'Starting openlanev2 evaluate...')
        #     metric_results = openlanev2_evaluate_centerline(gt_dict, pred_dict, eval_topo)
        if self.point_cloud_range[3] - self.point_cloud_range[0] == 200:
            print(f'---------- MAP_RANGE = [-50, -25, 50, 25] ----------')
            print(f'Starting format results...')
            ann_file = '../OpenLane-V2/data/OpenLane-V2/data_dict_subset_A_val.pkl'
            self.data_infos = self.load_annotations(ann_file)
            gt_dict = self.format_openlanev2_gt()
            pred_dict, eval_topo = self.format_results(results, clip_range=[-50, -25, 50, 25], fixed_num=11)
            print(f'Starting openlanev2 evaluate...')
            # metric_results = openlanev2_evaluate_centerline(gt_dict, pred_dict, eval_topo)
            metric_results['MAP_RANGE'] = openlanev2_evaluate_centerline(gt_dict, pred_dict, eval_topo)
        # format_metric(metric_results)
        # metric_results = {
        #     'OpenLane-V2 Score': metric_results['OpenLane-V2 Score']['score'],
        #     'DET_l': metric_results['OpenLane-V2 Score']['DET_l'],
        #     'DET_t': metric_results['OpenLane-V2 Score']['DET_t'],
        #     'TOP_ll': metric_results['OpenLane-V2 Score']['TOP_ll'],
        #     'TOP_lt': metric_results['OpenLane-V2 Score']['TOP_lt'],
        # }
        return metric_results

    def format_preds(self, results):

        predictions = {
            'method': 'dummy',
            'authors': ['dummy'],
            'e-mail': 'dummy',
            'institution / company': 'dummy',
            # 'country / region': None,
            'results': {},
        }
        for index, result in enumerate(results):
            prediction = {                
                'lane_centerline': [],
                'traffic_element': [],
                'topology_lclc': None,
                'topology_lcte': None,
            }

            # lc

            pred_lc = result['pred_lc']
            lanes, confidences = pred_lc[0], pred_lc[1][:, 0]

            lanes = lanes.reshape(-1, lanes.shape[-1] // 3, 3)

            def comb(n, k):
                return factorial(n) // (factorial(k) * factorial(n - k))
            n_points = 11
            n_control = lanes.shape[1]
            A = np.zeros((n_points, n_control))
            t = np.arange(n_points) / (n_points - 1)
            for i in range(n_points):
                for j in range(n_control):
                    A[i, j] = comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
            bezier_A = torch.tensor(A, dtype=torch.float32)
            lanes = torch.tensor(lanes, dtype=torch.float32)
            lanes = torch.einsum('ij,njk->nik', bezier_A, lanes)
            lanes = lanes.numpy()

            for i, (lane, confidence) in enumerate(zip(lanes, confidences)):
                prediction['lane_centerline'].append({
                    'id': i + 1000,
                    'points': lane.astype(np.float32),
                    'confidence': confidence,
                })

            # te

            pred_te = result['pred_te']
            for i, (bbox, confidence) in enumerate(zip(*pred_te)):
                prediction['traffic_element'].append({
                    'id': i + 2000,
                    'attribute': bbox[-1],
                    'points': bbox[:-1].reshape(2, 2).astype(np.float32),
                    'confidence': confidence,
                })

            # topology

            prediction['topology_lclc'] = result['pred_topology_lclc']
            prediction['topology_lcte'] = result['pred_topology_lcte']

            #

            predictions['results'][self.data_infos[index]] = {
                'predictions': prediction,
            }

        return predictions

    def visualize(self, pred_dict, visualization_dir, visualization_num, confidence_threshold=0.3, **kwargs):
        
        assert visualization_dir, 'Please specify visualization_dir for saving visualization.'

        print('\nStart visualization...\n')
            
        for index, (key, prediction) in enumerate(pred_dict['results'].items()):
            if visualization_num and index >= visualization_num:
                print(f'\nOnly {visualization_num} frames are visualized.\n')
                return

            frame = self.collection.get_frame_via_identifier(key)
            prediction = prediction['predictions']

            # calculate metric
            pred_result = {
                'method': 'dummy',
                'authors': 'dummy',
                'results': {
                    key: {
                        'predictions': prediction,
                    }
                }
            }
            gt_result = {key: {'annotation': frame.get_annotations()}}
            try:
                metric_results = openlanev2_evaluate(gt_result, pred_result, verbose=False)
            except Exception:
                metric_results = None

            # filter lc
            pred_lc_mask = np.array([lc['confidence'] for lc in prediction['lane_centerline']]) > confidence_threshold
            pred_lc = np.array([lc['points'] for lc in prediction['lane_centerline']])[pred_lc_mask]

            # filter te
            pred_te_mask = np.array([te['confidence'] for te in prediction['traffic_element']]) > confidence_threshold
            pred_te = np.array([te['points'].flatten() for te in prediction['traffic_element']])[pred_te_mask]
            pred_te_attr = np.array([te['attribute'] for te in prediction['traffic_element']])[pred_te_mask]

            # filter topology
            pred_topology_lclc = prediction['topology_lclc'][pred_lc_mask][:, pred_lc_mask] > confidence_threshold
            pred_topology_lcte = prediction['topology_lcte'][pred_lc_mask][:, pred_te_mask] > confidence_threshold
            
            data_info = self.get_data_info(index)
            if frame.get_annotations():
                gt_lc = np.array([lc['points'] for lc in frame.get_annotations_lane_centerlines()])

                gt_te = np.array([element['points'].flatten() for element in frame.get_annotations_traffic_elements()]).reshape(-1, 4)
                gt_te_attr = np.array([element['attribute']for element in frame.get_annotations_traffic_elements()])

                gt_topology_lclc = frame.get_annotations_topology_lclc()
                gt_topology_lcte = frame.get_annotations_topology_lcte()
            else:
                gt_lc, gt_te, gt_te_attr, gt_topology_lclc, gt_topology_lcte = None, None, None, None, None

            # render pv

            images = [mmcv.imread(img_path) for img_path in data_info['img_paths']]
            images = render_pv(
                images, data_info['lidar2img'], 
                gt_lc=gt_lc, pred_lc=pred_lc, 
                gt_te=gt_te, gt_te_attr=gt_te_attr, pred_te=pred_te, pred_te_attr=pred_te_attr,
            )
            for cam_idx, image in enumerate(images):
                output_path = os.path.join(visualization_dir, f'{"/".join(key)}/pv_{frame.get_camera_list()[cam_idx]}.jpg')
                mmcv.imwrite(image, output_path)

            img_pts = [
                (0, 3321, 2048, 4871),
                (356, 1273, 1906, 3321),
                (356, 4871, 1906, 6919),
                (2048, 4096, 3598, 6144),
                (2048, 2048, 3598, 4096),
                (2048, 6144, 3598, 8192),
                (2048, 0, 3598, 2048),
            ]
            multiview = np.zeros([3598, 8192, 3], dtype=np.uint8)
            for idx, pts in enumerate(img_pts):
                multiview[pts[0]:pts[2], pts[1]:pts[3]] = images[idx]
            multiview[2048:] = multiview[2048:, ::-1]
            multiview = cv2.resize(multiview, None, fx=0.5, fy=0.5)
            output_path = os.path.join(visualization_dir, f'{"/".join(key)}/pv_multiview.jpg')
            mmcv.imwrite(multiview, output_path)

            front_view = render_front_view(
                images[0], data_info['lidar2img'][0],
                gt_lc=gt_lc, pred_lc=pred_lc, 
                gt_te=gt_te, pred_te=pred_te,
                gt_topology_lcte=gt_topology_lcte,
                pred_topology_lcte=pred_topology_lcte,
            )
            output_path = os.path.join(visualization_dir, f'{"/".join(key)}/pv_{frame.get_camera_list()[0]}_topology.jpg')
            mmcv.imwrite(front_view, output_path)

            # render bev

            if metric_results is not None:
                info = []
                for k, v in metric_results['OpenLane-V2 Score'].items():
                    if k == 'score':
                        continue
                    info.append(f'{k}: {(lambda x: "%.2f" % x)(v)}')
                info = ' / '.join(info)
            else:
                info = '-'

            bev_lane = render_bev(
                gt_lc=gt_lc, pred_lc=pred_lc, 
                map_size=[-52, 55, -27, 27], scale=20,
            )
            bev_lane = cv2.putText(bev_lane, info, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GT, 2)
            output_path = os.path.join(visualization_dir, f'{"/".join(key)}/bev_lane.jpg')
            mmcv.imwrite(bev_lane, output_path)

            bev_gt = render_bev(
                gt_lc=gt_lc,
                gt_topology_lclc=gt_topology_lclc,
                map_size=[-52, 55, -27, 27], scale=20,
            )
            bev_pred = render_bev(
                pred_lc=pred_lc,  
                pred_topology_lclc=pred_topology_lclc,
                map_size=[-52, 55, -27, 27], scale=20,
            )
            divider = np.ones((bev_gt.shape[0], 7, 3), dtype=np.uint8) * 128
            bev_topology = np.concatenate([bev_gt, divider, bev_pred], axis=1)
            bev_topology = cv2.putText(bev_topology, info, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GT, 2)
            output_path = os.path.join(visualization_dir, f'{"/".join(key)}/bev_topology.jpg')
            mmcv.imwrite(bev_topology, output_path)
