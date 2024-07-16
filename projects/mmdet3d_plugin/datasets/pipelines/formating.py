
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet3d.core.points import BasePoints
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets.pipelines import DefaultFormatBundle3D

@PIPELINES.register_module()
class CustomDefaultFormatBundleOpenLaneV2:

    def __init__(self):
        pass

    def __call__(self, results):

        temp = to_tensor(np.concatenate([i[None, ...] for i in results['img']], axis=0))
        results['img'] = DC(temp.permute(0, 3, 1, 2), stack=True)
        
        if 'gt_lc' in results:
            results['gt_lc'] = DC(to_tensor(results['gt_lc']))
        if 'gt_lc_labels' in results:
            results['gt_lc_labels'] = DC(to_tensor(results['gt_lc_labels']))
        if 'gt_te' in results:
            results['gt_te'] = DC(to_tensor(results['gt_te']))
        if 'gt_te_labels' in results:
            results['gt_te_labels'] = DC(to_tensor(results['gt_te_labels']))
        if 'gt_topology_lclc' in results:
            results['gt_topology_lclc'] = DC(to_tensor(results['gt_topology_lclc']))
        if 'gt_topology_lcte' in results:
            results['gt_topology_lcte'] = DC(to_tensor(results['gt_topology_lcte']))
        
        for key in ['gt_3dlanes', 'gt_2dlanes', 'gt_2dboxes', 'gt_labels', 'gt_labels_3d']:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))

        gt_camera_intrinsic = np.stack(results['cam_intrinsic'], axis=0)
        gt_camera_extrinsic = np.zeros_like(gt_camera_intrinsic)
        gt_project_matrix = np.stack(results['lidar2img'], axis=0)
        gt_homography_matrix = np.zeros_like(gt_project_matrix)
        results['gt_camera_extrinsic'] = DC(to_tensor(gt_camera_extrinsic.astype(np.float32)), stack=True)
        results['gt_camera_intrinsic'] = DC(to_tensor(gt_camera_intrinsic.astype(np.float32)), stack=True)
        results['gt_project_matrix'] = DC(to_tensor(gt_project_matrix.astype(np.float32)), stack=True)
        results['gt_homography_matrix'] = DC(to_tensor(gt_homography_matrix.astype(np.float32)), stack=True)

        return results

@PIPELINES.register_module()
class CustomDefaultFormatBundleOpenLaneV2FrontView:

    def __init__(self):
        pass

    def __call__(self, results):

        # N, H, W, C -> N, C, H, W
        # temp = to_tensor(np.concatenate([i[None, ...] for i in results['img']], axis=0))
        # results['img'] = DC(temp.permute(0, 3, 1, 2), stack=True)
        temp = [to_tensor(i).permute(2, 0, 1) for i in results['img']] # H, W, C -> C, H, W
        results['img'] = DC(temp, stack=False) # (C, H, W) * N
        
        if 'gt_lc' in results:
            results['gt_lc'] = DC(to_tensor(results['gt_lc']))
        if 'gt_lc_labels' in results:
            results['gt_lc_labels'] = DC(to_tensor(results['gt_lc_labels']))
        if 'gt_te' in results:
            results['gt_te'] = DC(to_tensor(results['gt_te']))
        if 'gt_te_labels' in results:
            results['gt_te_labels'] = DC(to_tensor(results['gt_te_labels']))
        if 'gt_topology_lclc' in results:
            results['gt_topology_lclc'] = DC(to_tensor(results['gt_topology_lclc']))
        if 'gt_topology_lcte' in results:
            results['gt_topology_lcte'] = DC(to_tensor(results['gt_topology_lcte']))
        
        for key in ['gt_3dlanes', 'gt_2dlanes', 'gt_2dboxes', 'gt_labels', 'gt_labels_3d']:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))

        gt_camera_intrinsic = np.stack(results['cam_intrinsic'], axis=0)
        gt_camera_extrinsic = np.zeros_like(gt_camera_intrinsic)
        gt_project_matrix = np.stack(results['lidar2img'], axis=0)
        gt_homography_matrix = np.zeros_like(gt_project_matrix)
        results['gt_camera_extrinsic'] = DC(to_tensor(gt_camera_extrinsic.astype(np.float32)), stack=True)
        results['gt_camera_intrinsic'] = DC(to_tensor(gt_camera_intrinsic.astype(np.float32)), stack=True)
        results['gt_project_matrix'] = DC(to_tensor(gt_project_matrix.astype(np.float32)), stack=True)
        results['gt_homography_matrix'] = DC(to_tensor(gt_homography_matrix.astype(np.float32)), stack=True)

        return results

@PIPELINES.register_module()
class LaneFormat(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and other lane data. These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if len(img.shape) > 3:
                # [H, W, 3, N] -> [3, H, W, N]
                img = np.ascontiguousarray(img.transpose(2, 0, 1, 3))
            else:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(to_tensor(img), stack=True)
        for key in ['gt_3dlanes', 'gt_2dlanes', 'gt_2dboxes', 'gt_labels']:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))
        # [BUG] DO NOT STACK
        if 'seg_idx_label' in results:
            results['seg_idx_label'] = DC(to_tensor(results['seg_idx_label']))
        # if 'gt_3dlanes' in results:
        #     results['gt_3dlanes'] = DC(to_tensor(results['gt_3dlanes'].astype(np.float32)))
        # if 'gt_2dlanes' in results:
        #     results['gt_2dlanes'] = DC(to_tensor(results['gt_2dlanes'].astype(np.float32)))
        if 'gt_camera_extrinsic' in results:
            results['gt_camera_extrinsic'] = DC(to_tensor(results['gt_camera_extrinsic'][None, ...].astype(np.float32)), stack=True)
        if 'gt_camera_intrinsic' in results:
            results['gt_camera_intrinsic'] = DC(to_tensor(results['gt_camera_intrinsic'][None, ...].astype(np.float32)), stack=True)
        if 'gt_project_matrix' in results:
            results['gt_project_matrix'] = DC(to_tensor(results['gt_project_matrix'][None, ...].astype(np.float32)), stack=True)
        if 'gt_homography_matrix' in results:
            results['gt_homography_matrix'] = DC(to_tensor(results['gt_homography_matrix'][None, ...].astype(np.float32)), stack=True)
        if 'gt_camera_pitch' in results:
            results['gt_camera_pitch'] = DC(to_tensor([results['gt_camera_pitch']]))
        if 'gt_camera_height' in results:
            results['gt_camera_height'] = DC(to_tensor([results['gt_camera_height']]))
        return results

    def __repr__(self):
        return self.__class__.__name__

@PIPELINES.register_module()
class CustomDefaultFormatBundle3D(DefaultFormatBundle3D):
    """Default formatting bundle.
    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.
    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict contains the data to convert.
        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        results = super(CustomDefaultFormatBundle3D, self).__call__(results)
        results['gt_map_masks'] = DC(
            to_tensor(results['gt_map_masks']), stack=True)

        return results