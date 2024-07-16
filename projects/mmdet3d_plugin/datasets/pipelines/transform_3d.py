import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from math import factorial

@PIPELINES.register_module()
class CustomParameterizeLane:

    def __init__(self, method, method_para):
        method_list = ['bezier', 'polygon', 'bezier_Direction_attribute', 'bezier_Endpointfixed']
        self.method = method
        if not self.method in method_list:
            raise Exception("Not implemented!")
        self.method_para = method_para

    def __call__(self, results):
        centerlines = results['gt_lc']
        para_centerlines = getattr(self, self.method)(centerlines, **self.method_para)
        results['gt_lc'] = para_centerlines
        return results

    def comb(self, n, k):
        return factorial(n) // (factorial(k) * factorial(n - k))

    def fit_bezier(self, points, n_control):
        n_points = len(points)
        A = np.zeros((n_points, n_control))
        t = np.arange(n_points) / (n_points - 1)
        for i in range(n_points):
            for j in range(n_control):
                A[i, j] = self.comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
        conts = np.linalg.lstsq(A, points, rcond=None)
        return conts

    def fit_bezier_Endpointfixed(self, points, n_control):
        n_points = len(points)
        A = np.zeros((n_points, n_control))
        t = np.arange(n_points) / (n_points - 1)
        for i in range(n_points):
            for j in range(n_control):
                A[i, j] = self.comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
        A_BE = A[1:-1, 1:-1]
        _points = points[1:-1]
        _points = _points - A[1:-1, 0].reshape(-1, 1) @ points[0].reshape(1, -1) - A[1:-1, -1].reshape(-1, 1) @ points[-1].reshape(1, -1)

        conts = np.linalg.lstsq(A_BE, _points, rcond=None)

        control_points = np.zeros((n_control, points.shape[1]))
        control_points[0] = points[0]
        control_points[-1] = points[-1]
        control_points[1:-1] = conts[0]

        return control_points

    def bezier(self, input_data, n_control=2):

        coeffs_list = []
        for idx, centerline in enumerate(input_data):
            sorted_x = np.array(centerline[:, 1])
            sorted_y = np.array(centerline[:, 0])
            points = np.array(list(zip(sorted_x, sorted_y)))
            res = self.fit_bezier(points, n_control)[0]
            start_res = res[0]
            end_res = res[-1]
            first_diff = (np.sum(np.square(start_res - points[0]))) + (np.sum(np.square(end_res - points[-1])))
            second_diff = (np.sum(np.square(start_res - points[-1]))) + (np.sum(np.square(end_res - points[0])))

            if first_diff <= second_diff:
                fin_res = res
            else:
                fin_res = np.zeros_like(res)
                for m in range(len(res)):
                    fin_res[len(res) - m - 1] = res[m]

            fin_res = np.clip(fin_res, 0, 1)
            coeffs_list.append(np.reshape(np.float32(fin_res), (-1)))

        return np.array(coeffs_list)

    def bezier_Direction_attribute(self, input_data, n_control=3):
        coeffs_list = []
        for idx, centerline in enumerate(input_data):
            centerline[:, 1] = centerline[:, 1]
            centerline[:, 0] = centerline[:, 0]
            sorted_x = np.array(centerline[:, 1])
            sorted_y = np.array(centerline[:, 0])
            points = np.array(list(zip(sorted_x, sorted_y)))
            res = self.fit_bezier(points, n_control)[0]
            fin_res = np.clip(res, 0, 1)
            start_res = res[0]
            end_res = res[-1]
            first_diff = (np.sum(np.square(start_res - points[0]))) + (np.sum(np.square(end_res - points[-1])))
            second_diff = (np.sum(np.square(start_res - points[-1]))) + (np.sum(np.square(end_res - points[0])))
            if first_diff <= second_diff:
                da = 0
            else:
                da = 1
            fin_res = np.append(fin_res, da)
            coeffs_list.append(np.reshape(np.float32(fin_res), (-1)))
        return np.array(coeffs_list)

    def bezier_Endpointfixed(self, input_data, n_control=2):
        coeffs_list = []
        for idx, centerline in enumerate(input_data):
            res = self.fit_bezier_Endpointfixed(centerline, n_control)
            coeffs = res.flatten()
            coeffs_list.append(coeffs)
        return np.array(coeffs_list, dtype=np.float32)

    def polygon(self, input_data, key_rep='Bounding Box'):
        keypoints = []
        for idx, centerline in enumerate(input_data):
            centerline[:, 1] = centerline[:, 1]
            centerline[:, 0] = centerline[:, 0]
            sorted_x = np.array(centerline[:, 1])
            sorted_y = np.array(centerline[:, 0])
            points = np.array(list(zip(sorted_x, sorted_y)))
            if key_rep not in ['Bounding Box', 'SME', 'Extreme Points']:
                raise Exception(f"{key_rep} not existed!")
            elif key_rep == 'Bounding Box':
                res = np.array(
                    [points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()]).reshape((2, 2))
                keypoints.append(np.reshape(np.float32(res), (-1)))
            elif key_rep == 'SME':
                res = np.array([points[0], points[-1], points[int(len(points) / 2)]])
                keypoints.append(np.reshape(np.float32(res), (-1)))
            else:
                min_x = np.min([points[:, 0] for p in points])
                ind_left = np.where(points[:, 0] == min_x)
                max_x = np.max([points[:, 0] for p in points])
                ind_right = np.where(points[:, 0] == max_x)
                max_y = np.max([points[:, 1] for p in points])
                ind_top = np.where(points[:, 1] == max_y)
                min_y = np.min([points[:, 1] for p in points])
                ind_botton = np.where(points[:, 1] == min_y)
                res = np.array(
                    [points[ind_left[0][0]], points[ind_right[0][0]], points[ind_top[0][0]], points[ind_botton[0][0]]])
                keypoints.append(np.reshape(np.float32(res), (-1)))
        return np.array(keypoints)

@PIPELINES.register_module()
class CustomPadMultiViewImage:

    def __init__(self, size_divisor=None, pad_val=0):
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def __call__(self, results):
        # max_h = max([img.shape[0] for img in results['img']])
        # max_w = max([img.shape[1] for img in results['img']])
        # padded_img = [mmcv.impad(img, shape=(max_h, max_w), pad_val=self.pad_val) for img in results['img']]
        padded_img = results['img']
        if self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in padded_img]
        
        results['img'] = padded_img
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = None
        results['pad_size_divisor'] = self.size_divisor

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str

@PIPELINES.register_module()
class CropFrontViewImageOpenLaneV2(object):

    def __init__(self, crop_h=(356, 1906), pad_val=0):
        self.crop_h = crop_h
        self.pad_val = pad_val

    def _crop_img(self, results):
        # crop
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'][0] = results['img'][0][self.crop_h[0]:self.crop_h[1]]
        # pad
        max_h = max([img.shape[0] for img in results['img']])
        max_w = max([img.shape[1] for img in results['img']])
        padded_img = [mmcv.impad(img, shape=(max_h, max_w), pad_val=self.pad_val) for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in results['img']]
        # pad all view
        self.size_divisor = 64 # 32 / 0.5
        padded_img = [mmcv.impad_to_multiple(
            img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]    
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = None
        results['pad_size_divisor'] = self.size_divisor

    def _crop_cam_intrinsic(self, results):
        results['cam_intrinsic'][0][1, 2] -= self.crop_h[0]
        results['lidar2img'][0] = results['cam_intrinsic'][0] @ results['lidar2cam'][0]

    def _crop_bbox(self, results):
        if 'gt_te' in results.keys():
            results['gt_te'][:, 1] -= self.crop_h[0]
            results['gt_te'][:, 3] -= self.crop_h[0]

            mask = results['gt_te'][:, 3] > 0
            results['gt_te'] = results['gt_te'][mask]
            results['gt_te_labels'] = results['gt_te_labels'][mask]
            if 'gt_topology_lcte' in results.keys():
                results['gt_topology_lcte'] = results['gt_topology_lcte'][:, mask]

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._crop_img(results)
        self._crop_cam_intrinsic(results)
        # self._crop_bbox(results)
        # filename = results['img_paths']
        # img = mmcv.imread(filename[0])
        # _img = img[self.crop_h[0]:self.crop_h[1]]
        # import cv2
        # for i, lane in enumerate(results['gt_3dlanes']):
        #     lane = lane[0] # direction
        #     lidar2img = results['lidar2img'][0]

        #     xyz1 = np.concatenate([lane, np.ones((lane.shape[0], 1))], axis=1)
        #     xyz1 = xyz1 @ lidar2img.T
        #     xyz1 = xyz1[xyz1[:, 2] > 1e-5] # maybe empty
        #     points_2d = xyz1[:, :2] / xyz1[:, 2:3]
            
        #     valid_flag = np.logical_and(points_2d[:, 0] >= 0, 
        #                     np.logical_and(points_2d[:, 0] <= 1550,
        #                         np.logical_and(points_2d[:, 1] >= 0,
        #                             points_2d[:, 1] <= 1550)))
        #     valid_pts = np.where(valid_flag)
        #     points_2d = points_2d[valid_pts]

        #     if len(points_2d >= 2):  
        #         _img = cv2.polylines(_img, [points_2d.astype(int)], False, (255, 0, 0), 2)
        # for i, lane in enumerate(results['gt_2dlanes'][0]):
        #     lane = lane[0] # direction
        #     _img = cv2.polylines(_img, [lane.astype(int)], False, (255, 0, 0), 2)
        # mmcv.imwrite(_img, 'test.jpg')
        return results

@PIPELINES.register_module()
class ResizeFrontView:

    def __init__(self, img_scale):
        self.w = img_scale[0] # 800
        self.h = img_scale[1] # 480
        pass

    def __call__(self, results):
        assert 'ring_front_center' in results['img_paths'][0], \
            'the first image should be the front view'

        # image
        front_view = results['img'][0]
        h, w, _ = front_view.shape
        resiezed_front_view, w_scale, h_scale = mmcv.imresize(
            front_view,
            (h, w),
            return_scale=True,
        )
        results['img'][0] = resiezed_front_view
        results['img_shape'][0] = resiezed_front_view.shape

        # gt
        scale_factor = np.array(
            [w_scale, h_scale, w_scale, h_scale],
            dtype=np.float32,
        )
        results['scale_factor'] = scale_factor
        if 'gt_2dlanes' in results and len(results['gt_2dlanes'][0]):
            results['gt_2dlanes'][0] = results['gt_2dlanes'][0] * results['scale_factor'][:2]

        # intrinsic
        lidar2cam_r = results['rots'][0]
        lidar2cam_t = (-results['trans'][0]) @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t

        intrinsic = results['cam2imgs'][0]
        viewpad = np.eye(4)
        viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

        cam_s = np.eye(4)
        cam_s[0, 0] *= w_scale
        cam_s[1, 1] *= h_scale

        viewpad = cam_s @ viewpad 
        intrinsic = viewpad[:intrinsic.shape[0], :intrinsic.shape[1]]
        lidar2img_rt = (viewpad @ lidar2cam_rt.T)

        results['cam_intrinsic'][0] = viewpad
        results['lidar2img'][0] = lidar2img_rt
        results['cam2imgs'][0] = intrinsic

        # [NOTE] scale all view img here
        for idx in range(len(results['img'])):
            view = results['img'][idx]
            resiezed_view, w_scale, h_scale = mmcv.imresize(
                view,
                (self.w, self.h),
                return_scale=True,
            )
            results['img'][idx] = resiezed_view
        return results

@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb


    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        # import pdb; pdb.set_trace()
        if not isinstance(imgs, list):
            imgs = [imgs]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        if not isinstance(results['img'], list):
            new_imgs = new_imgs[0]
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str



@PIPELINES.register_module()
class CustomCollect3D(object):
    """Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".
    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:
        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'scene_token',
                            'can_bus','lidar2global',
                            'camera2ego','camera_intrinsics','img_aug_matrix','lidar2ego'
                            )):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.
        Args:
            results (dict): Result dict contains the data to collect.
        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
       
        data = {}
        img_metas = {}
      
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'

@PIPELINES.register_module()
class CustomLoadRandomScaleImageMultiViewImage(object):
    """Random scale the image
    Args:
        scales
    """

    def __init__(self, scales=[]):
        self.scales = scales
        assert len(self.scales)==1

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]

        y_size = [int(img.shape[0] * rand_scale) for img in results['img']]
        x_size = [int(img.shape[1] * rand_scale) for img in results['img']]
        results['img'] = [mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False) for idx, img in
                          enumerate(results['img'])]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str


@PIPELINES.register_module()
class RandomScaleImageMultiViewImage(object):
    """Random scale the image
    Args:
        scales
    """

    def __init__(self, scales=[]):
        self.scales = scales
        assert len(self.scales)==1

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]

        y_size = [int(img.shape[0] * rand_scale) for img in results['img']]
        x_size = [int(img.shape[1] * rand_scale) for img in results['img']]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results['img'] = [mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False) for idx, img in
                          enumerate(results['img'])]
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        img_aug_matrix = [scale_factor for _ in results['lidar2img']]
        results['lidar2img'] = lidar2img
        results['img_aug_matrix'] = img_aug_matrix
        results['img_shape'] = [img.shape for img in results['img']]
        results['ori_shape'] = [img.shape for img in results['img']]

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str


@PIPELINES.register_module()
class CustomPointsRangeFilter:
    """Filter points by the range.
    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, data):
        """Call function to filter points by the range.
        Args:
            data (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask' \
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = data["points"]
        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = points[points_mask]
        data["points"] = clean_points
        return data