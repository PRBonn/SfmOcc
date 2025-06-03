# Copyright (c) OpenMMLab. All rights reserved.
import os

import numpy as np
from tqdm import tqdm

from .builder import DATASETS
from .nuscenes_dataset import NuScenesDataset
from .occ_metrics import Metric_mIoU


@DATASETS.register_module()
class NuScenesDatasetOccpancy(NuScenesDataset):
    def __init__(
        self,
        depth_gt_path=None,
        sfm_depth_threshold=100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth_gt_path = depth_gt_path
        self.sfm_depth_threshold = sfm_depth_threshold

    def get_data_info(self, index):
        input_dict = super(NuScenesDatasetOccpancy, self).get_data_info(index)
        if self.test_mode:
            occ_path = "occ_path"
        else:
            occ_path = "sfm_occ_path"
        input_dict["occ_path"] = self.data_infos[index][occ_path]
        input_dict["curr_depth"] = self.get_curr_depth(index)
        return input_dict

    def get_curr_depth(self, index):
        info = self.data_infos[index]
        curr_depth = []
        curr_coor = []
        for cam_name in info["cams"].keys():
            img_file_path = info["cams"][cam_name]["data_path"]
            if self.test_mode:
                coor, label_depth = [], []
            else:
                coor, label_depth = self.load_sfm_depth(img_file_path)
            curr_depth.append(label_depth.copy())
            curr_coor.append(coor.copy())
        return [curr_depth, curr_coor]

    def load_sfm_depth(self, img_file_path):
        fname = os.path.basename(img_file_path)[:-4]
        depth_fname = os.path.join(self.depth_gt_path, fname + ".npz")
        img = np.load(depth_fname)["depth"]
        img[img > self.sfm_depth_threshold] = 0  # depth filtering
        coords = np.argwhere(img != 0).astype(np.int16)
        coords = coords[:, [1, 0]]  # swap camera axis
        depth_label = img[coords[:, 1], coords[:, 0]]
        return coords, depth_label

    def evaluate(self, occ_results):
        self.occ_eval_miou = Metric_mIoU(num_classes=18)
        self.occ_eval_iou = Metric_mIoU(num_classes=2)
        print("\nStarting Evaluation...")
        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            occ_gt = np.load(os.path.join(info["occ_path"], "labels.npz"))
            gt_semantics = occ_gt["semantics"]
            mask_lidar = occ_gt["mask_lidar"].astype(bool)
            mask_camera = occ_gt["mask_camera"].astype(bool)
            self.occ_eval_miou.add_batch(
                occ_pred, gt_semantics, mask_lidar, mask_camera
            )
            gt_semantics_iou, occ_pred_iou = sem2occ(gt_semantics, occ_pred)
            self.occ_eval_iou.add_batch(
                occ_pred_iou, gt_semantics_iou, mask_lidar, mask_camera
            )
        eval_dict = {}
        eval_dict["mIoU"] = self.occ_eval_miou.count_miou()[2]
        eval_dict["IoU"] = self.occ_eval_iou.count_miou()[2]
        return eval_dict


def sem2occ(gt_semantics, occ_pred):
    gt_semantics = gt_semantics.copy()
    occ_pred = occ_pred.copy()
    # 17: empty --> 0:free, 1: occupied
    gt_semantics[gt_semantics == 0] = 1
    gt_semantics[gt_semantics == 17] = 0
    gt_semantics[gt_semantics > 0] = 1
    occ_pred[occ_pred == 0] = 1
    occ_pred[occ_pred == 17] = 0
    occ_pred[occ_pred > 0] = 1
    return gt_semantics, occ_pred
