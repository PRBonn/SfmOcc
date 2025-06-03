import argparse
import os
import pickle as pkl
import platform
import sys
import time
from copy import deepcopy
from functools import reduce
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import KDTree
from termcolor import colored
from tqdm import tqdm

np.seterr(divide="ignore", invalid="ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Metric_mIoU:
    def __init__(self, num_classes=18, use_image_mask=True):
        self.class_names = [
            "others",
            "barrier",
            "bicycle",
            "bus",
            "car",
            "construction_vehicle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "trailer",
            "truck",
            "driveable_surface",
            "other_flat",
            "sidewalk",
            "terrain",
            "manmade",
            "vegetation",
            "free",
        ]
        if num_classes == 2:
            self.class_names = ["free", "occupied"]
        self.use_image_mask = use_image_mask
        self.num_classes = num_classes

        self.point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
        self.occupancy_size = [0.4, 0.4, 0.4]
        self.voxel_size = 0.4
        self.occ_xdim = int(
            (self.point_cloud_range[3] - self.point_cloud_range[0])
            / self.occupancy_size[0]
        )
        self.occ_ydim = int(
            (self.point_cloud_range[4] - self.point_cloud_range[1])
            / self.occupancy_size[1]
        )
        self.occ_zdim = int(
            (self.point_cloud_range[5] - self.point_cloud_range[2])
            / self.occupancy_size[2]
        )
        self.voxel_num = self.occ_xdim * self.occ_ydim * self.occ_zdim
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

    def reset(self):
        self.hist = np.zeros((self.num_classes, self.num_classes))
        self.cnt = 0

    def hist_info(self, n_cl, pred, gt):
        """
        build confusion matrix
        # empty classes:0
        non-empty class: 0-16
        free voxel class: 17

        Args:
            n_cl (int): num_classes_occupancy
            pred (1-d array): pred_occupancy_label
            gt (1-d array): gt_occupancu_label

        Returns:
            tuple:(hist, correctly number_predicted_labels, num_labelled_sample)
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl**2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    def per_class_iu(self, hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def compute_mIoU(self, pred, label, n_classes):
        hist = np.zeros((n_classes, n_classes))
        new_hist, correct, labeled = self.hist_info(
            n_classes, pred.flatten(), label.flatten()
        )
        hist += new_hist
        mIoUs = self.per_class_iu(hist)
        return round(np.nanmean(mIoUs) * 100, 2), hist

    def add_batch(self, semantics_pred, semantics_gt, mask_lidar, mask_camera):
        self.cnt += 1
        if self.use_image_mask:
            masked_semantics_gt = semantics_gt[mask_camera]
            masked_semantics_pred = semantics_pred[mask_camera]
        else:
            masked_semantics_gt = semantics_gt
            masked_semantics_pred = semantics_pred

        _, _hist = self.compute_mIoU(
            masked_semantics_pred, masked_semantics_gt, self.num_classes
        )
        self.hist += _hist

    def count_miou(self):
        mIoU = self.per_class_iu(self.hist)
        print(f"===> per class IoU of {self.cnt} samples:")
        for ind_class in range(self.num_classes - 1):
            print(
                f"===> {self.class_names[ind_class]} - IoU = "
                + str(round(mIoU[ind_class] * 100, 2))
            )

        if self.num_classes == 2:
            meanIou = round(np.nanmean(mIoU[1]) * 100, 2)
        else:
            meanIou = round(np.nanmean(mIoU[: self.num_classes - 1]) * 100, 2)

        print(f"===> mIoU of {self.cnt} samples: " + str(meanIou))
        return self.class_names, mIoU, meanIou, self.cnt
