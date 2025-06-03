# Copyright (c) Phigent Robotics. All rights reserved.
from mmdet.models.builder import build_loss
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn.bricks.conv_module import ConvModule
from torch import nn

from mmdet.models import DETECTORS
from .bevdet_occ import BEVStereo4DOCC


@DETECTORS.register_module()
class SfmOcc(BEVStereo4DOCC):
    def __init__(
        self,
        in_dim=32,
        out_dim=32,
        num_classes=18,
        test_threshold=0.6,
        w_free=1.0,
        loss_occ=None,
        **kwargs
    ):
        super(SfmOcc, self).__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.test_threshold = test_threshold
        self.w_free = w_free
        self.num_classes = num_classes

        cls_freq = sfm_class_freq
        class_weights = torch.from_numpy(1 / np.log(cls_freq[:17] + 0.001)).float()
        class_weights[class_weights < 0] = 0
        self.class_weights = class_weights

        self.semantic_loss = nn.CrossEntropyLoss(
            weight=self.class_weights, reduction="mean"
        )
        self.final_conv = ConvModule(
            self.in_dim,
            self.out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type="Conv3d"),
        )
        self.density_mlp = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim * 2),
            nn.Softplus(),
            nn.Linear(self.out_dim * 2, 2),
            nn.Softplus(),
        )
        self.semantic_mlp = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim * 2),
            nn.Softplus(),
            nn.Linear(self.out_dim * 2, (num_classes - 1)),
        )
        self.loss_occ = build_loss(loss_occ)

    def forward_train(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        img_inputs = self.prepare_inputs(img_inputs, stereo=True)
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        voxel_feats = self.final_conv(img_feats[0])
        voxel_feats = voxel_feats.permute(0, 4, 3, 2, 1)

        density_prob = self.density_mlp(voxel_feats)
        semantic = self.semantic_mlp(voxel_feats)

        losses = dict()
        voxel_semantics = kwargs["voxel_semantics"]
        # mask out non observed voxels
        sfm_mask = kwargs["mask_camera"]
        # 17:free, 18: occupied but uncertain, keep for occupancy, mask out for semantics
        assert voxel_semantics.min() >= 0 and voxel_semantics.max() <= 18
        loss_occ = self.loss_sfm_3d(voxel_semantics, density_prob, semantic, sfm_mask)
        losses.update(loss_occ)

        loss_depth = self.img_view_transformer.get_depth_loss(kwargs["gt_depth"], depth)
        losses["loss_lss_depth"] = loss_depth
        return losses

    def loss_sfm_3d(self, voxel_semantics, density_prob, semantic, sfm_mask):
        voxel_semantics = voxel_semantics.long()
        voxel_semantics = voxel_semantics.reshape(-1)
        density_prob = density_prob.reshape(-1, 2)
        semantic = semantic.reshape(-1, self.num_classes - 1)
        density_target = (voxel_semantics == 17).long()
        semantic_mask = voxel_semantics != 17
        # mask out class 18 for semantics: occupied but uncertain class
        semantic_mask = torch.logical_and(semantic_mask, voxel_semantics != 18)

        loss = dict()
        # apply sfm mask to mask out voxels with no label
        sfm_mask = sfm_mask.reshape(-1)
        num_total_samples = sfm_mask.sum()
        semantic_mask = torch.logical_and(semantic_mask, sfm_mask)
        loss_geo = self.loss_occ(
            density_prob, density_target, sfm_mask, avg_factor=num_total_samples
        )
        loss["loss_3d_geo"] = loss_geo

        loss_sem = self.semantic_loss(
            semantic[semantic_mask], voxel_semantics[semantic_mask].long()
        )
        loss["loss_3d_sem"] = loss_sem

        return loss

    def simple_test(self, points, img_metas, img, rescale=False, **kwargs):
        img_inputs = self.prepare_inputs(img, stereo=True)
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas, **kwargs)
        voxel_feats = self.final_conv(img_feats[0])
        voxel_feats = voxel_feats.permute(0, 4, 3, 2, 1)

        density_prob = self.density_mlp(voxel_feats)
        density = F.softmax(density_prob, dim=-1)[..., 0]
        no_empty_mask = density > self.test_threshold

        semantic = self.semantic_mlp(voxel_feats)
        semantic_res = semantic.argmax(-1)

        occ = torch.ones_like(semantic_res) * (self.num_classes - 1)
        occ[no_empty_mask] = semantic_res[no_empty_mask]
        occ = occ.squeeze(dim=0).cpu().numpy().astype(np.uint8)
        return [occ]


sfm_class_freq = np.array(
    [
        10228045930,
        48677027,
        1491649,
        3249318,
        27963205,
        971892,
        1356472,
        2590408,
        414744,
        3425268,
        13323052,
        72806042,
        0,
        20364579,
        26816381,
        112609310,
        54216509,
        2916213963,
        53304251,
    ]
)
