# Copyright (c) Phigent Robotics. All rights reserved.
import torch

from mmdet.models import DETECTORS
from .bevdet import BEVStereo4D


@DETECTORS.register_module()
class BEVStereo4DOCC(BEVStereo4D):
    def __init__(self, **kwargs):
        super(BEVStereo4DOCC, self).__init__(**kwargs)

    def prepare_inputs(self, inputs, stereo=False):
        # split the inputs into each frame
        B, N, C, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, C, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = inputs[1:7]

        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego   # 取的是第一帧、第一个相机时间戳下的pose作为key
        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = (
            sensor2keyegos.float()
        )  # B, T, N, 4, 4     #这里得到了所有时序、相机帧 keyego的pose！

        curr2adjsensor = None
        if stereo:
            sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
            sensor2egos_curr = sensor2egos_cv[:, : self.temporal_frame, ...].double()
            ego2globals_curr = ego2globals_cv[:, : self.temporal_frame, ...].double()
            sensor2egos_adj = sensor2egos_cv[
                :, 1 : self.temporal_frame + 1, ...
            ].double()
            ego2globals_adj = ego2globals_cv[
                :, 1 : self.temporal_frame + 1, ...
            ].double()
            curr2adjsensor = (
                torch.inverse(ego2globals_adj @ sensor2egos_adj)
                @ ego2globals_curr
                @ sensor2egos_curr
            )
            curr2adjsensor = curr2adjsensor.float()
            curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
            curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
            curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
            assert len(curr2adjsensor) == self.num_frame

        extra = [
            sensor2keyegos,  # camera2keyego
            ego2globals,
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(
                B, self.num_frame, N, 3, 3
            ),  # ida_aug，不同相机aug不同，同一相机不同时间戳共享aug参数
            post_trans.view(
                B, self.num_frame, N, 3
            ),  # ida_aug，不同相机aug不同，同一相机不同时间戳共享aug参数
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra
        return (
            imgs,
            sensor2keyegos,
            ego2globals,
            intrins,
            post_rots,
            post_trans,
            bda,
            curr2adjsensor,
        )

    def prepare_bev_feat(
        self,
        img,
        sensor2keyego,
        ego2global,
        intrin,
        post_rot,
        post_tran,
        bda,
        mlp_input,
        feat_prev_iv,
        k2s_sensor,
        extra_ref_frame,
        depth_gt=None,
    ):
        if extra_ref_frame:
            stereo_feat = self.extract_stereo_ref_feat(img)
            return None, None, stereo_feat
        x, stereo_feat = self.image_encoder(img, stereo=True)
        metas = dict(
            k2s_sensor=k2s_sensor,
            intrins=intrin,
            post_rots=post_rot,
            post_trans=post_tran,
            frustum=self.img_view_transformer.cv_frustum.to(x),
            cv_downsample=4,
            downsample=self.img_view_transformer.downsample,
            grid_config=self.img_view_transformer.grid_config,
            cv_feat_list=[feat_prev_iv, stereo_feat],
        )

        bev_feat, depth = self.img_view_transformer(
            [x, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda, mlp_input],
            metas,
            depth_gt,
        )
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, depth, stereo_feat

    def extract_img_feat(
        self, img_inputs, img_metas, pred_prev=False, sequential=False, **kwargs
    ):
        (
            imgs,
            sensor2keyegos,
            ego2globals,
            intrins,
            post_rots,
            post_trans,
            bda,
            curr2adjsensor,
        ) = img_inputs
        """Extract features of images."""
        bev_feat_list = []
        depth_key_frame = None
        feat_prev_iv = None

        for fid in range(self.num_frame - 1, -1, -1):
            img, sensor2keyego, ego2global, intrin, post_rot, post_tran = (
                imgs[fid],
                sensor2keyegos[fid],
                ego2globals[fid],
                intrins[fid],
                post_rots[fid],
                post_trans[fid],
            )
            key_frame = fid == 0
            extra_ref_frame = fid == self.num_frame - self.extra_ref_frames
            if key_frame or self.with_prev:
                mlp_input = self.img_view_transformer.get_mlp_input(
                    sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda
                )

                depth_gt = None

                inputs_curr = (
                    img,
                    sensor2keyego,
                    ego2global,
                    intrin,
                    post_rot,
                    post_tran,
                    bda,
                    mlp_input,
                    feat_prev_iv,
                    curr2adjsensor[fid],
                    extra_ref_frame,
                    depth_gt,
                )
                if key_frame:
                    bev_feat, depth, feat_curr_iv = self.prepare_bev_feat(*inputs_curr)
                    depth_key_frame = depth
                else:
                    with torch.no_grad():
                        bev_feat, depth, feat_curr_iv = self.prepare_bev_feat(
                            *inputs_curr
                        )
                if not extra_ref_frame:
                    bev_feat_list.append(bev_feat)
                feat_prev_iv = feat_curr_iv

        if not self.with_prev:
            bev_feat_key = bev_feat_list[0]
            if len(bev_feat_key.shape) == 4:
                b, c, h, w = bev_feat_key.shape
                bev_feat_list = [
                    torch.zeros(
                        [b, c * (self.num_frame - self.extra_ref_frames - 1), h, w]
                    ).to(bev_feat_key),
                    bev_feat_key,
                ]
            else:
                b, c, z, h, w = bev_feat_key.shape
                bev_feat_list = [
                    torch.zeros(
                        [b, c * (self.num_frame - self.extra_ref_frames - 1), z, h, w]
                    ).to(bev_feat_key),
                    bev_feat_key,
                ]
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        return [x], depth_key_frame
