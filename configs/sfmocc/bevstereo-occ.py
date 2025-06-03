_base_ = ["../_base_/datasets/nus-3d.py", "../_base_/default_runtime.py"]

# Global
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

data_config = {
    "cams": [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ],
    "Ncams": 6,
    "input_size": (512, 1408),
    "src_size": (900, 1600),
    # Augmentation
    "resize": (-0.06, 0.11),
    "rot": (-5.4, 5.4),
    "flip": True,
    "crop_h": (0.0, 0.0),
    "resize_test": 0.00,
}

# Model
grid_config = {
    "x": [-40, 40, 0.4],
    "y": [-40, 40, 0.4],
    "z": [-1, 5.4, 0.4],
    "depth": [1.0, 45.0, 0.5],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 32

multi_adj_frame_id_cfg = (1, 1 + 1, 1)

model = dict(
    type="BEVStereo4DOCC",
    num_adj=len(range(*multi_adj_frame_id_cfg)),
    img_backbone=dict(
        type="SwinTransformer",
        pretrain_img_size=224,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        use_abs_pos_embed=False,
        return_stereo_feat=True,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="LN", requires_grad=True),
        pretrain_style="official",
        output_missing_index_as_none=False,
    ),
    img_neck=dict(
        type="FPN_LSS",
        in_channels=512 + 1024,
        out_channels=512,
        # with_cp=False,
        extra_upsample=None,
        input_feature_index=(0, 1),
        scale_factor=2,
    ),
    img_view_transformer=dict(
        type="LSSViewTransformerBEVStereo",
        grid_config=grid_config,
        input_size=data_config["input_size"],
        in_channels=512,
        out_channels=numC_Trans,
        sid=False,
        collapse_z=False,
        loss_depth_weight=0.05,
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96, stereo=True, bias=5.0),
        downsample=16,
    ),
    img_bev_encoder_backbone=dict(
        type="CustomResNet3D",
        numC_input=numC_Trans * (len(range(*multi_adj_frame_id_cfg)) + 1),
        num_layer=[1, 2, 4],
        with_cp=False,
        num_channels=[numC_Trans, numC_Trans * 2, numC_Trans * 4],
        stride=[1, 2, 2],
        backbone_output_ids=[0, 1, 2],
    ),
    img_bev_encoder_neck=dict(
        type="LSSFPN3D", in_channels=numC_Trans * 7, out_channels=numC_Trans
    ),
    pre_process=dict(
        type="CustomResNet3D",
        numC_input=numC_Trans,
        with_cp=False,
        num_layer=[
            1,
        ],
        num_channels=[
            numC_Trans,
        ],
        stride=[
            1,
        ],
        backbone_output_ids=[
            0,
        ],
    ),
)

# Data
dataset_type = "NuScenesDatasetOccpancy"
data_root = "data/nuscenes/"
file_client_args = dict(backend="disk")

bda_aug_conf = dict(
    rot_lim=(-0.0, 0.0), scale_lim=(1.0, 1.0), flip_dx_ratio=0.5, flip_dy_ratio=0.5
)

train_pipeline = [
    dict(type="PrepareImageInputs", is_train=True, data_config=data_config),
    dict(type="LoadOccGTFromFile"),
    dict(
        type="LoadAnnotationsBEVDepth",
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True,
    ),
    dict(type="GenerateDepthSup", data_config=data_config),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="Collect3D",
        keys=[
            "img_inputs",
            "gt_depth",
            "voxel_semantics",
            "mask_lidar",
            "mask_camera",
        ],
    ),
]

test_pipeline = [
    dict(type="PrepareImageInputs", is_train=False, data_config=data_config),
    dict(type="LoadOccGTFromFile"),
    dict(
        type="LoadAnnotationsBEVDepth",
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False,
    ),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="DefaultFormatBundle3D", class_names=class_names, with_label=False
            ),
            dict(
                type="Collect3D",
                keys=[
                    "img_inputs",
                    "voxel_semantics",
                    "mask_lidar",
                    "mask_camera",
                ],
            ),
        ],
    ),
]

input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False
)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    stereo=True,
    filter_empty_gt=False,
    img_info_prototype="bevdet4d",
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
)

ann_file_train = data_root + "nuscenes_infos_train_sfm.pkl"
ann_file_val = data_root + "nuscenes_infos_val.pkl"
ann_file_test = data_root + "nuscenes_infos_test.pkl"

train_data_config = dict(
    pipeline=train_pipeline,
    ann_file=ann_file_train,
    depth_gt_path=data_root + "sfm_depth",
    test_mode=False,
    use_valid_flag=True,
    sfm_depth_threshold=100,
)
val_data_config = dict(
    pipeline=test_pipeline,
    ann_file=ann_file_val,
)
test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=ann_file_test,
)

data = dict(
    train=train_data_config,
    val=val_data_config,
    test=test_data_config,
)

for key in ["val", "train", "test"]:
    data[key].update(share_data_config)

# Optimizer
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[
        100,
    ],
)
custom_hooks = [
    dict(
        type="MEGVIIEMAHook",
        init_updates=10560,
        priority="NORMAL",
    ),
]
evaluation = dict(interval=3, pipeline=test_pipeline)
load_from = "./ckpts/bevdet-stbase-4d-stereo-512x1408-cbgs.pth"
