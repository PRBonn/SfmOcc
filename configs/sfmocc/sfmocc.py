_base_ = ["./bevstereo-occ.py"]

model = dict(
    type="SfmOcc",
    in_dim=32,
    out_dim=32,
    num_classes=18,
    test_threshold=0.6,
    w_free=0.15,
    loss_occ=dict(
        type="CrossEntropyLoss",
        use_sigmoid=False,
        class_weight=[1, 0.15],
        loss_weight=1.0,
    ),
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=5,
)
runner = dict(type="EpochBasedRunner", max_epochs=12)
log_config = dict(interval=50)
