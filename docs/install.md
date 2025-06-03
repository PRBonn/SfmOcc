# Installation instructions

We use python 3.8.10

**Install torch and torchvision**
```shell
pip3 install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

**Install mmcv-full**
```shell
pip3 install mmcv-full==1.6.0
```

**Install mmdet and mmseg**
```shell
pip3 install mmdet==2.24.0
pip3 install mmsegmentation==0.24.0
```

**Install sfm_occ from source code**
```shell
pip3 install -U -e .
```

**Install requirements**
```shell
pip3 install -r requirements.txt
```
