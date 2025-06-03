## Prepare NuScenes Dataset

### For evaluation

* Download nuScenes V1.0 full dataset data from [here](https://www.nuscenes.org/download) and place it in `./data/nuscenes/`.

* Download occupancy ground truth for [Occ3D-nuScenes](https://tsinghua-mars-lab.github.io/Occ3D/) and place it in `./data/nucenes/gts`.

* Download [pkl files](https://www.ipb.uni-bonn.de/html/projects/sfmocc2025/pkl_files.zip) and place them in `./data/nuscenes/`:

* Download pretrained [weights](https://www.ipb.uni-bonn.de/html/projects/sfmocc2025/sfm_occ.pth) and put them in `./ckpt`


### For training
* Download [depth images](https://www.ipb.uni-bonn.de/html/projects/sfmocc2025/sfm_depth.zip) and place them in `.data/nuscenes/`.

* Download [pseudo-labels](https://www.ipb.uni-bonn.de/html/projects/sfmocc2025/sfm_occ.zip) and place them in `./data/nuscenes/`.



**Directory structure**
```
sfm_occ
├── mmdet3d/
├── tools/
├── configs/
├── ckpts/
│   ├── sfm_occ.pth
├── data/
│   ├── nuscenes/
│   │   ├── gts/
│   │   ├── samples/
|   |   ├── nuscenes_infos_train_sfm.pkl
|   |   ├── nuscenes_infos_val.pkl
|   |   ├── nuscenes_infos_test.pkl
│   │   ├── sfm_occ/
|   |   ├── sfm_depth/
|   |   └── v1.0-{trainval}
|   |       └── Usual files (e.g. attribute.json, calibrated_sensor.json etc.)


```