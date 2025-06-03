# SfmOcc: Vision-Based 3D Semantic Occupancy Prediction in Urban Environments

This repository contains the implementation of our [paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/marcuzzi2023ral.pdf).

## Getting Started

- [Installation](docs/install.md)

- [Prepare data](docs/prepare_data.md)

- Train 
  
  ```bash
  # Single GPU
  python3 tools/train.py ./configs/sfmocc/sfmocc.py

  # 8 GPUs
  ./tools/dist_train.sh ./configs/sfmocc/sfmocc.py 8
  ```

- Evaluation 
  
  ```bash
  # Single GPU
  python3 tools/test.py ./configs/sfmocc/sfmocc.py ./path/to/ckpts.pth

  # 8 GPUs
  ./tools/dist_test.sh ./configs/sfmocc/sfmocc.py ./path/to/ckpts.pth 8
  ```

- Visualization
  
  ```bash
  # Dump predictions
  python3 tools/test.py configs/sfmocc/sfmocc.py ./path/to/ckpt.pth --dump_dir=pred_dir

  # Visualization (select scene-id)
  python tools/visualization/visual.py pred_dir/scene-xxxx
  ```

## Acknowledgement

Many thanks to the authors of [RenderOcc](https://github.com/pmj110119/RenderOcc) for the codebase.

## Citation

```bibtex
@article{marcuzzi2025ral,
  title={},
  author={},
  journal={},
  year={2025}
}
```

## Licence
Copyright 2025, Rodrigo Marcuzzi, Cyrill Stachniss, Photogrammetry and Robotics Lab, University of Bonn.

This project is free software made available under the MIT License. For details see the LICENSE file
