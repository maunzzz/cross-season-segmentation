# A Cross-Season Correspondence Dataset for Robust Semantic Segmentation
This is an implementation of the work published in A Cross-Season Correspondence Dataset for Robust Semantic Segmentation (https://arxiv.org/abs/1903.06916)

## Resources 
The datasets used in the paper will be published at visuallocalization.net
At the moment they are only available upon request

## Trained Models
https://drive.google.com/open?id=14joxT0XFreW1WX3M8oTiCV69hZTiJTMV

## Installation
A Dockerfile is provided, either build a docker image using this or refer to the requirements listed in the file.

## Usage 
- Download Cityscapes and Mapillary Vistas
- Use /utils/convert_vistas_to_cityscapes.py to create cityscapes class annotations for the Vistas images
- Download the correspondence datasets
- Download the images associated with the correspondence datasets (instructions available in dataset readme)
- Create a global_otps.json and set the paths (see global_opts_example.json)
- Train, see train/train_many.py for reproduction of paper experiments

## Reference

If you use this code, please cite the following paper:

Måns Larsson, Erik Stenborg, Lars Hammarstrand, Torsten Sattler, Mark Pollefeys and Fredrik Kahl 
"A Cross-Season Correspondence Dataset for Robust Semantic Segmentation" Proc. CVPR (2019).

```
@article{larsson2019corr,
  title={A Cross-Season Correspondence Dataset for Robust Semantic Segmentation},
  author={Larsson, M{\aa}ns and Stenborg, Erik and Hammarstrand, Lars and Sattler, Torsten and Pollefeys, Mark and Kahl, Fredrik},
  journal={arXiv preprint arXiv:1903.06916},
  year={2019}
}
```

## Other
Some code from https://github.com/zijundeng/pytorch-semantic-segmentation and https://github.com/kazuto1011/pspnet-pytorch was used.