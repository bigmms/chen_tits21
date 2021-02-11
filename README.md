[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fbigmms%2Fchen_tits21&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
# Deep Trident Decomposition Network for Single License Plate Image Glare Removal

## Introduction
We propose a deep trident decomposition network with a large-scale sun glare image dataset for glare removal from single images. Specifically, the proposed network is designed and implemented with a trident decomposition module for decomposing an input glare image into occlusion, foreground, and coarse glare-free images by exploring background features from spatial locations. Moreover, a residual refinement module is adopted to refine the coarse glare-free image into fine glare-free image by learning the residuals from features of multiscale receptive field.

**Authors**: Bo-Hao Chen, Shiting Ye, Jia-Li Yin, Hsiang-Yin Cheng, and Dewang Chen

**Paper**: [PDF](https://ieeexplore.ieee.org/document/9325516)

## Requirements
* Python 3.5
* numpy 1.15.0
* openCV 4.3.0.38
* keras 2.3.1
* tensorflow 1.14.0
* scikit-image 0.14.2
* pillow 6.2.1
* sewar 0.4.3
* segmentation-models 0.2.1
