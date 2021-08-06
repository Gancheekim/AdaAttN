# AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer

> [[Paper]](http://47.103.30.151/research/PartialConvDepthLossVST.pdf) [[PyTorch Implementation]](https://github.com/Huage001/AdaAttN) [Paddle Implementation]

## Overview

This repository contains the official PyTorch implementation of paper:

*AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer*, 

Songhua Liu, Tianwei Lin, Dongliang He, Fu Li, Meiling Wang, Xin Li, Zhengxing Sun, Qian Li, Errui Ding

ICCV 2021

![](picture/picture.png)

## Prerequisites
* Linux or macOS
* Python 3
* PyTorch 1.7+ and other dependencies (torchvision, visdom, dominate, and other common python libs)

## Getting Started

* Clone this repository:

  ```shell
  git clone https://github.com/Huage001/AdaAttN
  cd AdaAttN
  ```

* Inference: 

  * Configure content_path and style_path in test_adaattn.sh firstly, indicating paths to folders of testing content images and testing style images respectively.

  * Then, simply run: 

    ```shell
    bash test_adaattn.sh
    ```

  * Check the results under results/AdaAttN folder.

* Train:

  * Download [COCO dataset](http://images.cocodataset.org/zips/train2014.zip) and [WikiArt dataset](http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip) and then extract them.

  * Configure content_path and style_path in train_adaattn.sh, indicating paths to folders of training content images and training style images respectively.
  
  * Before training, start *visdom* server:

    ```shell
    python -m visdom.server
    ```

  * Then, simply run: 
  
    ```shell
    bash train_adaattn.sh
    ```

  * You can monitor training status at http://localhost:8097/ and models would be saved at checkpoints/AdaAttN folder.
  
  * You may feel free to try other training options written in train_adaattn.sh. 

## Acknowledgments

* This implementation is developed based on the code framework of **[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)** by Junyan Zhu *et al.*