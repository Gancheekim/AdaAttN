# AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer

## (子鈞) To use this:
1. 把該裝的library都裝好（有點太多了，你可以等噴bug的時候再去裝）
2. 下載pretrained model: 
  - mkdir checkpoints
  - download pretrained model from [Google Drive](https://drive.google.com/file/d/1XvpD1eI4JeCBIaW5uwMT6ojF_qlzM_lo/view?usp=sharing), move it to checkpoints directory, and unzip:

      ```shell
      mv [Download Directory]/AdaAttN_model.zip checkpoints/
      unzip checkpoints/AdaAttN_model.zip
      rm checkpoints/AdaAttN_model.zip
      ```
3. 這時候就可以用了，command是：
    ```
    python test.py --name ./AdaAttN_model/AdaAttN --model adaattn --dataset_mode unaligned --image_encoder_path checkpoints/AdaAttN_model/vgg_normalised.pth --gpu_ids 0 --skip_connection_3 --shallow_layer --content_path <path_to_content_img_dir> --style_path <path_to_style_img_dir>
    ```
    請記得把--gpus_ids換成-1，如果使用cpu的話  
    注: <path_to_content_img_dir> 和 <path_to_style_img_dir> 都是directory name，不是filename，把所有想要用的content img和style img放入資料及内就能自動全部弄好


> [[Paper](https://arxiv.org/abs/2108.03647)] [[PyTorch Implementation](https://github.com/Huage001/AdaAttN)] [[Paddle Implementation](https://github.com/PaddlePaddle/PaddleGAN)]

<a href="https://replicate.ai/huage001/adaattn"><img src="https://img.shields.io/static/v1?label=Replicate&message=Demo and Docker Image&color=blue"></a>
## Overview

This repository contains the **officially unofficial** PyTorch **re-**implementation of paper:

*AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer*, 

Songhua Liu, Tianwei Lin, Dongliang He, Fu Li, Meiling Wang, Xin Li, Zhengxing Sun, Qian Li, Errui Ding

ICCV 2021

![](picture/picture.png)

## Updates
* [2022-12-07] Upload script of user control. Please see *user_specify_demo.py*
* [2022-12-07] Upload inference code of video style transfer. Please see *inference_frame.py*. Please download checkpoints from [here](https://drive.google.com/file/d/1ZrFPrbSR56WXL2l62Jh7qjmuaWugJtvL/view?usp=sharing) and extract the package to the main directory of this repo before running.

## Prerequisites
* Linux or macOS
* Python 3
* PyTorch 1.7+ and other dependencies (torchvision, visdom, dominate, and other common python libs)

* ## Getting Started

  * Clone this repository:

    ```shell
    git clone https://github.com/Huage001/AdaAttN
    cd AdaAttN
    ```

  * Inference: 

    * Make a directory for checkpoints if there is not:

      ```shell
      mkdir checkpoints
      ```

    * Download pretrained model from [Google Drive](https://drive.google.com/file/d/1XvpD1eI4JeCBIaW5uwMT6ojF_qlzM_lo/view?usp=sharing), move it to checkpoints directory, and unzip:

      ```shell
      mv [Download Directory]/AdaAttN_model.zip checkpoints/
      unzip checkpoints/AdaAttN_model.zip
      rm checkpoints/AdaAttN_model.zip
      ```

    * Configure content_path and style_path in test_adaattn.sh firstly, indicating paths to folders of testing content images and testing style images respectively.

    * Then, simply run: 

      ```shell
      bash test_adaattn.sh
      ```

    * Check the results under results/AdaAttN folder.

  * Train:

    * Download 'vgg_normalised.pth' from [here](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing).

    * Download [COCO dataset](http://images.cocodataset.org/zips/train2014.zip) and [WikiArt dataset](http://web.fsktm.um.edu.my/~cschan/source/ICIP2017/wikiart.zip) and then extract them.

    * Configure content_path, style_path, and image_encoder_path in train_adaattn.sh, indicating paths to folders of training content images, training style images, and 'vgg_normalised.pth' respectively.

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

  ## Citation

  * If you find ideas or codes useful for your research, please cite:

    ```
    @inproceedings{liu2021adaattn,
      title={AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer},
      author={Liu, Songhua and Lin, Tianwei and He, Dongliang and Li, Fu and Wang, Meiling and Li, Xin and Sun, Zhengxing and Li, Qian and Ding, Errui},
      booktitle={Proceedings of the IEEE International Conference on Computer Vision},
      year={2021}
    }
    ```

  ## Acknowledgments

  * This implementation is developed based on the code framework of **[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)** by Junyan Zhu *et al.*
