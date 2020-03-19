# Pytorch Panoptic DeepLab

The objective of this repository is to create the panoptic deeplab model and training pipeline as presented in the [paper](https://arxiv.org/pdf/1911.10194.pdf). The code base is adopted from the [pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception) repository. Much will be changed as time goes on, and when all objectives have been achieved, the name of the repo will be changed to reflect the updated content.

## Solution Pipeline

We already have a semantic segmentation branch of DeepLabV3+, so the rest of the pipeline will be as follows;

- [X] Add an instance decoder head
- [X] Groundtruth instance centers encoding by a 2D Gaussian with standard deviation of 8 pixels
- [X] Add compound loss(criterion) function.
- [X] Panoptic Dataloader for Citiscapes
- [X] Apply 5×5 depthwise-separable convolution. (Not as in the paper)
- [X] Test training with slim backbone (mobileNet), and
- [X] Training with xception backbone

## Running on google cloud platform

Cityscapes dataset can be downloaded to a cloud platform as explained [here](https://github.com/reger-men/keras_multi_gpus/wiki/How-to-download-cityscapes-dataset-via-terminal). Install Anaconda environment with the `environment.yml` file.


---

**See the original README for DeepLab Xception for Semantic Segmentation below below:**

## pytorch-deeplab-xception

**Update on 2018/12/06. Provide model trained on VOC and SBD datasets.**  

**Update on 2018/11/24. Release newest version code, which fix some previous issues and also add support for new backbones and multi-gpu training. For previous code, please see in `previous` branch**  

### TODO

- [x] Support different backbones
- [x] Support VOC, SBD, Cityscapes and COCO datasets
- [x] Multi-GPU training

| Backbone  | train/eval os | mIoU in val |                                  Pretrained Model                                  |
| :-------- | :-----------: | :---------: | :--------------------------------------------------------------------------------: |
| ResNet    |     16/16     |   78.43%    | [google drive](https://drive.google.com/open?id=1NwcwlWqA-0HqAPk3dSNNPipGMF0iS0Zu) |
| MobileNet |     16/16     |   70.81%    | [google drive](https://drive.google.com/open?id=1G9mWafUAj09P4KvGSRVzIsV_U5OqFLdt) |
| DRN       |     16/16     |   78.87%    | [google drive](https://drive.google.com/open?id=131gZN_dKEXO79NknIQazPJ-4UmRrZAfI) |

### Introduction

This is a PyTorch(0.4.1) implementation of [DeepLab-V3-Plus](https://arxiv.org/pdf/1802.02611). It
can use Modified Aligned Xception and ResNet as backbone. Currently, we train DeepLab V3 Plus
using Pascal VOC 2012, SBD and Cityscapes datasets.

![Results](doc/results.png)

### Installation

The code was tested with Anaconda and Python 3.6. After installing the Anaconda environment:

1. Clone the repo:

    ```Shell
    git clone https://github.com/jfzhang95/pytorch-deeplab-xception.git
    cd pytorch-deeplab-xception
    ```

2. Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    For custom dependencies:

    ```Shell
    pip install matplotlib pillow tensorboardX tqdm
    ```

### Training

Follow steps below to train your model:

1. Configure your dataset path in [mypath.py](https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/mypath.py).

2. Input arguments: (see full input arguments via python train.py --help):

    ```Shell
    usage: train.py [-h] [--backbone {resnet,xception,drn,mobilenet}]
                [--out-stride OUT_STRIDE] [--dataset {pascal,coco,cityscapes}]
                [--use-sbd] [--workers N] [--base-size BASE_SIZE]
                [--crop-size CROP_SIZE] [--sync-bn SYNC_BN]
                [--freeze-bn FREEZE_BN] [--loss-type {ce,focal}] [--epochs N]
                [--start_epoch N] [--batch-size N] [--test-batch-size N]
                [--use-balanced-weights] [--lr LR]
                [--lr-scheduler {poly,step,cos}] [--momentum M]
                [--weight-decay M] [--nesterov] [--no-cuda]
                [--gpu-ids GPU_IDS] [--seed S] [--resume RESUME]
                [--checkname CHECKNAME] [--ft] [--eval-interval EVAL_INTERVAL]
                [--no-val]

    ```

3. To train deeplabv3+ using Pascal VOC dataset and ResNet as backbone:

    ```Shell
    bash train_voc.sh
    ```

4. To train deeplabv3+ using COCO dataset and ResNet as backbone:

    ```Shell
    bash train_coco.sh
    ```

### Acknowledgement

[PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)

[Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)

[drn](https://github.com/fyu/drn)
