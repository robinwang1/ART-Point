# ART-Point: Improving Rotation Robustness of Point Cloud Classifiers via Adversarial Rotation

![image](https://github.com/robinwang1/ART-Point/blob/main/assets/fig1.png)

## Introduction
PyTorch implementation for the paper [ART-Point: Improving Rotation Robustness of Point Cloud Classifiers via Adversarial Rotation (CVPR 2022)](http://arxiv.org/abs/2203.03888).

Repository still under construction/refactoring. 

## Installation
#### Install Requirements
    $ cd ART-Point/
    $ conda env create -f environment.yaml

#### Download ModelNet40 and ShapeNet Parts
We use two datasets:
* [ModeNet40](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)
* [ShapeNet16](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)

After downloading, you should convert the .txt dataset into numpy file (.npy). Then, you can use our code for training and evaluation.
You can use the codes in "https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master/data_utils" for pre-pocessing.

#### Pretraining Models
We use the folloing implemetations to respectively pretrain classifiers on ModelNet40 and ShapeNet16.
* [DGCNN](https://github.com/WangYueFt/dgcnn/tree/master/pytorch)
* [PointNet/PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

After pre-training, you should move the pre-trained models into corresponding folders at "./pretrained_models/"

## Train and Evaluate

#### ModelNet40
To train and evaluate ART-Point with one-step optimization on ModelNet40 using PointNet backends run: 
```
$ python train_classification_onestep.py --angles 1 --batch_size 17 --inner_epoch 200 --iters 10 --log_dir pn1_onestep --rp
```

To train and evaluate ART-Point with iterative optimization on ModelNet40 using PointNet backend run: 
```
$ python train_classification_dynamic.py --angles 1 --batch_size 17 --epoch 50 --inner_epoch 50 --iters 10 --log_dir pn1_dynamic --rp
```

#### ShapeNet16

To train and evaluate ART-Point with one-step optimization on ShapeNet16 using PointNet backends run: 
```
$ python train_classification_onestep_s16.py --angles 1 --batch_size 17 --inner_epoch 200 --iters 10 --log_dir pn1_onestep_s16 --rp
```

## Contact 
You are welcome to send pull requests or share some ideas with us. Contact information: Robin Wang (robin_wang@pku.edu.cn).

