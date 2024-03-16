# InfoNet PyTorch Implementation

This repository contains a PyTorch implementation of the article "InfoNet: An Efficient Feed-Forward Neural Estimator for Mutual Information." Please note that this implementation is an initial version and may be subject to further updates and improvements.

## Getting Started

### Description

This is a differentiable version of infonet Pytorch implementation, which uses linear scale instead of rank data as a data preprocessing. The prediction result might be not that good compared to the previous version and we are still working on it to improve its overall performance. 

### Training InfoNet from Scratch

To train an InfoNet model from scratch, you can use the `train.py` script. This script will guide you through the process of initializing and training your model using the default or custom datasets.

### Fine-Tuning on Pretrained Checkpoints

If you have a pre-trained InfoNet model and wish to fine-tune it on a new dataset (some distributions that cannot be expressed by GMM well) or improve its performance, the `train.py` script also supports fine-tuning. Simply provide the path to your existing checkpoint, and the script will resume training from there.

### Estimating Mutual Information

For applications involving mutual information estimation using InfoNet, the `infer.py` script is designed to facilitate this process. It allows you to estimate mutual information values based on your trained model and input data. Pre-trained checkpoint can be found in: [Download Checkpoint](https://drive.google.com/file/d/1yh9cwdh08sHDsRjl2hEOiPHXby_6Yl7j/view?usp=drive_link)

