# EOS
This repository contains source code for the paper, "Efficient Augmentation for Imbalanced Deep Learning."
We provide source code, trained models and data to illustrate the application of EOS to the CIFAR-10 dataset.
Training a model with EOS consists of several steps:
1. Train a combined CNN on imbalanced data with cifar_train.py.  The best model path is saved and used in the next step.
2. Extract lower dimensional feature embeddings from the trained CNN for each training instance and save the feature embeddings, using cifar_FE.py. 
3. Perform minority class data augmentation in feature embedding space with EOS, using EOS.py and save output.
4. Train a classifier with EOS generated minority class augmentations plus the original training data in FE space, with cifar_train_os.py.
5. Reassemble the CNN so that the original extraction network contains an updated classifier, with reassemble.py.

Please note that we trained our models with a single NVIDIA 3070 GPU using the following packages:
- cudatoolkit 11.1.1
- imbalanced learn 0.9.0
- python 3.7.1
- pytorch 1.9.0
- scikit learn 1.0.2
- spyder 5.0.5
- tensorboard 1.15.0
- torchvision 0.10.0
