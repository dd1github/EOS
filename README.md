# EOS
This repository contains source code for the paper, "Efficient Augmentation for Imbalanced Deep Learning."
We provide source code, trained models and data to illustrate the application of EOS on the CIFAR-10 dataset.
Training a model with EOS consists of several steps:
1. Train a combined CNN on imbalanced data with cifar_train.py.  The best model path is saved and used in the next step.
2. Extract lower dimensional feature embeddings from the trained CNN for each training instance and save the feature embeddings, using cifar_FE.py. 
