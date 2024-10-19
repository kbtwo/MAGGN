# MAGGN

Classification of thermophilic proteins based on multi-aspect gated graph neural networks

# Document introduction

1.The dataset folder: The three datasets used in the experiment are: training dataset、independent  test set1、independent  test set2。

2.The magnn-ipynb folder: Two ways of using the model are provided, and this folder is the IPYNB file. Recommend using IPYNB files.

3.The magnn-py folder: Two ways of using the model are provided, and this folder is the PY file.The Graph_prepare.py file is used to predict protein contact maps, the Datasprocess.cy file is used for data preprocessing, and the Traint_test. py file is used for model training and testing. The GRU. py file is the source code for GRU gating, and the MAGGN. py file is the source code for the MAGGN model.

4.The pssm folder: This folder contains position specific scoring matrices for all protein sequences.

5.The MAGNN.model file: This file is the weight file of the model.

# How to use MAGGN

This model provides two ways of use, and we recommend using IPYNB files. Load the IPYNB file into the compiler and execute it sequentially.