# MF-FedAvg: Privacy preserving neural collaborative filtering using federated learning

This repository comtains the implementation of MF-FedAvg from the paper "Federated Neural Collaborative Filtering" written by Vasileios Perifanis and Pavlos S. Efraimidis. [(Paper)](https://arxiv.org/abs/2106.04405)

This paper proposed two averaging algorithms:

1. MF-FedAvg: It is an extention of FedAvg algortihms to handle the latent parameters of Matrix Factorization (MF)

2. MF-SecAvg: An extention of MF-FedAvg addressing the issues of privacy leakage during communications rounds by adversaries and HBC (Honest But Curious) clients. This is done by implementing a secure weights aggregation strategy using homomorphic encryption.

The dataset used in this implementation is [MovieLens 100k](https://grouplens.org/datasets/movielens/)

`config.ini` contains parameters used for the federated setting and training.

<img src="ncf_img.png" alt="drawing" width="600" margin="auto" style="display:block;float:none;margin-left:automargin-right:auto"/>

NOTE: This repository only contains the implementation of the MF-FedAvg algorithm.
