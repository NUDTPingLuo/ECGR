# ECGR: Exploratory–Convergent Gradient Re-aggregation for Federated Learning

This repository provides an official PyTorch implementation of **ECGR (Exploratory–Convergent Gradient Re-aggregation)**, a plug-and-play optimization strategy designed to improve the stability and generalization of **federated learning (FL)** under **heterogeneous (non-IID) data distributions**.

ECGR decomposes local update dynamics into *exploratory* and *convergent* gradient components and selectively re-aggregates local gradients to suppress noisy or conflicting updates while preserving effective descent directions. The proposed strategy can be seamlessly integrated into existing FL baselines without modifying local training procedures.

---

## Features

* **Federated Learning Baselines**: PyTorch implementations of representative FL baselines and their ECGR-enhanced variants. The currently supported methods include:
  + [FedAvg](https://arxiv.org/abs/1602.05629) (H. B. McMahan et al., AISTATS 2017)
  + [FedNova](https://arxiv.org/abs/2007.07481) (J. Wang et al., NeurIPS 2020) [:octocat:](https://github.com/JYWa/FedNova)
  + [FedProx](https://arxiv.org/abs/1812.06127) (T. Li et al., MLSys 2020) [:octocat:](https://github.com/litian96/FedProx)
  + [SCAFFOLD](https://arxiv.org/abs/1910.06378) (S. P. Karimireddy et al., ICML 2020) [:octocat:](https://github.com/ki-ljl/Scaffold-Federated-Learning)

* **ECGR Aggregation Module**: Implementation of **Exploratory–Convergent Gradient Re-aggregation (ECGR)** as a modular aggregation strategy that can be combined with existing FL baselines to improve robustness under non-IID data distributions.

* **Dataset Preprocessing**: Automated downloading and preprocessing of benchmark datasets, followed by partitioning into multiple clients according to federated learning settings. Non-IID data distributions are simulated via Dirichlet-based label skew. The currently supported datasets include MNIST, Fashion-MNIST, SVHN, CIFAR-10, and CIFAR-100. Other datasets (e.g., medical imaging datasets) need to be downloaded and organized manually.

* **Postprocessing and Visualization**: Tools for visualizing training dynamics and evaluating global model performance, including testing accuracy curves averaged over multiple random seeds.

---

## Installation

### Dependencies

- Python (3.8)
- PyTorch (1.8.1)
- OpenCV (4.5)
- NumPy (1.21.5)

### Install Requirements

Install all required packages by running:

```bash
pip install -r requirements.txt

### Federated Dataset Preprocessing

This preprocessing module partitions the entire dataset into a specified number of clients according to federated learning settings.  
To simulate realistic heterogeneous environments, datasets are split into **non-IID local datasets** with **label distribution skew**, which is controlled by a Dirichlet distribution.

Specifically, each client is assigned a subset of samples drawn from a Dirichlet distribution over class labels, resulting in statistically heterogeneous local data distributions across clients. Smaller values of the Dirichlet concentration parameter correspond to higher degrees of data heterogeneity, which better reflect real-world federated learning scenarios.

The preprocessing pipeline supports:
- IID and non-IID data partitioning
- Dirichlet-based label skew with configurable concentration parameter
- Flexible number of clients and samples per client
- Consistent data splits for reproducible experiments

---

## Running Federated Learning with ECGR

### Test Run

All hyperparameters are specified in a YAML configuration file (e.g., `./config/test_config.yaml`).  
To run federated learning experiments with ECGR or baseline methods, execute:

```bash
python fl_main.py --config "./config/test_config.yaml"
