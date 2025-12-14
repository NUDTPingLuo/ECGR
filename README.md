# ECGR
ECGR: Exploratory–Convergent Gradient Re-aggregation for Federated Learning. ECGR is a plug-and-play gradient re-aggregation strategy for robust federated learning under heterogeneous data.

* **Current Baseline Implementations**: PyTorch implementations of representative federated learning baselines and their ECGR-enhanced variants. The currently supported methods include FedAvg, FedNova, FedProx, and SCAFFOLD, each optionally integrated with the proposed ECGR strategy:
  + [FedAvg](https://arxiv.org/abs/1602.05629) (H. B. McMahan et al., AISTATS 2017)
  + [FedNova](https://arxiv.org/abs/2007.07481) (J. Wang et al., NeurIPS 2020) [:octocat:](https://github.com/JYWa/FedNova)
  + [FedProx](https://arxiv.org/abs/1812.06127) (T. Li et al., MLSys 2020) [:octocat:](https://github.com/litian96/FedProx)
  + [SCAFFOLD](https://arxiv.org/abs/1910.06378) (S. P. Karimireddy et al., ICML 2020) [:octocat:](https://github.com/ki-ljl/Scaffold-Federated-Learning)

* **ECGR Aggregation Module**: Implementation of **Exploratory–Convergent Gradient Re-aggregation (ECGR)**, a plug-and-play optimization strategy for federated learning under heterogeneous (non-IID) data distributions. ECGR can be seamlessly combined with existing FL baselines to improve convergence stability and generalization.

* **Dataset Preprocessing**: Automated downloading and preprocessing of benchmark datasets, followed by partitioning into multiple clients according to federated learning settings. Non-IID data distributions are simulated via Dirichlet-based label skew. The currently supported datasets include MNIST, Fashion-MNIST, SVHN, CIFAR-10, and CIFAR-100. Other datasets (e.g., medical imaging datasets) need to be downloaded and organized manually.

* **Postprocessing and Visualization**: Tools for visualizing training dynamics and evaluating global model performance, including accuracy curves averaged over multiple random seeds.

---

## Installation

### Dependencies

- Python (3.8)
- PyTorch (1.8.1)
- OpenCV (4.5)
- NumPy (1.21.5)

### Install Requirements

Run the following command to install all required packages:

```bash
pip install -r requirements.txt

python fl_main.py --config "./config/test_config.yaml"


