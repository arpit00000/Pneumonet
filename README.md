# PneumoNet 🫁 | Federated Deep Learning for Pulmonary Disease Detection

A privacy-preserving Federated Learning project that uses Deep Learning to detect lung diseases (e.g., pneumonia) from Chest X-ray images using decentralized data simulation.

## 🧠 Project Overview

This project simulates **federated learning** where each batch of training data represents a separate client. It leverages the power of **ResNet101** and **PyTorch** to train a CNN model on Chest X-ray datasets for disease classification, with a focus on **decentralized and privacy-preserving training**.

## 📌 Features

- ✅ Federated Learning Simulation
- ✅ Client-wise training using batches
- ✅ ResNet101-based CNN architecture
- ✅ Chest X-ray disease classification
- ✅ PyTorch implementation
- ✅ Central aggregator to update global model
- ✅ Performance metrics: Accuracy, Loss tracking

## 🛠️ Tech Stack

- **Python**
- **PyTorch**
- **Matplotlib, NumPy**
- **Sklearn**
- **OpenCV** (optional)
- **Jupyter Notebook / VS Code**

## 🧪 Dataset

- Publicly available **Chest X-ray** datasets:
  - [Kaggle Chest X-ray Pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
  - Others can be simulated in local environment.

> Note: Data is split to simulate **federated clients**.

## 🔄 How Federated Learning Works Here

1. Each batch = a simulated client.
2. Each client trains a local model.
3. A global model is updated via aggregation.
4. The process repeats for multiple communication rounds.

<p align="center">
  <img src="https://github.com/arpit00000/Pneumonet/assets/128667568/9429b028-607e-45f8-8fae-228ff7a2f135" alt="Chat LLB UI" width="600"/>
</p>

## 🚀 Getting Started

### Clone the repo
```bash
git clone https://github.com/arpit00000/Pneumonet.git
cd Pneumonet
```

### Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the training
```bash
python federated_train.py
```

> Make sure your dataset is properly placed and paths configured.

## 📈 Results

- Achieved **xx% accuracy** on Chest X-ray images.
- Training logs and graphs available in `/results`.

## 📷 Demo Screenshots

![Model UI](https://github.com/arpit00000/Pneumonet/assets/128667568/9429b028-607e-45f8-8fae-228ff7a2f135)

## ⚠️ Disclaimer

This model is for **academic and research purposes only**. Medical diagnosis should only be performed by licensed professionals.

## 🤝 Contributors

- **Arpit Jaiswal** — [@arpit00000](https://github.com/arpit00000)

## 📬 Contact

For inquiries:  
📧 arpit.jaiswal2021@vitstudent.ac.in  
📱 +91 6387145766
