# 🔒 Privacy-Preserving Federated Learning for Phishing Website Detection

## 📌 Overview

This project proposes a Privacy-Preserving and Communication-Efficient Federated Learning framework for phishing website detection. Unlike traditional centralized machine learning approaches, the proposed system allows multiple clients to collaboratively train a phishing detection model without sharing raw data, thereby preserving user privacy.

The framework integrates:

- Federated Learning using Flower
- Non-IID Client Simulation
- Differential Privacy (DP)
- Selective Client Updates
- Communication-Efficient Training

The proposed approach demonstrates that privacy-preserving phishing detection can be achieved with competitive accuracy while significantly reducing communication overhead.

---

## 🎯 Problem Statement

Traditional phishing detection systems require centralized collection of user data for model training, raising significant privacy concerns and increasing the risk of data leakage.

This project aims to develop a privacy-preserving phishing detection framework using Federated Learning, Differential Privacy, and Selective Updates to enable collaborative model training while maintaining data confidentiality and reducing communication costs.

---

## 🚀 Key Features

✅ Federated Learning with Flower

✅ Differential Privacy for client updates

✅ Non-IID data distribution simulation

✅ Selective update mechanism

✅ Communication reduction analysis

✅ Streamlit Dashboard

✅ Real-time phishing prediction

---

## 🏗️ System Architecture

```text
Dataset
   │
   ▼
Preprocessing
   │
   ▼
Train/Test Split
   │
   ▼
Client Distribution
(IID / Non-IID)
   │
   ▼
Local Model Training
(Logistic Regression)
   │
   ▼
Differential Privacy
(Add Noise)
   │
   ▼
Selective Updates
(Send Only Useful Updates)
   │
   ▼
Flower Federated Server
(FedAvg Aggregation)
   │
   ▼
Global Model
   │
   ▼
Evaluation & Dashboard
```

---

## 📂 Project Structure

```text
Privacy_preserving_FL_Phishing/

├── dataset/
│   ├── processed/
│   ├── clients/
│   └── non_iid_clients/
│
├── federated/
│   ├── client.py
│   └── server.py
│
├── models/
│   ├── baseline_model.pkl
│   └── scaler.pkl
│
├── results/
│   ├── accuracy_results.csv
│   ├── communication_results.csv
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── streamlit_app/
│   ├── app.py
│   └── pages/
│
├── visualizations/
│
└── requirements.txt
```

---

## 📊 Dataset

Dataset: UCI Phishing Websites Dataset

Features: 30

Target Classes:

- 0 → Legitimate Website
- 1 → Phishing Website

Total Samples: 11,055

---

## ⚙️ Technologies Used

- Python
- Scikit-Learn
- Flower Federated Learning
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

## 📈 Experimental Results

### Baseline Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 91.54% |
| Precision | 87.80% |
| Recall    | 93.98% |
| F1 Score  | 90.78% |
| ROC-AUC   | 97.79% |

### Federated Learning Performance

| Method                     | Accuracy |
| -------------------------- | -------- |
| IID Federated Learning     | 92.88%   |
| Non-IID Federated Learning | 86.13%   |
| Differential Privacy FL    | 86.14%   |
| Proposed Method            | 86.07%   |

### Communication Analysis

| Metric                  | Value |
| ----------------------- | ----- |
| Standard FL Updates     | 20    |
| Proposed Updates        | 7     |
| Communication Reduction | 65%   |

---

## 🖥️ Running the Project

### Clone Repository

```bash
git clone https://github.com/Shreyascr5/privacy-preserving-federated-phishing-detection.git

cd privacy-preserving-federated-phishing-detection
```

### Create Virtual Environment

```bash
python3 -m venv venv

source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start Federated Server

```bash
python federated/server.py
```

### Start Clients

```bash
python federated/client.py 1
python federated/client.py 2
python federated/client.py 3
python federated/client.py 4
```

### Launch Dashboard

```bash
streamlit run streamlit_app/app.py
```

---

## 📚 Future Work

- Deep Learning-Based Federated Models
- Secure Aggregation
- Homomorphic Encryption
- Federated Transformer Architectures
- Real-Time URL Analysis API
- Cross-Device Federated Learning

---

## 👨‍💻 Author

Shreyas C R

M.Tech Data Science

Ramaiah Institute of Technology (MSRIT)

Academic Project – Privacy-Preserving Federated Learning for Phishing Website Detection
