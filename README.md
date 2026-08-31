# 🧠 Federated Explainable AI for Alzheimer's Disease Prediction

> **Author:** Kasyap M (Roll No. 56) · B.Tech S7/S8 Academic Project

A privacy-preserving, multimodal deep learning framework for Alzheimer's Disease prediction using **Federated Learning**, **Explainable AI**, and a **Clinical Web Dashboard**.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                       │
│              (localhost:8501)                                │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │Dashboard │  │New Prediction│  │Prediction History    │  │
│  │Overview  │  │MRI + Clinical│  │Browse Past Results   │  │
│  └──────────┘  └──────┬───────┘  └──────────────────────┘  │
└─────────────────────────┼───────────────────────────────────┘
                          │ HTTP (REST API)
┌─────────────────────────▼───────────────────────────────────┐
│                    FastAPI Gateway                           │
│              (localhost:8000)                                │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐     │
│  │Model       │  │Grad-CAM      │  │SHAP             │     │
│  │Service     │  │Service       │  │Service           │     │
│  └────────────┘  └──────────────┘  └─────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│               MultimodalAlzheimersModel                     │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │CNN Branch   │  │MLP Branch    │  │Fusion Classifier│    │
│  │(ResNet-18)  │  │(3-layer MLP) │  │(Late Concat)    │    │
│  │MRI → 512-d  │  │Tab → 128-d   │  │640-d → 4 class  │    │
│  └─────────────┘  └──────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                          │ Trained via
┌─────────────────────────▼───────────────────────────────────┐
│              Flower Federated Learning                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │Client 1  │  │Client 2  │  │Client N  │  ← IID Partitions│
│  │Local Data│  │Local Data│  │Local Data│                  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
│       └──────────────┼─────────────┘                        │
│              ┌───────▼───────┐                              │
│              │  FedAvg Server│ → Aggregated Global Model     │
│              └───────────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Federated Learning Simulation
```bash
python federated_core/run_simulation.py --num-clients 3 --num-rounds 5
```

### 3. Start the API Gateway
```bash
uvicorn api_gateway.main:app --host 0.0.0.0 --port 8000
```

### 4. Launch the Dashboard
```bash
streamlit run streamlit_app/app.py
```

## Dataset

Uses the `archive (2)/` dataset with 4 classes of brain MRI scans:

| Class | Original | Augmented |
|---|---|---|
| Non Demented | 3,200 | 9,600 |
| Very Mild Dementia | 2,240 | 8,960 |
| Mild Dementia | 896 | 8,960 |
| Moderate Dementia | 64 | 6,464 |

## Tech Stack

- **Federated Learning:** Flower (`flwr`) with FedAvg strategy
- **Deep Learning:** PyTorch, ResNet-18, Multimodal Fusion
- **Explainable AI:** Grad-CAM (MRI heatmaps) + SHAP (clinical feature importance)
- **API:** FastAPI + Pydantic v2
- **Dashboard:** Streamlit
- **Clinical Features:** 9 synthetic features (MMSE, CDR, Age, EDUC, nWBV, eTIV, ASF, SES, Gender)
