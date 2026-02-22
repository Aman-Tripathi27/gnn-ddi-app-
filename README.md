# 💊 Drug–Drug Interaction Predictor using GNN

A machine learning system that predicts drug–drug interactions (DDIs) using **Graph Neural Networks (GNNs)**. Trained on real-world biomedical data from DrugBank, this app helps assess interaction probability between any two drugs — useful for doctors, pharmacists, and researchers.

> ⚠️ Built with 20+ days of iteration, debugging, and refining. Special attention given to real-world drug names, clean UX, and medically honest output.

---

## 🚀 Live Demo

🌐 [streamlit](https://gnndrugdiscovery.streamlit.app) — Select two drugs, get instant interaction probability with visual gauges.

---

## 📌 Key Features

- ✅ **GNN-based DDI prediction** — Embedding-based link prediction using `SAGEConv` (GraphSAGE)
- 🧠 **Trained on DrugBank** — Built from 3,709 real drugs with 2.3M+ interaction pairs
- 📉 **Interaction probability gauge** — Dynamic Plotly gauge chart (0–100%)
- 🔍 **Human-readable drug names** — Cleaned using RxNorm & manual mappings
- 🧬 **Embedding similarity chart** — Visualizes how molecularly similar the drugs are
- 🔬 **Novel interaction detection** — Flags drug pairs predicted by GNN but not in DrugBank
- 🐳 **Docker deployed** — Runs on Hugging Face Spaces with CPU Basic (free tier)

---

## 🛠 Tech Stack

| Component | Tech / Library |
|-----------|----------------|
| ML Model | BetterGCN (SAGEConv × 3 layers) |
| GNN Framework | PyTorch Geometric |
| Node Features | 136D (128D Morgan FP + 8 Molecular Descriptors) |
| UI | Streamlit + Plotly |
| Dataset | DrugBank (SMILES + Interactions) |
| Name Cleaning | Manual + RxNorm Mapping |
| Deployment | Hugging Face Spaces (Docker) |

---

## 🧪 How It Works

1. **Graph Construction**
   - Nodes = drugs (3,709 total)
   - Edges = known interactions (2.3M+ pairs)
   - Node features = 136D molecular fingerprints from SMILES

2. **Model Architecture — BetterGCN**
