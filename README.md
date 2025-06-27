# 💊 Drug–Drug Interaction Prediction System (GNN-based)

This project implements a **Graph Neural Network (GNN)** to predict potential **drug–drug interactions (DDIs)** using real-world data. The model leverages chemical and interaction data from DrugBank to assess and visualize the risk level when two drugs are taken together.

> ✅ Fully functional **Streamlit web app** included  
> ✅ Trained on ~17,000 drugs  
> ✅ Embeds an intuitive risk gauge + drug subgraph  
> ✅ Built using PyTorch Geometric and GCN-GAE

---

## 🔬 Project Motivation

Adverse drug–drug interactions are a major safety concern in clinical treatment. Our goal was to build a personalized, explainable tool to:

- Simulate interaction risk between any two known drugs  
- Help doctors/pharmacists make informed prescription decisions  
- Leverage **graph-based deep learning** for molecular-level insight

---

## 🧠 Model Architecture

- **Input Data**: DrugBank records with `drug_id`, `drug_name`, `SMILES`, and interaction links
- **Graph Structure**: Drugs as nodes, interactions as edges
- **Model**: GCN-based Autoencoder (GAE)
  - Encoder: 2-layer Graph Convolutional Network
  - Embedding space: `z = encoder(x, edge_index)`
  - Similarity score: `sigmoid(dot(z1, z2))`

---

## ⚙️ Tech Stack

| Component | Tool |
|----------|------|
| Web UI | Streamlit |
| Deep Learning | PyTorch + PyTorch Geometric |
| Graphs | NetworkX, Plotly |
| Data Source | DrugBank CSV |
| Deployment | Streamlit Cloud-ready |

---

## 🚀 Features

- 🔍 Predict interaction between any two drugs
- 📉 Risk score (Low / Moderate / High)
- 📊 Gauge visual for clinical interpretability
- 🧠 GNN-trained embeddings (based on real interaction graph)
- 🌐 Fast & intuitive drug search via dropdowns
- 🔗 Trained model + embeddings are preloaded

---

## 📸 App Preview

https://gnn-ddi.streamlit.app/

---

## 🏁 Getting Started


