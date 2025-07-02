# 💊 Drug–Drug Interaction Prediction using GNN

A cutting-edge machine learning project that predicts drug–drug interactions (DDIs) using **graph neural networks (GNNs)**. Trained on real-world biomedical data (DrugBank), this app helps assess potential risks when combining medications — especially useful for doctors, pharmacists, or researchers.

> ⚠️ Built with 20+ days of iteration, debugging, and refining. Special attention given to real-world drug names and clean UX.

---

## 🚀 Live Demo
🌐 [Streamlit App](https://gnn-ddi.streamlit.app) – Interact with the model, select two drugs, and get an instant risk score with visual gauges.

---

## 📌 Key Features

- ✅ **GNN-based DDI prediction** – Embedding-based link prediction using `torch-geometric`
- 🧠 **Trained on DrugBank** – Built from ~17k real drugs with SMILES + interaction pairs
- 📉 **Risk meter** – Dynamic Plotly gauge chart for risk visualization
- 🔍 **Name-cleaned UI** – Human-readable drug names (cleaned using RxNorm & manual mappings)
- 🧬 **Embedding similarity bar chart** – Understand how similar the drugs are internally
- 🌐 **Streamlit Cloud Ready** – Lightweight deployment, fast inference

---

## 🛠 Tech Stack

| Component        | Tech / Library        |
|------------------|------------------------|
| ML Model         | Graph AutoEncoder (GAE) |
| GNN Framework    | PyTorch Geometric       |
| UI               | Streamlit + Plotly      |
| Dataset          | DrugBank (SMILES, Interactions) |
| Name Cleaning    | Manual + RxNorm Mapping |
| Deployment       | Streamlit Cloud         |

---

## 🧪 How It Works

1. Drugs and their interactions are converted into a graph:  
   - Nodes = drugs  
   - Edges = interactions  
   - Node features = SMILES-derived embeddings

2. A GAE model is trained to predict links between drug pairs using their embeddings.

3. Users select two drugs → model calculates the interaction probability.

---

## 📂 File Structure

├── app.py # Streamlit frontend
├── gnn_ddi_model.pt # Trained GAE model
├── drug_embeddings.pt # Node embeddings (from encoder)
├── drugbank_extracted_cleaned.csv # Final cleaned drug metadata
├── requirements.txt # Dependencies for Streamlit Cloud
└── README.md # Project description

---

## 🙋‍♂️ Future Work

- Add molecule rendering using RDKit (local only, not on Streamlit Cloud)
- Enhance with side effect types or severity classification
- Integrate PubChem or RxNorm APIs for live name matching

---

## 🙌 Acknowledgements

- DrugBank (data)
- PyTorch Geometric
- Streamlit
- Plotly

---

## 🧠 Author

**Aman Tripathi**  
Aspiring Healthcare AI Researcher & Data Consultant  
Let’s connect on [LinkedIn](https://www.linkedin.com/in/amantripathi27)

---

## 📜 License

MIT License
