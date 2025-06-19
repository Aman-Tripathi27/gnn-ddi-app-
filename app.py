import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import os

# ✅ MUST BE FIRST
st.set_page_config(page_title="Drug–Drug Interaction Predictor", layout="wide")


# ----------------- CONFIG -----------------
MODEL_PATH = "gnn_ddi_model.pt"
EMBEDDING_PATH = "drug_embeddings.pt"
CSV_PATH = "drugbank_extracted.csv"  # Your original training data
IN_CHANNELS = 5  # Must match the trained model

# ----------------- MODEL ------------------
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels=5, hidden_channels=128):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 64)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# ----------------- LOAD DATA ------------------
@st.cache_data
def load_data_and_model():
    df = pd.read_csv(CSV_PATH)
    df = df[df['name'].notna()]
    df = df[df['name'].str.strip() != '']
    df.reset_index(drop=True, inplace=True)

    name_to_id = {row['name']: row['drugbank_id'] for _, row in df.iterrows()}
    id_to_index = {row['drugbank_id']: i for i, row in df.iterrows()}
    index_to_id = {i: row['drugbank_id'] for i, row in df.iterrows()}
    id_to_name = {row['drugbank_id']: row['name'] for _, row in df.iterrows()}
    drug_names = sorted(df['name'].unique())

    # Load model
    encoder = GCNEncoder(in_channels=IN_CHANNELS, hidden_channels=128)
    model = GAE(encoder)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # Load embeddings
    z = torch.load(EMBEDDING_PATH, map_location="cpu")

    return df, model, z, name_to_id, id_to_index, index_to_id, id_to_name, drug_names

df, model, z, name_to_id, id_to_index, index_to_id, id_to_name, drug_names = load_data_and_model()

# ----------------- UTILS ------------------
def predict_interaction(drug1, drug2):
    try:
        id1 = name_to_id[drug1]
        id2 = name_to_id[drug2]
        idx1 = id_to_index[id1]
        idx2 = id_to_index[id2]

        if idx1 >= len(z) or idx2 >= len(z):
            return None

        score = torch.sigmoid((z[idx1] * z[idx2]).sum()).item()
        return score
    except Exception as e:
        return None

def get_risk_label(score):
    if score >= 0.75:
        return "🔴 High Risk", "red"
    elif score >= 0.4:
        return "🟡 Moderate Risk", "orange"
    else:
        return "🟢 Low Risk", "green"

def build_ddi_graph():
    edge_list = []
    for i, row in df.iterrows():
        src = id_to_index.get(row['drugbank_id'])
        for tgt_id in str(row.get('interactions', '')).split('|'):
            tgt = id_to_index.get(tgt_id)
            if tgt is not None:
                edge_list.append([src, tgt])
    if not edge_list:
        return None
    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    x_dummy = torch.randn(len(z), IN_CHANNELS)
    return Data(x=x_dummy, edge_index=edge_index)

def show_interaction_graph(drug1, drug2):
    G_data = build_ddi_graph()
    if G_data is None:
        st.warning("⚠️ No interaction graph could be built.")
        return
    G = to_networkx(G_data, to_undirected=True)
    id1 = name_to_id[drug1]
    id2 = name_to_id[drug2]
    idx1 = id_to_index[id1]
    idx2 = id_to_index[id2]
    nodes = {idx1, idx2}
    nodes.update(G.neighbors(idx1))
    nodes.update(G.neighbors(idx2))
    subG = G.subgraph(nodes)
    color_map = ["red" if n == idx1 else "green" if n == idx2 else "gray" for n in subG.nodes]
    labels = {n: id_to_name.get(index_to_id[n], f"Drug {n}") for n in subG.nodes}
    pos = nx.spring_layout(subG)
    fig, ax = plt.subplots(figsize=(8, 8))
    nx.draw(subG, pos, labels=labels, node_color=color_map, node_size=700, font_size=9, ax=ax)
    st.pyplot(fig)

# ----------------- UI ------------------
st.set_page_config(page_title="Drug–Drug Interaction Predictor", layout="wide")
st.title("💊 Drug–Drug Interaction Prediction (GNN)")
st.markdown("Select two real-world drugs and predict their interaction using a GNN model.")

col1, col2 = st.columns(2)
with col1:
    drug1 = st.selectbox("Select Drug 1", drug_names)
with col2:
    drug2 = st.selectbox("Select Drug 2", drug_names)

if st.button("🔍 Predict Interaction"):
    score = predict_interaction(drug1, drug2)
    if score is None:
        st.error("❌ Could not compute score for these drugs.")
    else:
        label, color = get_risk_label(score)
        st.markdown(f"### Interaction Score: <span style='color:limegreen'><b>{score:.4f}</b></span>", unsafe_allow_html=True)
        st.markdown(f"### Risk Level: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("🧠 Drug Interaction Subgraph")
        show_interaction_graph(drug1, drug2)
