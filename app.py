import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="Drug–Drug Interaction Predictor", layout="centered")

# --- FILES ---
MODEL_PATH = "gnn_ddi_model.pt"
EMBEDDING_PATH = "drug_embeddings.pt"
DRUG_LIST_PATH = "drug_list.txt"

# --- LOAD DRUG LIST ---
@st.cache_data
def load_drug_list():
    with open(DRUG_LIST_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]

# --- DEFINE MODEL ENCODER (same as training) ---
class GNNEncoder(torch.nn.Module):
    def __init__(self, in_channels=50, hidden_channels=128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

# --- LOAD MODEL + EMBEDDINGS ---
@st.cache_resource
def load_model_and_embeddings():
    encoder = GNNEncoder()
    model = GAE(encoder)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    embeddings = torch.load(EMBEDDING_PATH, map_location="cpu")
    return model, embeddings

# --- INTERACTION PREDICTION ---
def predict_interaction(model, embeddings, index1, index2):
    emb1 = embeddings[index1]
    emb2 = embeddings[index2]
    return torch.sigmoid((emb1 * emb2).sum()).item()

# --- MAIN APP ---
st.title("🔬 Drug–Drug Interaction Predictor (GNN-based)")

drug_list = load_drug_list()
model, embeddings = load_model_and_embeddings()

drug1 = st.selectbox("Choose Drug 1", drug_list, index=0)
drug2 = st.selectbox("Choose Drug 2", drug_list, index=1)

if st.button("🔍 Predict Interaction"):
    try:
        idx1 = drug_list.index(drug1)
        idx2 = drug_list.index(drug2)
        score = predict_interaction(model, embeddings, idx1, idx2)
        st.success(f"✅ Interaction Score: **{score:.4f}**")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
