import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import gdown
import os

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Drug–Drug Interaction Predictor", layout="wide")

CSV_FILE_ID = "191koSx0r0cBSxgU8U8kNcmYGA6rcrl1E"
CSV_FILENAME = "drugbank_cleaned.csv"
MODEL_FILE = "gnn_ddi_model.pt"
EMBEDDING_FILE = "drug_embeddings.pt"

# ------------------ MODEL CLASS ------------------
class GNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# ------------------ DOWNLOAD DATA ------------------
@st.cache_data
def download_csv():
    if not os.path.exists(CSV_FILENAME):
        gdown.download(f"https://drive.google.com/uc?id={CSV_FILE_ID}", CSV_FILENAME, quiet=False)
    return pd.read_csv(CSV_FILENAME)

# ------------------ LOAD MODEL AND DATA ------------------
@st.cache_resource
def load_all():
    df = download_csv()

    # Rebuild model and load weights
    encoder = GNNEncoder(in_channels=50, hidden_channels=64)  # adjust if needed
    model = GAE(encoder)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device("cpu")))
    model.eval()

    embeddings = torch.load(EMBEDDING_FILE, map_location=torch.device("cpu"))

    drug_id_to_index = {drug_id: i for i, drug_id in enumerate(df['drugbank_id'])}
    id_to_name = dict(zip(df['drugbank_id'], df['name']))

    valid_ids = list(drug_id_to_index.keys())[:embeddings.shape[0]]
    drug_list = [id_to_name[drug_id] for drug_id in valid_ids]
    name_to_id = {name: drug_id for drug_id, name in id_to_name.items() if drug_id in valid_ids}

    return model, embeddings, drug_list, name_to_id, drug_id_to_index

model, embeddings, drug_list, name_to_id, drug_id_to_index = load_all()

# ------------------ PREDICT FUNCTION ------------------
def predict_interaction(drug1, drug2):
    try:
        id1 = name_to_id[drug1]
        id2 = name_to_id[drug2]
        idx1 = drug_id_to_index[id1]
        idx2 = drug_id_to_index[id2]
        emb1 = embeddings[idx1]
        emb2 = embeddings[idx2]
        score = torch.sigmoid((emb1 * emb2).sum()).item()
        return round(score, 4)
    except Exception as e:
        return f"Error: {str(e)}"

# ------------------ STREAMLIT UI ------------------
st.title("💊 Drug–Drug Interaction Prediction (GNN)")
st.markdown("Select two real-world drugs and predict their interaction score using a trained GNN model.")

col1, col2 = st.columns(2)
with col1:
    drug1 = st.selectbox("Select Drug 1", drug_list)
with col2:
    drug2 = st.selectbox("Select Drug 2", drug_list)

if st.button("Predict Interaction"):
    score = predict_interaction(drug1, drug2)
    if isinstance(score, float):
        st.success(f"✅ Interaction score between **{drug1}** and **{drug2}**: `{score}`")
    else:
        st.error(f"❌ Could not compute score for these drugs.\n\n{score}")
