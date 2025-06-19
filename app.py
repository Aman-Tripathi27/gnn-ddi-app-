import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import os
import gdown

# --- SET CONFIG ---
st.set_page_config(page_title="Drug–Drug Interaction Predictor", layout="wide")

# --- CONSTANTS ---
CSV_FILE_ID = "191koSx0r0cBSxgU8U8kNcmYGA6rcrl1E"
CSV_FILENAME = "drugbank_cleaned.csv"
EMBEDDING_FILE = "drug_embeddings.pt"
MODEL_FILE = "gnn_ddi_model.pt"

# --- DOWNLOAD CSV IF NEEDED ---
@st.cache_data
def download_csv():
    if not os.path.exists(CSV_FILENAME):
        gdown.download(f"https://drive.google.com/uc?id={CSV_FILE_ID}", CSV_FILENAME, quiet=False)
    return pd.read_csv(CSV_FILENAME)

# --- LOAD EVERYTHING ---
@st.cache_resource
def load_all():
    df = download_csv()

    # Load embeddings and model
    model = torch.load(MODEL_FILE, map_location=torch.device("cpu"))
    model.eval()
    embeddings = torch.load(EMBEDDING_FILE, map_location=torch.device("cpu"))

    # Create mappings
    drug_id_to_index = {drug_id: i for i, drug_id in enumerate(df['drugbank_id'])}
    id_to_name = dict(zip(df['drugbank_id'], df['name']))

    # Filter only drugs in embedding
    valid_ids = list(drug_id_to_index.keys())[:embeddings.shape[0]]
    drug_list = [id_to_name[drug_id] for drug_id in valid_ids]
    name_to_id = {name: drug_id for drug_id, name in id_to_name.items() if drug_id in valid_ids}

    return model, embeddings, drug_list, name_to_id, drug_id_to_index

model, embeddings, drug_list, name_to_id, drug_id_to_index = load_all()

# --- PREDICT FUNCTION ---
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
        return f"Error: {e}"

# --- UI ---
st.title("💊 Drug–Drug Interaction Prediction (GNN)")
st.markdown("Select two real-world drugs from the training set to predict their interaction using a GNN model.")

col1, col2 = st.columns(2)

with col1:
    drug1 = st.selectbox("Select Drug 1", drug_list)

with col2:
    drug2 = st.selectbox("Select Drug 2", drug_list, index=1)

if st.button("🔍 Predict Interaction"):
    if drug1 == drug2:
        st.warning("Please select two different drugs.")
    else:
        with st.spinner("Calculating interaction..."):
            result = predict_interaction(drug1, drug2)
            if isinstance(result, str) and result.startswith("Error"):
                st.error("❌ Could not compute score for these drugs.")
                st.caption(result)
            else:
                st.success(f"✅ Interaction Score: {result}")
