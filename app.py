import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.data import Data
import os

# ✅ MUST BE FIRST
st.set_page_config(page_title="Drug–Drug Interaction Predictor", layout="wide")

# ----------------- CONFIG -----------------
MODEL_PATH = "gnn_ddi_model.pt"
EMBEDDING_PATH = "drug_embeddings.pt"
CSV_PATH = "drugbank_extracted.csv"
MATCHED_NAMES_PATH = "matched_names.csv"
IN_CHANNELS = 5

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
@st.cache_resource
def load_data_and_model():
    df = pd.read_csv(CSV_PATH)
    df = df[df['name'].notna()]
    df = df[df['name'].str.strip() != '']
    df.reset_index(drop=True, inplace=True)

    # ✅ Inject better names if available
    try:
        match_df = pd.read_csv(MATCHED_NAMES_PATH)
        match_dict = dict(zip(match_df["drugbank_id"], match_df["matched_name"]))
        df["name"] = df["drugbank_id"].map(match_dict).fillna(df["name"])
    except Exception as e:
        st.warning(f"⚠️ Clean name mapping skipped: {e}")

    name_to_id = {row['name']: row['drugbank_id'] for _, row in df.iterrows()}
    id_to_index = {row['drugbank_id']: i for i, row in df.iterrows()}
    index_to_id = {i: row['drugbank_id'] for i, row in df.iterrows()}
    id_to_name = {row['drugbank_id']: row['name'] for _, row in df.iterrows()}
    drug_names = sorted(df['name'].unique())

    encoder = GCNEncoder(in_channels=IN_CHANNELS, hidden_channels=128)
    model = GAE(encoder)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

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
            return None, None, None
        score = torch.sigmoid((z[idx1] * z[idx2]).sum()).item()
        return score, idx1, idx2
    except Exception:
        return None, None, None

def get_risk_label(score):
    if score >= 0.75:
        return "🔴 High Risk", "red"
    elif score >= 0.4:
        return "🟡 Moderate Risk", "orange"
    else:
        return "🟢 Low Risk", "green"

def show_risk_gauge(score):
    st.subheader("📉 Interaction Risk Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        title={'text': "Interaction Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.8,
                'value': score * 100
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

def show_similarity_chart(idx1, idx2, score):
    st.subheader("🧬 Embedding-Based Similarity")
    fig = go.Figure(data=[
        go.Bar(name="Self Similarity", x=[drug1, drug2], y=[1.0, 1.0], marker_color="green"),
        go.Bar(name="Mutual Similarity", x=[f"{drug1} vs {drug2}"], y=[score], marker_color="orange")
    ])
    fig.update_layout(barmode='group', yaxis=dict(range=[0, 1.1]), height=400)
    st.plotly_chart(fig, use_container_width=True)

# ----------------- UI ------------------
st.title("💊 Drug–Drug Interaction Prediction (GNN)")
st.markdown("Select two real-world drugs and predict their interaction using a GNN model.")

col1, col2 = st.columns(2)
with col1:
    drug1 = st.selectbox("Select Drug 1", drug_names)
with col2:
    drug2 = st.selectbox("Select Drug 2", drug_names)

if st.button("🔍 Predict Interaction"):
    score, idx1, idx2 = predict_interaction(drug1, drug2)
    if score is None:
        st.error("❌ Could not compute score for these drugs.")
    else:
        label, color = get_risk_label(score)
        st.markdown(f"### Interaction Score: <span style='color:limegreen'><b>{score:.4f}</b></span>", unsafe_allow_html=True)
        st.markdown(f"### Risk Level: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
        st.markdown("---")
        show_risk_gauge(score)
        show_similarity_chart(idx1, idx2, score)
