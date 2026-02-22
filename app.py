import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import plotly.graph_objects as go
from torch_geometric.nn import SAGEConv

# ✅ MUST BE FIRST
st.set_page_config(
    page_title="Drug–Drug Interaction Predictor",
    page_icon="💊",
    layout="wide"
)

# ----------------- CONFIG -----------------
MODEL_PATH       = "best_sage_model.pt"
FEATURES_PATH    = "drug_features.npy"
DRUG2IDX_PATH    = "drug_to_index.pkl"
IDX2DRUG_PATH    = "index_to_drug.pkl"
CSV_PATH         = "drugbank_extracted_cleaned.csv"

IN_CHANNELS      = 136
HIDDEN_CHANNELS  = 256
OUT_CHANNELS     = 128
DROPOUT          = 0.4

# ----------------- MODEL DEFINITION -----------------
class BetterGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.4):
        super(BetterGCN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.bn1   = nn.BatchNorm1d(hidden_channels)
        self.bn2   = nn.BatchNorm1d(hidden_channels)
        self.dropout = dropout

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        return x

    def decode(self, z, idx1, idx2):
        src = F.normalize(z[idx1], p=2, dim=-1)
        dst = F.normalize(z[idx2], p=2, dim=-1)
        return (src * dst).sum(dim=-1)

    def forward(self, x, edge_index):
        return self.encode(x, edge_index)


# ----------------- LOAD EVERYTHING -----------------
@st.cache_resource
def load_all():
    # Load CSV for drug names
    df = pd.read_csv(CSV_PATH)
    df = df[df['clean_name'].notna()]
    df = df[df['clean_name'].str.strip() != '']
    df.reset_index(drop=True, inplace=True)

    # Load mappings
    with open(DRUG2IDX_PATH, 'rb') as f:
        drug_to_index = pickle.load(f)
    with open(IDX2DRUG_PATH, 'rb') as f:
        index_to_drug = pickle.load(f)

    # Map clean_name → drugbank_id → index
    id_to_name  = {row['drugbank_id']: row['clean_name'] for _, row in df.iterrows()}
    name_to_id  = {row['clean_name']: row['drugbank_id'] for _, row in df.iterrows()}
    drug_names  = sorted(df['clean_name'].dropna().unique().tolist())

    # Load drug features
    drug_features = np.load(FEATURES_PATH)
    x = torch.FloatTensor(drug_features)

    # Build edge index from CSV (all known interactions)
    src_list, dst_list = [], []
    for _, row in df.iterrows():
        id1 = row['drugbank_id']
        if id1 not in drug_to_index:
            continue
        if pd.notna(row.get('interacting_drug', None)):
            id2 = row['interacting_drug']
            if id2 in drug_to_index:
                src_list.append(drug_to_index[id1])
                dst_list.append(drug_to_index[id2])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Load model
    model = BetterGCN(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, DROPOUT)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    # Get embeddings
    with torch.no_grad():
        z = model.encode(x, edge_index)

    return z, name_to_id, drug_to_index, id_to_name, drug_names


z, name_to_id, drug_to_index, id_to_name, drug_names = load_all()


# ----------------- PREDICTION -----------------
def predict(drug1_name, drug2_name):
    try:
        id1  = name_to_id[drug1_name]
        id2  = name_to_id[drug2_name]
        idx1 = drug_to_index[id1]
        idx2 = drug_to_index[id2]

        with torch.no_grad():
            score = torch.sigmoid(
                (F.normalize(z[idx1], p=2, dim=-1) *
                 F.normalize(z[idx2], p=2, dim=-1)).sum()
            ).item()

        return score, idx1, idx2
    except Exception as e:
        return None, None, None


def risk_label(score):
    if score >= 0.75:
        return "🔴 High Risk", "red"
    elif score >= 0.4:
        return "🟡 Moderate Risk", "orange"
    else:
        return "🟢 Low Risk", "green"


# ----------------- CHARTS -----------------
def gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = round(score * 100, 2),
        title = {'text': "Interaction Risk (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar':  {'color': "darkblue"},
            'steps': [
                {'range': [0,  40], 'color': "lightgreen"},
                {'range': [40, 75], 'color': "orange"},
                {'range': [75,100], 'color': "red"},
            ],
            'threshold': {
                'line':      {'color': "black", 'width': 4},
                'thickness': 0.8,
                'value':     score * 100
            }
        }
    ))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def similarity_chart(drug1, drug2, score):
    fig = go.Figure(data=[
        go.Bar(
            name="Self Similarity",
            x=[drug1, drug2],
            y=[1.0, 1.0],
            marker_color="green"
        ),
        go.Bar(
            name="Mutual Interaction Score",
            x=[f"{drug1} ↔ {drug2}"],
            y=[round(score, 4)],
            marker_color="orange"
        )
    ])
    fig.update_layout(
        barmode='group',
        yaxis=dict(range=[0, 1.1]),
        height=400,
        title="Embedding-Based Similarity"
    )
    st.plotly_chart(fig, use_container_width=True)


# ----------------- UI -----------------
st.title("💊 Drug–Drug Interaction Predictor")
st.markdown("""
Predict potential interactions between two drugs using a 
**Graph Neural Network (SAGEConv)** trained on DrugBank.  
> 🧠 Model: BetterGCN | Features: 136D (Morgan FP + Descriptors) | AUC: ~89%
""")

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    drug1 = st.selectbox("💊 Select Drug 1", drug_names, index=0)
with col2:
    drug2 = st.selectbox("💊 Select Drug 2", drug_names, index=1)

st.markdown("")

if st.button("🔍 Predict Interaction", use_container_width=True):
    if drug1 == drug2:
        st.warning("⚠️ Please select two different drugs!")
    else:
        with st.spinner("Running GNN prediction..."):
            score, idx1, idx2 = predict(drug1, drug2)

        if score is None:
            st.error("❌ Could not compute score. Drug mapping missing.")
        else:
            label, color = risk_label(score)

            st.markdown("---")
            r1, r2, r3 = st.columns(3)
            r1.metric("Drug 1", drug1)
            r2.metric("Drug 2", drug2)
            r3.metric("Interaction Score", f"{score:.4f}")

            st.markdown(f"## Risk Level: {label}")
            st.markdown("---")

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("📉 Risk Gauge")
                gauge_chart(score)
            with col_b:
                st.subheader("🧬 Similarity Chart")
                similarity_chart(drug1, drug2, score)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model:** BetterGCN (SAGEConv)  
    **Dataset:** DrugBank  
    **Drugs:** 3,709  
    **Interactions:** 2.3M+  
    **Features:** 136D per drug  
    - 128D Morgan Fingerprints  
    - 8 Molecular Descriptors  
    **AUC:** ~89%  
    **Validation:** 5-Fold CV  
    """)

    st.markdown("---")
    st.header("🎨 Risk Levels")
    st.markdown("""
    🔴 **High Risk:** Score ≥ 0.75  
    🟡 **Moderate Risk:** 0.40 – 0.74  
    🟢 **Low Risk:** Score < 0.40  
    """)

    st.markdown("---")
    st.caption("Built by Aman Tripathi | GNN-DDI Project")
