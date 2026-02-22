import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import plotly.graph_objects as go
from torch_geometric.nn import SAGEConv

st.set_page_config(
    page_title="Drug–Drug Interaction Predictor",
    page_icon="💊",
    layout="wide"
)

# ----------------- CONFIG -----------------
MODEL_PATH      = "best_sage_model.pt"
FEATURES_PATH   = "drug_features.npy"
DRUG2IDX_PATH   = "drug_to_index.pkl"
IDX2DRUG_PATH   = "index_to_drug.pkl"
CSV_CLEAN_PATH  = "drugbank_extracted_cleaned.csv"
CSV_RAW_PATH    = "drugbank_extracted.csv"   # ← has descriptions!

IN_CHANNELS     = 136
HIDDEN_CHANNELS = 256
OUT_CHANNELS    = 128
DROPOUT         = 0.4

# ----------------- MODEL -----------------
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

    def forward(self, x, edge_index):
        return self.encode(x, edge_index)


# ----------------- LOAD EVERYTHING -----------------
@st.cache_resource
def load_all():
    # Load mappings
    with open(DRUG2IDX_PATH, 'rb') as f:
        drug_to_index = pickle.load(f)
    with open(IDX2DRUG_PATH, 'rb') as f:
        index_to_drug = pickle.load(f)

    # Load cleaned CSV for drug name → id mapping
    df_clean = pd.read_csv(CSV_CLEAN_PATH)
    df_clean = df_clean[df_clean['clean_name'].notna()].copy()
    df_clean = df_clean[df_clean['clean_name'].str.strip() != ''].copy()
    df_clean = df_clean[df_clean['drugbank_id'].isin(drug_to_index.keys())].copy()
    df_clean.reset_index(drop=True, inplace=True)

    # Build name → drugbank_id
    name_to_id = {}
    for _, row in df_clean.iterrows():
        name = row['clean_name'].strip()
        did  = row['drugbank_id']
        if name not in name_to_id and did in drug_to_index:
            name_to_id[name] = did

    drug_names = sorted(name_to_id.keys())

    # Load raw CSV for descriptions
    # Build lookup: (drugbank_id, interacting_drug) → description
    df_raw = pd.read_csv(CSV_RAW_PATH)
    desc_lookup = {}
    for _, row in df_raw.iterrows():
        id1  = row['drugbank_id']
        id2  = row['interacting_drug']
        desc = row['description'] if pd.notna(row.get('description')) else None
        if desc:
            desc_lookup[(id1, id2)] = desc
            desc_lookup[(id2, id1)] = desc  # bidirectional!

    # Load features
    drug_features = np.load(FEATURES_PATH)
    x = torch.FloatTensor(drug_features)

    # Build edge_index
    src_list, dst_list = [], []
    for _, row in df_clean.iterrows():
        id1 = row['drugbank_id']
        id2 = row.get('interacting_drug', None)
        if pd.notna(id2) and id1 in drug_to_index and id2 in drug_to_index:
            src_list.append(drug_to_index[id1])
            dst_list.append(drug_to_index[id2])

    if len(src_list) == 0:
        n = len(drug_to_index)
        edge_index = torch.zeros((2, n), dtype=torch.long)
        for i in range(n):
            edge_index[0, i] = i
            edge_index[1, i] = i
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # Load model & compute embeddings
    model = BetterGCN(IN_CHANNELS, HIDDEN_CHANNELS, OUT_CHANNELS, DROPOUT)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        z = model.encode(x, edge_index)

    return z, name_to_id, drug_to_index, drug_names, desc_lookup


z, name_to_id, drug_to_index, drug_names, desc_lookup = load_all()


# ----------------- NLP CLASSIFIER -----------------
def classify_description(description):
    """
    Reads description text and returns:
    - category label
    - emoji
    - plain english explanation
    """
    if not description:
        return None, None, None

    desc = description.lower()

    # ⚠️ DANGEROUS patterns
    dangerous_keywords = [
        "risk or severity of adverse",
        "toxic activities",
        "increased risk of bleeding",
        "risk of a life-threatening",
        "increased risk of",
        "severe adverse",
        "potentially fatal",
        "respiratory depression",
        "cardiotoxic",
        "nephrotoxic",
        "hepatotoxic",
        "increased serum concentration",
        "risk of hypoglycemia",
        "risk of hyperkalemia",
        "risk of hypokalemia",
        "risk of serotonin",
        "risk of qt",
    ]

    # ✅ BENEFICIAL patterns
    beneficial_keywords = [
        "therapeutic efficacy.*increased",
        "therapeutic efficacy.*enhanced",
        "analgesic activities.*increased",
        "may enhance the.*activit",
        "increased.*therapeutic",
        "enhanced.*therapeutic",
        "efficacy can be increased",
        "efficacy may be increased",
        "may increase the analgesic",
        "may increase the therapeutic",
        "immunosuppressive activities.*increased",
    ]

    # 🔄 REDUCED EFFECTIVENESS patterns
    reduced_keywords = [
        "therapeutic efficacy.*decrease",
        "therapeutic efficacy.*reduce",
        "may decrease the effectiveness",
        "may reduce the effectiveness",
        "may inhibit the",
        "absorption.*decreased",
        "decreased.*absorption",
        "efficacy.*decreased",
        "efficacy.*reduced",
        "may antagonize",
    ]

    # ℹ️ ALTERED LEVELS patterns
    altered_keywords = [
        "serum concentration.*increase",
        "serum concentration.*decrease",
        "plasma concentration",
        "blood levels",
        "serum levels",
        "may increase the.*concentration",
        "may decrease the.*concentration",
        "metabolism.*affected",
        "excretion.*increase",
        "excretion.*decrease",
    ]

    import re

    # Check dangerous first (highest priority)
    for kw in dangerous_keywords:
        if re.search(kw, desc):
            return (
                "DANGEROUS",
                "⚠️",
                "Increased risk of adverse or toxic effects"
            )

    # Check beneficial
    for kw in beneficial_keywords:
        if re.search(kw, desc):
            return (
                "BENEFICIAL",
                "✅",
                "May enhance therapeutic effectiveness"
            )

    # Check reduced effectiveness
    for kw in reduced_keywords:
        if re.search(kw, desc):
            return (
                "REDUCED EFFECTIVENESS",
                "🔄",
                "One drug may reduce the other's effectiveness"
            )

    # Check altered levels
    for kw in altered_keywords:
        if re.search(kw, desc):
            return (
                "ALTERED DRUG LEVELS",
                "ℹ️",
                "Drug concentration in blood may be affected"
            )

    # Default: interaction exists but unclear type
    return (
        "INTERACTION EXISTS",
        "🔬",
        "Pharmacological interaction detected, nature unclear"
    )


# ----------------- DESCRIPTION LOOKUP -----------------
def get_description(id1, id2):
    """Look up description for drug pair, both directions"""
    desc = desc_lookup.get((id1, id2))
    if not desc:
        desc = desc_lookup.get((id2, id1))
    return desc


# ----------------- GNN PREDICT -----------------
def predict(drug1_name, drug2_name):
    try:
        id1  = name_to_id[drug1_name]
        id2  = name_to_id[drug2_name]
        idx1 = drug_to_index[id1]
        idx2 = drug_to_index[id2]

        src = F.normalize(z[idx1], p=2, dim=-1)
        dst = F.normalize(z[idx2], p=2, dim=-1)
        score = torch.sigmoid((src * dst).sum()).item()
        return score, id1, id2
    except Exception as e:
        return None, None, None


def interaction_label(score):
    if score >= 0.75:
        return "🔴 High Interaction Probability"
    elif score >= 0.4:
        return "🟡 Moderate Interaction Probability"
    else:
        return "🟢 Low Interaction Probability"


# ----------------- CHARTS -----------------
def gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 2),
        title={'text': "Interaction Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0,  40], 'color': "lightgreen"},
                {'range': [40, 75], 'color': "orange"},
                {'range': [75,100], 'color': "red"},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.8,
                'value': score * 100
            }
        }
    ))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def similarity_chart(drug1, drug2, score):
    fig = go.Figure(data=[
        go.Bar(name="Self Similarity",
               x=[drug1, drug2], y=[1.0, 1.0],
               marker_color="green"),
        go.Bar(name="Interaction Score",
               x=[f"{drug1} ↔ {drug2}"], y=[round(score, 4)],
               marker_color="orange")
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

if st.button("🔍 Predict Interaction", use_container_width=True):
    if drug1 == drug2:
        st.warning("⚠️ Please select two different drugs!")
    else:
        with st.spinner("Running GNN prediction..."):
            score, id1, id2 = predict(drug1, drug2)

        if score is None:
            st.error("❌ Could not compute score.")
        else:
            # GNN Result
            label = interaction_label(score)
            st.markdown("---")

            r1, r2, r3 = st.columns(3)
            r1.metric("Drug 1", drug1)
            r2.metric("Drug 2", drug2)
            r3.metric("Interaction Score", f"{score:.4f}")

            st.markdown(f"### {label}")
            st.markdown("---")

            # Description Lookup + NLP
            description = get_description(id1, id2)

            if description:
                category, emoji, meaning = classify_description(description)

                # Show NLP result
                st.markdown(f"### {emoji} Interaction Type: **{category}**")
                st.info(f"**Clinical Meaning:** {meaning}")

                with st.expander("📋 View Full DrugBank Description"):
                    st.write(description)

            else:
                st.markdown("### 🔬 Clinical Description")
                st.warning(
                    "No clinical description available in DrugBank for this drug pair.\n\n"
                    "The GNN model has predicted an interaction based on molecular "
                    "structure and known interaction patterns — this may represent "
                    "a **novel or undocumented interaction**."
                )

            # Always show disclaimer
            st.error(
                "⚕️ **Medical Disclaimer:** This tool is for research purposes only. "
                "Always consult a qualified healthcare professional before making "
                "any medical decisions."
            )

            st.markdown("---")

            # Charts
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("📉 Interaction Probability Gauge")
                gauge_chart(score)
            with col_b:
                st.subheader("🧬 Embedding Similarity")
                similarity_chart(drug1, drug2, score)


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model:** BetterGCN (SAGEConv)
    **Dataset:** DrugBank
    **Drugs:** 3,709
    **Features:** 136D per drug
    - 128D Morgan Fingerprints
    - 8 Molecular Descriptors

    **AUC:** ~89%
    **Validation:** 5-Fold CV
    """)

    st.markdown("---")
    st.header("📊 Interaction Types")
    st.markdown("""
    ✅ **Beneficial** — Enhances therapeutic effect

    ⚠️ **Dangerous** — Increases adverse/toxic risk

    🔄 **Reduced Effectiveness** — One drug weakens the other

    ℹ️ **Altered Levels** — Changes drug concentration in blood

    🔬 **Novel** — Predicted by GNN, not in DrugBank yet
    """)

    st.markdown("---")
    st.header("🎯 Score Interpretation")
    st.markdown("""
    🔴 **≥ 0.75** — High interaction probability

    🟡 **0.40–0.74** — Moderate interaction probability

    🟢 **< 0.40** — Low interaction probability
    """)

    st.markdown("---")
    st.caption("Built by Aman Tripathi | GNN-DDI Project | DrugBank Data")
