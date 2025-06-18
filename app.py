import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

st.set_page_config(page_title="Drug–Drug Interaction Predictor", layout="wide")
st.title("💊 Drug–Drug Interaction Predictor")
st.write("🚀 App is starting...")

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("drugbank_cleaned.csv.zip")
    df = df[df['display_name'].notnull() & (df['display_name'].str.strip() != '')]
    return df

drug_df = load_data()

@st.cache_resource
def load_model_and_embeddings():
    model_state = torch.load("gnn_ddi_model.pt", map_location="cpu")
    embeddings = torch.load("drug_embeddings.pt", map_location="cpu")
    return model_state, embeddings

try:
    model_state, z = load_model_and_embeddings()
except Exception as e:
    st.error(f"❌ Model load failed: {e}")
    st.stop()

# ---------------- GCN Model ----------------
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

in_channels = 5
out_channels = 64
model = GAE(GCNEncoder(in_channels, out_channels))
model.load_state_dict(model_state)
model.eval()

# ---------------- Mapping ----------------
drug_id_to_index = {row['drugbank_id']: i for i, row in drug_df.iterrows()}
index_to_id = {i: row['drugbank_id'] for i, row in drug_df.iterrows()}
id_to_name = {row['drugbank_id']: row['display_name'] for _, row in drug_df.iterrows()}
drug_names = sorted(drug_df['display_name'].unique().tolist())

# ---------------- Prediction Logic ----------------
def predict_interaction(drug1, drug2):
    try:
        id1 = drug_df[drug_df['display_name'] == drug1]['drugbank_id'].values[0]
        id2 = drug_df[drug_df['display_name'] == drug2]['drugbank_id'].values[0]
        idx1 = drug_id_to_index.get(id1)
        idx2 = drug_id_to_index.get(id2)
        if idx1 is None or idx2 is None or idx1 >= len(z) or idx2 >= len(z):
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

def show_molecule_placeholder(drug_name):
    st.markdown(f"🧬 Molecule for **{drug_name}** not available (RDKit excluded).")

def build_ddi_graph():
    edge_list = []
    for i, row in drug_df.iterrows():
        src = drug_id_to_index.get(row['drugbank_id'])
        for tgt_id in str(row.get('interactions', '')).split('|'):
            tgt = drug_id_to_index.get(tgt_id)
            if tgt is not None:
                edge_list.append([src, tgt])
    if not edge_list:
        return None
    edge_index = torch.tensor(edge_list, dtype=torch.long).T
    x_dummy = torch.randn(len(z), in_channels)
    return Data(x=x_dummy, edge_index=edge_index)

def show_interaction_graph(drug1, drug2):
    try:
        G_data = build_ddi_graph()
        if G_data is None:
            st.warning("⚠️ No interaction graph could be built.")
            return
        G = to_networkx(G_data, to_undirected=True)
        id1 = drug_df[drug_df['display_name'] == drug1]['drugbank_id'].values[0]
        id2 = drug_df[drug_df['display_name'] == drug2]['drugbank_id'].values[0]
        idx1 = drug_id_to_index.get(id1)
        idx2 = drug_id_to_index.get(id2)
        nodes = {idx1, idx2}
        nodes.update(G.neighbors(idx1))
        nodes.update(G.neighbors(idx2))
        subG = G.subgraph(nodes)
        color_map = ["red" if n == idx1 else "green" if n == idx2 else "gray" for n in subG.nodes]
        labels = {n: id_to_name.get(index_to_id[n], f"Drug {n}") for n in subG.nodes}
        pos = nx.spring_layout(subG)
        fig, ax = plt.subplots(figsize=(8, 8))
        nx.draw(subG, pos, labels=labels, node_color=color_map, node_size=800, font_size=9, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.warning("❗Could not render graph.")

# ---------------- UI ----------------
st.markdown("Select two drugs below to predict their interaction.")

col1, col2 = st.columns(2)
with col1:
    drug1 = st.selectbox("Select Drug 1", drug_names)
    show_molecule_placeholder(drug1)

with col2:
    drug2 = st.selectbox("Select Drug 2", drug_names)
    show_molecule_placeholder(drug2)

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
