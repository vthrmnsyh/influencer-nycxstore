import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from pathlib import Path

# ── Optimasi Konfigurasi Windows & Streamlit ────────────────────────────────
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Robust path — works on Streamlit Cloud, local, dan Windows
DEPLOY_DIR = str(Path(__file__).resolve().parent / "deployment")

st.set_page_config(
    page_title="Influencer @NYCXSTORE",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────
# KONSTANTA & PATH
# ────────────────────────────────────────────────────────────────────────────
TIER_COLORS = {
    "Tier 1 - Top Influencer":   "#e74c3c",
    "Tier 2 - Micro Influencer": "#f39c12",
    "Tier 3 - Regular User":     "#3498db",
}
TIER_ORDER = list(TIER_COLORS.keys())

# ────────────────────────────────────────────────────────────────────────────
# DEFENSIVE IMPORT LIBRARIES (Menghindari Crash Native)
# ────────────────────────────────────────────────────────────────────────────
ML_AVAILABLE = False
ML_ERROR_MSG = ""
GNNInfluencer = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv, GATConv

    class GNNInfluencerModel(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim=2, heads=4, dropout=0.3):
            super().__init__()
            self.input_norm = nn.LayerNorm(in_dim)
            self.sage1 = SAGEConv(in_dim, hidden_dim)
            self.bn1   = nn.BatchNorm1d(hidden_dim)
            self.gat   = GATConv(hidden_dim, hidden_dim // heads,
                                  heads=heads, dropout=dropout, concat=True)
            self.bn2   = nn.BatchNorm1d(hidden_dim)
            self.sage2 = SAGEConv(hidden_dim, hidden_dim // 2)
            self.bn3   = nn.BatchNorm1d(hidden_dim // 2)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim // 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, out_dim),
            )
            self.dropout = dropout

        def forward(self, x, edge_index):
            x = self.input_norm(x)
            x = F.relu(self.bn1(self.sage1(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.bn2(self.gat(x, edge_index)))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.bn3(self.sage2(x, edge_index)))
            return self.classifier(x)
    
    GNNInfluencer = GNNInfluencerModel
    ML_AVAILABLE = True
except Exception as e:
    ML_AVAILABLE = False
    ML_ERROR_MSG = str(e)


# ────────────────────────────────────────────────────────────────────────────
# CACHE: LOAD ARTEFAK DEPLOYMENT
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat artefak model…")
def load_bundle():
    path = os.path.join(DEPLOY_DIR, "gnn.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_data(show_spinner="Memuat data SNA…")
def load_sna_df():
    path = os.path.join(DEPLOY_DIR, "df_sna_results.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data(show_spinner="Memuat edge list…")
def load_edges():
    path = os.path.join(DEPLOY_DIR, "graph_edges.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data(show_spinner="Memuat histori training…")
def load_history():
    path = os.path.join(DEPLOY_DIR, "training_history.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_resource(show_spinner="Memuat bobot GNN…")
def load_gnn_model(_bundle):
    if not ML_AVAILABLE or _bundle is None or GNNInfluencer is None:
        return None
    pt_path = os.path.join(DEPLOY_DIR, "gnn_model_state.pt")
    if not os.path.exists(pt_path):
        return None
    try:
        ckpt = torch.load(pt_path, map_location="cpu")
        model = GNNInfluencer(
            in_dim     = ckpt["input_dim"],
            hidden_dim = ckpt["hidden_dim"],
            heads      = ckpt["heads"],
            dropout    = ckpt["dropout"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────────────
# TAMPILAN DASHBOARD
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card { background: #1e1e2e; border-radius: 12px; padding: 18px 22px; text-align: center; }
    .metric-card h2 { margin: 0; font-size: 2rem; }
    .metric-card p  { margin: 0; color: #aaa; font-size: .85rem; }
    .tier1 { border-left: 4px solid #e74c3c; }
    .tier2 { border-left: 4px solid #f39c12; }
    .tier3 { border-left: 4px solid #3498db; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background: #1e1e2e; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

bundle  = load_bundle()
df_sna  = load_sna_df()
df_edge = load_edges()
history = load_history()
model   = load_gnn_model(bundle) if bundle else None
DATA_OK = bundle is not None and not df_sna.empty

# ── UNCACHED DEBUG TEST ──────────────────────────────────────────────────────
with st.expander("🔍 Debug: Test Load Langsung", expanded=True):
    import traceback
    _pkl_path = os.path.join(DEPLOY_DIR, "gnn.pkl")
    _csv_path = os.path.join(DEPLOY_DIR, "df_sna_results.csv")
    st.code(f"DEPLOY_DIR : {DEPLOY_DIR}")
    st.code(f"bundle     : {type(bundle)} | keys: {list(bundle.keys()) if isinstance(bundle, dict) else 'N/A'}")
    st.code(f"df_sna     : {df_sna.shape if not df_sna.empty else 'EMPTY'}")
    st.code(f"DATA_OK    : {DATA_OK}")
    # Test joblib langsung tanpa cache
    try:
        _raw = joblib.load(_pkl_path)
        st.success(f"✅ joblib.load OK → type={type(_raw)}, keys={list(_raw.keys()) if isinstance(_raw, dict) else 'N/A'}")
    except Exception as _e:
        st.error(f"❌ joblib.load GAGAL: {_e}")
        st.code(traceback.format_exc())
    # Test pd.read_csv langsung
    try:
        _df = pd.read_csv(_csv_path, nrows=3)
        st.success(f"✅ pd.read_csv OK → shape={_df.shape}, cols={list(_df.columns)}")
    except Exception as _e:
        st.error(f"❌ pd.read_csv GAGAL: {_e}")
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/color/96/tiktok--v1.png", width=60)
    st.title("@NYCXSTORE")
    st.caption("Identifikasi Influencer TikTok\nGraphSAGE + GAT + IndoBERT")
    st.divider()

    if not ML_AVAILABLE:
        st.error(f"⚠️ PyTorch/Transformers gagal dimuat. Mode Terdegradasi Aktif.\n\nDetail: {ML_ERROR_MSG}")

    # DEBUG — hapus setelah deployment sukses
    with st.expander("🔍 Debug Info", expanded=False):
        st.code(f"DEPLOY_DIR : {DEPLOY_DIR}")
        st.code(f"Dir exists : {os.path.isdir(DEPLOY_DIR)}")
        if os.path.isdir(DEPLOY_DIR):
            files = os.listdir(DEPLOY_DIR)
            st.code(f"Files      : {files}")
            for f in ["gnn.pkl", "df_sna_results.csv", "graph_edges.csv", "training_history.json", "gnn_model_state.pt"]:
                fp = os.path.join(DEPLOY_DIR, f)
                size = os.path.getsize(fp) if os.path.exists(fp) else -1
                st.code(f"{f}: {'✅' if size > 0 else '❌'} ({size:,} bytes)")

    if DATA_OK:
        di = bundle["dataset_info"]
        mp = bundle["model_perf"]
        st.markdown("**📦 Dataset**")
        st.markdown(f"- Users: **{di['n_users']:,}**")
        st.markdown(f"- Komentar: **{di['n_comments']:,}**")
        st.markdown(f"- Video: **{di['n_videos']:,}**")
        st.divider()
        st.markdown("**🧠 Model Performance**")
        st.markdown(f"- Test F1 Macro: **{mp['test_f1_macro']}**")
        st.markdown(f"- AUC-ROC: **{mp['test_auc_roc']}**")
        st.markdown(f"- Best Val F1: **{mp['best_val_f1']}**")
    else:
        st.error("⚠️ Folder `deployment/` tidak ditemukan.")

st.title("📊 Dashboard Influencer @NYCXSTORE TikTok")
st.markdown("**Universitas Telkom Purwokerto** · Sistem Informasi 2022 · Vito Hermansyah (2211103095)")

if not DATA_OK:
    st.warning("Data deployment belum tersedia. Pastikan folder `deployment/` sudah ada.")
    st.stop()

di = bundle["dataset_info"]
td = bundle.get("tier_distribution", {})

col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.markdown(f'<div class="metric-card"><h2>{di["n_users"]:,}</h2><p>Total Users</p></div>', unsafe_allow_html=True)
with col2: st.markdown(f'<div class="metric-card tier1"><h2>{td.get("Tier 1 - Top Influencer", 0)}</h2><p>🔴 Top Influencer</p></div>', unsafe_allow_html=True)
with col3: st.markdown(f'<div class="metric-card tier2"><h2>{td.get("Tier 2 - Micro Influencer", 0)}</h2><p>🟡 Micro Influencer</p></div>', unsafe_allow_html=True)
with col4: st.markdown(f'<div class="metric-card tier3"><h2>{td.get("Tier 3 - Regular User", 0)}</h2><p>🔵 Regular User</p></div>', unsafe_allow_html=True)
with col5: st.markdown(f'<div class="metric-card"><h2>{di["graph_edges"]:,}</h2><p>Graf Edges</p></div>', unsafe_allow_html=True)

st.divider()

# Tab dikurangi menjadi 4
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Ranking Influencer", 
    "📈 Analitik SNA", 
    "🕸️ Jaringan Graf", 
    "🤖 Training GNN"
])

with tab1:
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1: sel_tier = st.multiselect("Filter Tier", options=TIER_ORDER, default=TIER_ORDER[:2])
    with col_f2: top_n = st.slider("Tampilkan Top-N", 5, 50, 20)

    df_show = df_sna[df_sna["tier"].isin(sel_tier)].nlargest(top_n, "final_score").reset_index(drop=True)
    df_show.index += 1
    fig_bar = px.bar(df_show, x="username", y="final_score", color="tier", color_discrete_map=TIER_COLORS, text=df_show["final_score"].round(3), template="plotly_dark", height=420)
    fig_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)
    st.dataframe(df_show, use_container_width=True, height=420)

with tab2:
    fig_sc = px.scatter(df_sna, x="pagerank", y="betweenness_cent", color="tier", color_discrete_map=TIER_COLORS, hover_data=["username"], opacity=0.65, template="plotly_dark", height=430)
    st.plotly_chart(fig_sc, use_container_width=True)

with tab3:
    st.subheader("🕸️ Visualisasi Jaringan Sosial")
    st.caption("Menampilkan subgraph berdasarkan Top-N user dengan Final Score tertinggi.")
    
    n_nodes = st.slider("Jumlah node (user) yang ditampilkan", min_value=10, max_value=100, value=20, step=5)
    
    top_df = df_sna.nlargest(n_nodes, "final_score")
    top_usernames = set(top_df["username"])
    
    if not df_edge.empty:
        filtered_edges = df_edge[df_edge["source"].isin(top_usernames) & df_edge["target"].isin(top_usernames)]
    else:
        filtered_edges = pd.DataFrame(columns=["source", "target"])
        
    G = nx.Graph()
    for _, row in top_df.iterrows():
        G.add_node(row["username"], tier=row["tier"], score=row["final_score"])
    for _, row in filtered_edges.iterrows():
        G.add_edge(row["source"], row["target"])
        
    pos = nx.spring_layout(G, k=0.45, seed=42)
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='#555555'),
        hoverinfo='none', mode='lines'
    )
    
    node_x, node_y, node_colors, node_text, node_sizes = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        tier_val = G.nodes[node].get("tier", "Tier 3 - Regular User")
        score_val = G.nodes[node].get("score", 0)
        node_colors.append(TIER_COLORS.get(tier_val, "#3498db"))
        node_text.append(f"<b>{node}</b><br>Tier: {tier_val}<br>Score: {score_val:.4f}")
        node_sizes.append(np.clip(score_val * 35, 12, 32))
        
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="top center",
        textfont=dict(size=9, color="white"),
        hovertext=node_text, hoverinfo='text',
        marker=dict(showscale=False, color=node_colors, size=node_sizes, line_width=1.5, line_color='#ffffff')
    )
    
    legend_traces = []
    for t_name, t_col in TIER_COLORS.items():
        legend_traces.append(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=t_col),
            name=t_name.split(" - ")[-1]
        ))

    fig_graph = go.Figure(data=[edge_trace, node_trace] + legend_traces)
    
    # PERBAIKAN SINTAKS PLOTLY DI SINI (title dibungkus kamus dict)
    fig_graph.update_layout(
        title=dict(
            text=f"Jaringan Sosial @NYCXSTORE — Top {n_nodes} Nodes",
            font=dict(size=14)
        ),
        showlegend=True, template="plotly_dark", height=540,
        legend=dict(x=0.85, y=0.98, bgcolor="rgba(0,0,0,0)"),
        margin=dict(b=20, l=10, r=10, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    st.plotly_chart(fig_graph, use_container_width=True)
    
    st.markdown("### Statistik Graf")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1: st.markdown(f"**Nodes**<br><span style='font-size:1.5rem'>{di.get('n_users', len(G.nodes())):,}</span>", unsafe_allow_html=True)
    with sc2: st.markdown(f"**Edges**<br><span style='font-size:1.5rem'>{di.get('graph_edges', len(G.edges())):,}</span>", unsafe_allow_html=True)
    with sc3: st.markdown(f"**Density**<br><span style='font-size:1.5rem'>{nx.density(G):.4f}</span>", unsafe_allow_html=True)
    with sc4: st.markdown(f"**Komunitas**<br><span style='font-size:1.5rem'>{len(list(nx.connected_components(G))):,}</span>", unsafe_allow_html=True)

with tab4:
    if not history:
        st.info("Histori training tidak ditemukan.")
    else:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=history["train_loss"], name="Train Loss", line=dict(color="royalblue", width=2)))
        fig_loss.add_trace(go.Scatter(y=history["val_loss"], name="Val Loss", line=dict(color="coral", width=2)))
        
        # PERBAIKAN SINTAKS PLOTLY DI SINI
        fig_loss.update_layout(
            title=dict(text="Loss Curve"), 
            template="plotly_dark", height=380
        )
        st.plotly_chart(fig_loss, use_container_width=True)
