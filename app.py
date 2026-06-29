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

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(
    page_title="Influencer @NYCXSTORE",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Global ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Font & Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp { background: #0d0d14; }
section[data-testid="stSidebar"] { background: #111120 !important; border-right: 1px solid #1e1e35; }

/* ── Metric Cards ── */
.kpi-row { display: flex; gap: 12px; margin-bottom: 20px; }
.kpi-card {
    flex: 1; background: linear-gradient(135deg, #161626 0%, #1a1a2e 100%);
    border-radius: 14px; padding: 20px 18px; text-align: center;
    border: 1px solid #252540; transition: transform .2s;
    position: relative; overflow: hidden;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
}
.kpi-card.total::before { background: linear-gradient(90deg, #6366f1, #8b5cf6); }
.kpi-card.tier1::before { background: linear-gradient(90deg, #ef4444, #f97316); }
.kpi-card.tier2::before { background: linear-gradient(90deg, #f59e0b, #eab308); }
.kpi-card.tier3::before { background: linear-gradient(90deg, #3b82f6, #06b6d4); }
.kpi-card.edges::before { background: linear-gradient(90deg, #10b981, #34d399); }
.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem; font-weight: 700; color: #f1f5f9; line-height: 1.1; margin-bottom: 4px;
}
.kpi-label { font-size: .78rem; color: #64748b; font-weight: 500; letter-spacing: .04em; text-transform: uppercase; }

/* ── Section Headers ── */
.section-header {
    display: flex; align-items: center; gap: 10px;
    margin: 20px 0 14px; padding-bottom: 10px;
    border-bottom: 1px solid #1e1e35;
}
.section-header h3 { margin: 0; font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; color: #e2e8f0; font-weight: 600; }
.section-badge {
    background: #1e1e35; border-radius: 20px; padding: 2px 10px;
    font-size: .73rem; color: #818cf8; font-weight: 500;
}

/* ── Sidebar Branding ── */
.sidebar-brand {
    text-align: center; padding: 16px 0 20px;
}
.sidebar-brand .brand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem; font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 8px 0 2px;
}
.sidebar-brand .brand-sub { font-size: .73rem; color: #475569; line-height: 1.5; }

/* ── Sidebar Stats ── */
.stat-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #1e1e2a; }
.stat-label { font-size: .8rem; color: #64748b; }
.stat-value { font-family: 'Space Grotesk', sans-serif; font-size: .9rem; color: #e2e8f0; font-weight: 600; }

/* ── Perf Badges ── */
.perf-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #1a1a2e; border: 1px solid #252540;
    border-radius: 8px; padding: 6px 12px; margin: 4px 0;
    width: 100%;
}
.perf-badge-label { font-size: .77rem; color: #64748b; flex: 1; }
.perf-badge-value { font-family: 'Space Grotesk', sans-serif; font-size: .9rem; font-weight: 700; color: #a78bfa; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #111120; padding: 4px; border-radius: 10px; border: 1px solid #1e1e35; }
.stTabs [data-baseweb="tab"] { background: transparent !important; border-radius: 8px !important; color: #64748b !important; font-weight: 500 !important; padding: 8px 16px !important; font-size: .85rem !important; }
.stTabs [aria-selected="true"] { background: #1e1e35 !important; color: #e2e8f0 !important; }

/* ── Page Title ── */
.page-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem; font-weight: 700;
    background: linear-gradient(135deg, #e2e8f0 0%, #94a3b8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.2;
}
.page-subtitle { font-size: .82rem; color: #475569; margin-top: 4px; }

/* ── Divider ── */
hr { border: none; border-top: 1px solid #1e1e35 !important; margin: 16px 0 !important; }

/* ── Streamlit overrides ── */
div[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }
.stDataFrame { border-radius: 10px; overflow: hidden; border: 1px solid #1e1e35; }
.stSlider label { color: #94a3b8 !important; }
.stNumberInput label { color: #94a3b8 !important; }
.stMultiSelect label { color: #94a3b8 !important; }

/* ── Info / Warning boxes ── */
.status-box {
    border-radius: 10px; padding: 12px 16px;
    font-size: .83rem; margin: 8px 0;
    display: flex; align-items: flex-start; gap: 10px;
}
.status-box.warning { background: #1c1710; border: 1px solid #854d0e; color: #fbbf24; }
.status-box.error   { background: #1c1010; border: 1px solid #7f1d1d; color: #f87171; }
.status-box.info    { background: #101a1c; border: 1px solid #164e63; color: #22d3ee; }
</style>
""", unsafe_allow_html=True)

# ── Konstanta ─────────────────────────────────────────────────────────────
TIER_COLORS = {
    "Tier 1 - Top Influencer":   "#ef4444",
    "Tier 2 - Micro Influencer": "#f59e0b",
    "Tier 3 - Regular User":     "#3b82f6",
}
TIER_ORDER = list(TIER_COLORS.keys())
DEPLOY_DIR = os.path.join(os.path.dirname(__file__), "deployment")

# ── Defensive ML Import ────────────────────────────────────────────────────
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
            self.gat   = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True)
            self.bn2   = nn.BatchNorm1d(hidden_dim)
            self.sage2 = SAGEConv(hidden_dim, hidden_dim // 2)
            self.bn3   = nn.BatchNorm1d(hidden_dim // 2)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim // 2, 64), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(64, out_dim),
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
    ML_ERROR_MSG = str(e)


# ── Cache Loaders ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat artefak model…")
def load_bundle():
    path = os.path.join(DEPLOY_DIR, "gnn.pkl")
    return joblib.load(path) if os.path.exists(path) else None

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
            in_dim=ckpt["input_dim"], hidden_dim=ckpt["hidden_dim"],
            heads=ckpt["heads"], dropout=ckpt["dropout"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model
    except Exception:
        return None


# ── Load Data ─────────────────────────────────────────────────────────────
bundle  = load_bundle()
df_sna  = load_sna_df()
df_edge = load_edges()
history = load_history()
model   = load_gnn_model(bundle) if bundle else None
DATA_OK = bundle is not None and not df_sna.empty


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <img src="https://img.icons8.com/color/96/tiktok--v1.png" width="52">
        <div class="brand-name">@NYCXSTORE</div>
        <div class="brand-sub">Identifikasi Influencer TikTok<br>GraphSAGE · GAT · IndoBERT</div>
    </div>
    """, unsafe_allow_html=True)

    if not ML_AVAILABLE:
        st.markdown(f'<div class="status-box error">⚠️ PyTorch gagal dimuat.<br><small>{ML_ERROR_MSG[:80]}</small></div>', unsafe_allow_html=True)

    if DATA_OK:
        di = bundle["dataset_info"]
        mp = bundle["model_perf"]

        st.markdown("**Dataset**")
        st.markdown(f"""
        <div class="stat-row"><span class="stat-label">Total Users</span><span class="stat-value">{di['n_users']:,}</span></div>
        <div class="stat-row"><span class="stat-label">Komentar</span><span class="stat-value">{di['n_comments']:,}</span></div>
        <div class="stat-row"><span class="stat-label">Video</span><span class="stat-value">{di['n_videos']:,}</span></div>
        """, unsafe_allow_html=True)

        st.markdown("<br>**Model Performance**", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="perf-badge"><span class="perf-badge-label">Test F1 Macro</span><span class="perf-badge-value">{mp['test_f1_macro']}</span></div>
        <div class="perf-badge"><span class="perf-badge-label">AUC-ROC</span><span class="perf-badge-value">{mp['test_auc_roc']}</span></div>
        <div class="perf-badge"><span class="perf-badge-label">Best Val F1</span><span class="perf-badge-value">{mp['best_val_f1']}</span></div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box error">⚠️ Folder <code>deployment/</code> tidak ditemukan.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="stat-label" style="text-align:center;font-size:.72rem">Universitas Telkom Purwokerto · SI 2022<br>Vito Hermansyah · 2211103095</div>', unsafe_allow_html=True)


# ── Page Header ───────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 20px 0 16px;">
    <div class="page-title">📊 Dashboard Influencer TikTok</div>
    <div class="page-subtitle">Analisis Jaringan Sosial &amp; Identifikasi Influencer @NYCXSTORE · GraphSAGE + GAT + IndoBERT</div>
</div>
""", unsafe_allow_html=True)

if not DATA_OK:
    st.markdown('<div class="status-box warning">⚠️ Data deployment belum tersedia. Pastikan folder <code>deployment/</code> sudah ada dan berisi semua artefak.</div>', unsafe_allow_html=True)
    st.stop()

di = bundle["dataset_info"]
mp = bundle["model_perf"]
td = bundle.get("tier_distribution", {})

# ── KPI Cards ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-card total">
        <div class="kpi-value">{di["n_users"]:,}</div>
        <div class="kpi-label">Total Users</div>
    </div>
    <div class="kpi-card tier1">
        <div class="kpi-value">{td.get("Tier 1 - Top Influencer", 0)}</div>
        <div class="kpi-label">🔴 Top Influencer</div>
    </div>
    <div class="kpi-card tier2">
        <div class="kpi-value">{td.get("Tier 2 - Micro Influencer", 0)}</div>
        <div class="kpi-label">🟡 Micro Influencer</div>
    </div>
    <div class="kpi-card tier3">
        <div class="kpi-value">{td.get("Tier 3 - Regular User", 0)}</div>
        <div class="kpi-label">🔵 Regular User</div>
    </div>
    <div class="kpi-card edges">
        <div class="kpi-value">{di["graph_edges"]:,}</div>
        <div class="kpi-label">Graf Edges</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Ranking Influencer",
    "📈 Analitik SNA",
    "🕸️ Jaringan Graf",
    "🤖 Training GNN",
    "📊 Model Performance",
])

# ── Tab 1: Ranking Influencer ─────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header"><h3>🏆 Ranking Influencer</h3><span class="section-badge">Final Score</span></div>', unsafe_allow_html=True)
    col_f1, col_f2 = st.columns([3, 1])
    with col_f1:
        sel_tier = st.multiselect("Filter Tier", options=TIER_ORDER, default=TIER_ORDER[:2], label_visibility="collapsed")
    with col_f2:
        top_n = st.slider("Top-N", 5, 50, 20, label_visibility="collapsed")

    df_show = df_sna[df_sna["tier"].isin(sel_tier)].nlargest(top_n, "final_score").reset_index(drop=True)
    df_show.index += 1

    fig_bar = px.bar(
        df_show, x="username", y="final_score", color="tier",
        color_discrete_map=TIER_COLORS,
        text=df_show["final_score"].round(3),
        template="plotly_dark", height=380,
    )
    fig_bar.update_traces(textposition="outside", textfont_size=10)
    fig_bar.update_layout(
        plot_bgcolor="#0d0d14", paper_bgcolor="#0d0d14",
        margin=dict(t=20, b=20, l=10, r=10),
        xaxis=dict(showgrid=False, tickfont=dict(size=9)),
        yaxis=dict(showgrid=True, gridcolor="#1e1e35"),
        legend=dict(bgcolor="rgba(0,0,0,0)", title=""),
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.dataframe(df_show, use_container_width=True, height=360)


# ── Tab 2: Analitik SNA ───────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header"><h3>📈 Analitik Social Network Analysis</h3><span class="section-badge">Pagerank × Betweenness</span></div>', unsafe_allow_html=True)

    fig_sc = px.scatter(
        df_sna, x="pagerank", y="betweenness_cent", color="tier",
        color_discrete_map=TIER_COLORS,
        hover_data=["username"], opacity=0.7,
        template="plotly_dark", height=430,
    )
    fig_sc.update_layout(
        plot_bgcolor="#0d0d14", paper_bgcolor="#0d0d14",
        margin=dict(t=20, b=20, l=10, r=10),
        xaxis=dict(showgrid=True, gridcolor="#1e1e35"),
        yaxis=dict(showgrid=True, gridcolor="#1e1e35"),
        legend=dict(bgcolor="rgba(0,0,0,0)", title=""),
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # Distribusi tier pie chart
    if td:
        col_p1, col_p2 = st.columns([1, 1])
        with col_p1:
            st.markdown('<div class="section-header"><h3>Distribusi Tier</h3></div>', unsafe_allow_html=True)
            fig_pie = px.pie(
                names=list(td.keys()), values=list(td.values()),
                color=list(td.keys()), color_discrete_map=TIER_COLORS,
                template="plotly_dark", hole=0.45,
            )
            fig_pie.update_layout(
                paper_bgcolor="#0d0d14", plot_bgcolor="#0d0d14",
                margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                font=dict(family="Inter"),
            )
            fig_pie.update_traces(textinfo="percent+label", textfont_size=11)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_p2:
            st.markdown('<div class="section-header"><h3>Statistik Final Score</h3></div>', unsafe_allow_html=True)
            stats_df = df_sna.groupby("tier")["final_score"].describe()[["mean","std","min","max"]].round(4)
            st.dataframe(stats_df, use_container_width=True)


# ── Tab 3: Jaringan Graf ──────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header"><h3>🕸️ Visualisasi Jaringan Sosial</h3><span class="section-badge">Spring Layout</span></div>', unsafe_allow_html=True)
    st.caption("Subgraph berdasarkan Top-N user dengan Final Score tertinggi.")

    col_inp, col_info = st.columns([2, 3])
    with col_inp:
        n_nodes = st.number_input(
            "Jumlah node yang ditampilkan",
            min_value=10, max_value=300, value=50, step=10,
            help="Masukkan jumlah node antara 10–300"
        )

    n_nodes = int(n_nodes)
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

    pos = nx.spring_layout(G, k=0.5, seed=42)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.6, color="#2a2a45"),
        hoverinfo="none", mode="lines",
    )

    node_x, node_y, node_colors, node_text, node_sizes = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        tier_val  = G.nodes[node].get("tier", "Tier 3 - Regular User")
        score_val = G.nodes[node].get("score", 0)
        node_colors.append(TIER_COLORS.get(tier_val, "#3b82f6"))
        node_text.append(f"<b>{node}</b><br>Tier: {tier_val}<br>Score: {score_val:.4f}")
        node_sizes.append(np.clip(score_val * 38, 10, 30))

    show_labels = n_nodes <= 80
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text" if show_labels else "markers",
        text=list(G.nodes()) if show_labels else None,
        textposition="top center",
        textfont=dict(size=8, color="#94a3b8"),
        hovertext=node_text, hoverinfo="text",
        marker=dict(
            showscale=False, color=node_colors, size=node_sizes,
            line=dict(width=1.2, color="#0d0d14"),
        ),
    )

    legend_traces = [
        go.Scatter(x=[None], y=[None], mode="markers",
                   marker=dict(size=10, color=c), name=n.split(" - ")[-1])
        for n, c in TIER_COLORS.items()
    ]

    fig_graph = go.Figure(data=[edge_trace, node_trace] + legend_traces)
    fig_graph.update_layout(
        title=dict(text=f"Jaringan Sosial @NYCXSTORE — Top {n_nodes} Nodes", font=dict(size=13, family="Space Grotesk")),
        showlegend=True, template="plotly_dark", height=560,
        paper_bgcolor="#0d0d14", plot_bgcolor="#0d0d14",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(13,13,20,.8)", bordercolor="#1e1e35", borderwidth=1),
        margin=dict(b=15, l=10, r=10, t=45),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig_graph, use_container_width=True)

    if show_labels is False:
        st.markdown('<div class="status-box info">💡 Label node disembunyikan untuk node &gt; 80. Hover pada node untuk melihat detail.</div>', unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    stats_data = [
        ("Nodes", f"{di.get('n_users', len(G.nodes())):,}"),
        ("Edges", f"{di.get('graph_edges', len(G.edges())):,}"),
        ("Density", f"{nx.density(G):.4f}"),
        ("Komunitas", f"{len(list(nx.connected_components(G))):,}"),
    ]
    for col, (label, val) in zip([sc1, sc2, sc3, sc4], stats_data):
        with col:
            st.markdown(f"""
            <div style="background:#161626;border:1px solid #1e1e35;border-radius:10px;padding:14px 16px;text-align:center;">
                <div style="font-size:.75rem;color:#64748b;text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px;">{label}</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:1.5rem;font-weight:700;color:#e2e8f0;">{val}</div>
            </div>
            """, unsafe_allow_html=True)


# ── Tab 4: Training GNN ───────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header"><h3>🤖 Histori Training GNN</h3><span class="section-badge">Loss & F1</span></div>', unsafe_allow_html=True)

    if not history:
        st.markdown('<div class="status-box info">Histori training tidak ditemukan.</div>', unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=history.get("train_loss", []), name="Train Loss", line=dict(color="#6366f1", width=2)))
            fig_loss.add_trace(go.Scatter(y=history.get("val_loss", []),   name="Val Loss",   line=dict(color="#f43f5e", width=2, dash="dot")))
            fig_loss.update_layout(
                title=dict(text="Loss Curve", font=dict(size=13, family="Space Grotesk")),
                template="plotly_dark", height=340,
                paper_bgcolor="#0d0d14", plot_bgcolor="#0d0d14",
                margin=dict(t=40, b=20, l=10, r=10),
                xaxis=dict(title="Epoch", showgrid=True, gridcolor="#1e1e35"),
                yaxis=dict(title="Loss",  showgrid=True, gridcolor="#1e1e35"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        with c2:
            fig_f1 = go.Figure()
            fig_f1.add_trace(go.Scatter(y=history.get("train_f1", []),  name="Train F1",  line=dict(color="#10b981", width=2)))
            fig_f1.add_trace(go.Scatter(y=history.get("val_f1", []),    name="Val F1",    line=dict(color="#f59e0b", width=2, dash="dot")))
            fig_f1.update_layout(
                title=dict(text="F1 Score Curve", font=dict(size=13, family="Space Grotesk")),
                template="plotly_dark", height=340,
                paper_bgcolor="#0d0d14", plot_bgcolor="#0d0d14",
                margin=dict(t=40, b=20, l=10, r=10),
                xaxis=dict(title="Epoch", showgrid=True, gridcolor="#1e1e35"),
                yaxis=dict(title="F1 Score", showgrid=True, gridcolor="#1e1e35"),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        # Epoch summary table
        if "train_loss" in history:
            epochs = len(history["train_loss"])
            summary = []
            for i in range(0, epochs, max(1, epochs // 10)):
                row = {"Epoch": i+1}
                if "train_loss" in history: row["Train Loss"] = round(history["train_loss"][i], 4)
                if "val_loss"   in history: row["Val Loss"]   = round(history["val_loss"][i], 4)
                if "train_f1"   in history: row["Train F1"]   = round(history["train_f1"][i], 4)
                if "val_f1"     in history: row["Val F1"]     = round(history["val_f1"][i], 4)
                summary.append(row)
            st.markdown('<div class="section-header"><h3>Summary Epoch</h3></div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)


# ── Tab 5: Model Performance ──────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header"><h3>📊 Model Performance</h3><span class="section-badge">Evaluasi GNN</span></div>', unsafe_allow_html=True)

    # Gauge cards
    g1, g2, g3 = st.columns(3)
    perf_items = [
        ("Test F1 Macro",   mp["test_f1_macro"],  "#6366f1", g1),
        ("AUC-ROC",         mp["test_auc_roc"],   "#10b981", g2),
        ("Best Val F1",     mp["best_val_f1"],    "#f59e0b", g3),
    ]
    for label, val, color, col in perf_items:
        val_f = float(val)
        with col:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val_f * 100,
                number={"suffix": "%", "font": {"family": "Space Grotesk", "size": 28, "color": "#e2e8f0"}},
                title={"text": label, "font": {"family": "Inter", "size": 13, "color": "#94a3b8"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"size": 9}},
                    "bar": {"color": color, "thickness": 0.7},
                    "bgcolor": "#161626",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 60],  "color": "#1a1a2e"},
                        {"range": [60, 80], "color": "#1e1e35"},
                        {"range": [80, 100],"color": "#252545"},
                    ],
                    "threshold": {
                        "line": {"color": color, "width": 3},
                        "thickness": 0.85,
                        "value": val_f * 100,
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0d0d14",
                height=240,
                margin=dict(t=30, b=10, l=20, r=20),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Radar chart: multi-metric comparison
    st.markdown('<div class="section-header"><h3>Radar Metrik Model</h3></div>', unsafe_allow_html=True)
    radar_metrics = ["F1 Macro", "AUC-ROC", "Val F1"]
    radar_vals    = [float(mp["test_f1_macro"]), float(mp["test_auc_roc"]), float(mp["best_val_f1"])]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_vals + [radar_vals[0]],
        theta=radar_metrics + [radar_metrics[0]],
        fill="toself",
        fillcolor="rgba(99,102,241,0.15)",
        line=dict(color="#6366f1", width=2),
        name="GNN Model",
        marker=dict(color="#6366f1", size=8),
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#161626",
            angularaxis=dict(linecolor="#1e1e35", gridcolor="#1e1e35", tickfont=dict(size=11, color="#94a3b8")),
            radialaxis=dict(range=[0, 1], linecolor="#1e1e35", gridcolor="#1e1e35", tickfont=dict(size=9, color="#64748b")),
        ),
        paper_bgcolor="#0d0d14",
        template="plotly_dark",
        height=350,
        margin=dict(t=20, b=20, l=40, r=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        font=dict(family="Inter"),
    )
    c_radar, c_table = st.columns([1, 1])
    with c_radar:
        st.plotly_chart(fig_radar, use_container_width=True)

    with c_table:
        st.markdown('<div class="section-header"><h3>Ringkasan Metrik</h3></div>', unsafe_allow_html=True)
        perf_rows = []
        for k, v in mp.items():
            label = k.replace("_", " ").title()
            try:
                val_f = float(v)
                bar_w = int(val_f * 100)
                perf_rows.append({"Metrik": label, "Nilai": v, "Bar": bar_w})
            except Exception:
                pass
        if perf_rows:
            df_perf = pd.DataFrame(perf_rows)
            st.dataframe(df_perf[["Metrik", "Nilai"]], use_container_width=True, hide_index=True)

            # horizontal bar
            fig_hbar = px.bar(
                df_perf, x="Bar", y="Metrik", orientation="h",
                color="Bar", color_continuous_scale=[[0, "#6366f1"], [1, "#10b981"]],
                template="plotly_dark", height=220,
                range_x=[0, 100],
                text=df_perf["Nilai"],
            )
            fig_hbar.update_traces(textposition="outside", textfont_size=11)
            fig_hbar.update_layout(
                paper_bgcolor="#0d0d14", plot_bgcolor="#0d0d14",
                margin=dict(t=10, b=10, l=10, r=50),
                showlegend=False, coloraxis_showscale=False,
                xaxis=dict(showgrid=True, gridcolor="#1e1e35"),
                yaxis=dict(showgrid=False),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig_hbar, use_container_width=True)
