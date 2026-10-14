"""Emoji Market · System Intelligence Dashboard.

A redesigned Streamlit dashboard inspired by the reference UI:
- Top KPI card row with gradient backgrounds
- Left sidebar for all controls
- Main area: capital chart, heatmap, network graph, intelligence metrics, event log
"""

import time
import io
import math
import random
import urllib.parse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx

import streamlit as st

from market_engine import (
    COMMODITIES,
    SimState,
    compute_intelligence_metrics,
    step,
)

# ──────────────────────────────────────────
# Page config & theme
# ──────────────────────────────────────────
st.set_page_config(
    page_title="Emoji Market Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+SC:wght@400;500;700&display=swap" rel="stylesheet">
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
:root {
    --brand: #3b82f6;
    --brand-dark: #1e3a8a;
    --accent: #f59e0b;
    --success: #10b981;
    --danger: #ef4444;
    --card-bg: #ffffff;
    --card-shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
    --text-primary: #1e293b;
    --text-secondary: #64748b;
    --bg: #f1f5f9;
}
html, body, .stApp {
    font-family: 'Inter', 'Noto Sans SC', -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text-primary);
}

/* Top header */
.app-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    color: #fff;
    padding: 14px 28px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(30,58,138,0.25);
}
.app-header .title {
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}
.app-header .subtitle {
    font-size: 0.8rem;
    opacity: 0.85;
    margin-top: 2px;
}
.app-header .run-controls {
    display: flex;
    gap: 10px;
    align-items: center;
}
.app-header .run-controls .stButton > button {
    background: rgba(255,255,255,0.2);
    color: #fff;
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 10px;
    padding: 6px 18px;
    font-weight: 600;
    backdrop-filter: blur(4px);
}
.app-header .run-controls .stButton > button:hover {
    background: rgba(255,255,255,0.3);
}

/* KPI cards */
.kpi-row {
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
}
.kpi-card {
    flex: 1;
    background: var(--card-bg);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: var(--card-shadow);
    border-top: 4px solid var(--brand);
    transition: transform 0.15s ease;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-card .label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin-bottom: 6px;
}
.kpi-card .value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1.1;
}
.kpi-card .delta {
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 4px;
}
.kpi-card .delta.up { color: var(--success); }
.kpi-card .delta.down { color: var(--danger); }
.kpi-card.accent { border-top-color: var(--accent); }
.kpi-card.success { border-top-color: var(--success); }
.kpi-card.danger  { border-top-color: var(--danger); }

/* Panel (white card) */
.panel {
    background: var(--card-bg);
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: var(--card-shadow);
    margin-bottom: 16px;
}
.panel h3, .panel h4 {
    font-weight: 700;
    margin-bottom: 12px;
    color: var(--text-primary);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border-right: 1px solid #e2e8f0;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--brand-dark);
}

/* Streamlit element overrides */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
    border: none !important;
    box-shadow: var(--card-shadow) !important;
    background: var(--card-bg) !important;
}
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}
.stMetric {
    background: var(--card-bg);
    border-radius: 10px;
    padding: 10px 14px;
    box-shadow: var(--card-shadow);
}
div[data-testid="stMetricValue"] {
    font-size: 1.4rem;
    font-weight: 800;
}

/* Event log */
.event-log {
    max-height: 300px;
    overflow-y: auto;
    font-family: 'SFMono-Regular', Consolas, monospace;
    font-size: 0.75rem;
    line-height: 1.6;
    color: var(--text-secondary);
}
.event-log .entry {
    padding: 3px 0;
    border-bottom: 1px solid #f1f5f9;
}
.event-log .entry.crisis { color: var(--danger); font-weight: 700; }
.event-log .entry.trade  { color: var(--brand-dark); }
.event-log .entry.tick   { color: var(--text-secondary); }
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────
# Session state initialization
# ──────────────────────────────────────────
DEFAULT_PARAMS = {
    "crisis_prob": 0.05,
    "profit_threshold": 1.05,
    "recovery_rate": 0.1,
    "trading_tax": 0.0,
    "sim_speed": 0.5,
}

COMMODITY_NAMES = [c[0] for c in COMMODITIES]

PRESETS = {
    "🏔️ Highland Crisis": {
        "crisis_prob": 0.15,
        "profit_threshold": 1.08,
        "recovery_rate": 0.03,
    },
    "🤖 AI War": {
        "crisis_prob": 0.05,
        "profit_threshold": 1.02,
        "recovery_rate": 0.20,
    },
    "⚡ Stable Equilibrium": {
        "crisis_prob": 0.02,
        "profit_threshold": 1.10,
        "recovery_rate": 0.15,
    },
}


def get_state() -> SimState:
    if "sim_state" not in st.session_state:
        s = SimState()
        s.init_agents()
        st.session_state.sim_state = s
    return st.session_state.sim_state


def get_params() -> dict:
    if "params" not in st.session_state:
        st.session_state.params = dict(DEFAULT_PARAMS)
        st.session_state.params["agent_targets"] = {
            "Alpha-Bot": "Wine",
            "Beta-Bot": "Wine",
        }
        st.session_state.params["agent_risks"] = {
            "Alpha-Bot": 0.3,
            "Beta-Bot": 0.7,
        }
        st.session_state.params["agent_volumes"] = {
            "Alpha-Bot": 0.8,
            "Beta-Bot": 1.0,
        }
    return st.session_state.params


def reset_state():
    configs = {}
    for a in get_state().agents:
        configs[a.name] = {
            "capital": a.capital,
            "target": a.target,
            "risk": a.risk,
            "volume_frac": a.volume_frac,
        }
    st.session_state.sim_state = SimState()
    st.session_state.sim_state.init_agents(configs)
    st.rerun()


# ──────────────────────────────────────────
# Sidebar — Control panel
# ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")

    state = get_state()
    params = get_params()

    st.markdown("### 🎛️ Simulation Parameters")
    params["crisis_prob"] = st.slider(
        "Crisis Probability", 0.0, 0.25, params["crisis_prob"], 0.01, format="%.0f%%"
    )
    params["profit_threshold"] = st.slider(
        "Profit Threshold", 1.0, 1.25, params["profit_threshold"], 0.01
    )
    params["recovery_rate"] = st.slider(
        "Market Recovery Rate", 0.01, 0.35, params["recovery_rate"], 0.01
    )
    params["sim_speed"] = st.slider(
        "Sim Speed (s/step)", 0.1, 2.0, params["sim_speed"], 0.1
    )

    st.markdown("---")
    st.markdown("### 🤖 Agent Configuration")
    for agent in state.agents:
        with st.expander(f"**{agent.name}**", expanded=False):
            params["agent_targets"][agent.name] = st.selectbox(
                "Target Commodity",
                COMMODITY_NAMES,
                index=COMMODITY_NAMES.index(
                    params["agent_targets"].get(agent.name, agent.target)
                ),
                key=f"target_{agent.name}",
            )
            agent.capital = st.number_input(
                "Capital",
                1,
                1_000_000,
                int(agent.capital),
                key=f"capital_{agent.name}",
            )
            params["agent_risks"][agent.name] = st.slider(
                "Risk Preference",
                0.0,
                1.0,
                params["agent_risks"].get(agent.name, agent.risk),
                0.05,
                key=f"risk_{agent.name}",
            )
            params["agent_volumes"][agent.name] = st.slider(
                "Volume Fraction",
                0.05,
                1.0,
                params["agent_volumes"].get(agent.name, agent.volume_frac),
                0.05,
                key=f"volume_{agent.name}",
            )
            eff = 1 + (params["profit_threshold"] - 1) * (
                1 - 0.5 * params["agent_risks"][agent.name]
            )
            st.caption(
                f"Effective threshold ≈ {eff:.3f} | "
                f"Impact ≈ {agent.capital * params['agent_volumes'][agent.name] * 0.0001:.4f}"
            )

    st.markdown("---")
    st.markdown("### 🏦 Policy Sandbox")
    params["trading_tax"] = st.slider(
        "Trading Tax Rate", 0.0, 0.30, params["trading_tax"], 0.01, format="%.0f%%"
    )

    cb_target = st.selectbox(
        "CB Operation Target", [a.name for a in state.agents], key="cb_target"
    )
    cb_amount = st.number_input(
        "CB Amount", 1, 100_000, 50, key="cb_amount"
    )
    cb1, cb2 = st.columns(2)
    with cb1:
        if st.button("💸 Inject", use_container_width=True):
            agent = next(a for a in state.agents if a.name == cb_target)
            state.market.inject_liquidity(agent, cb_amount)
            st.session_state.sim_state.logs.append(
                f"[CB] Injected {cb_amount} → {cb_target} | balance {agent.capital:.1f}"
            )
            st.rerun()
    with cb2:
        if st.button("💵 Drain", use_container_width=True):
            agent = next(a for a in state.agents if a.name == cb_target)
            state.market.drain_liquidity(agent, cb_amount)
            st.session_state.sim_state.logs.append(
                f"[CB] Drained {cb_amount} from {cb_target} | balance {agent.capital:.1f}"
            )
            st.rerun()

    edge_list = [f"{u}→{v}" for u, v, _ in state.market.G.edges(data=True)]
    if edge_list:
        repair_edge = st.selectbox("Targeted Repair Edge", edge_list, key="repair_edge")
        if st.button("🔧 Reset to Base", use_container_width=True):
            u, v = repair_edge.split("→")
            if state.market.reset_edge((u, v)):
                st.session_state.sim_state.logs.append(
                    f"[CB] Repaired {u}→{v} to base rate"
                )
                st.rerun()

    st.markdown("---")
    st.markdown("### 📋 Preset Scenarios")
    preset_name = st.selectbox(
        "Load Scenario", ["(none)"] + list(PRESETS.keys()), key="preset_select"
    )
    if preset_name != "(none)":
        if st.button("✅ Apply", use_container_width=True):
            p = PRESETS[preset_name]
            params.update(p)
            if "AI War" in preset_name:
                for name in params["agent_risks"]:
                    params["agent_risks"][name] = 0.9
                    params["agent_volumes"][name] = 1.0
            elif "Stable" in preset_name:
                for name in params["agent_risks"]:
                    params["agent_risks"][name] = 0.2
                    params["agent_volumes"][name] = 0.3
            st.session_state.sim_state.logs.append(f"[Scenario] Loaded: {preset_name}")
            st.rerun()

    st.markdown("---")
    if st.button("🔄 Reset Simulation", use_container_width=True):
        reset_state()


# ──────────────────────────────────────────
# Header bar
# ──────────────────────────────────────────
header_cols = st.columns([5, 2])
with header_cols[0]:
    st.markdown(
        """
        <div class="app-header">
            <div>
                <div class="title">🌍 Emoji Market · System Intelligence Dashboard</div>
                <div class="subtitle">Market ↔ Intelligence isomorphism · from dialogue to dashboard</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_cols[1]:
    st.markdown("")
    auto_run = st.toggle("▶ Auto Run", value=False, key="auto_run")
    step_clicked = st.button("⏭ Step", use_container_width=True, key="step_btn")

# --- Run logic ---
if auto_run:
    events = step(state, params)
    state.logs.extend(f"[T{state.tick}] {e}" for e in events)
    time.sleep(params["sim_speed"])
    st.rerun()
elif step_clicked:
    events = step(state, params)
    state.logs.extend(f"[T{state.tick}] {e}" for e in events)
    st.rerun()

st.markdown("")

# ──────────────────────────────────────────
# KPI Cards row
# ──────────────────────────────────────────
total_cap = sum(a.capital for a in state.agents)
crisis_count = len(state.crisis_marks)
total_vol = sum(d.get("volume", 0) for _, _, d in state.market.G.edges(data=True))
active_agents = sum(1 for a in state.agents if a.capital > 10)

m_now = compute_intelligence_metrics(
    state.market, state.tick, state.crisis_marks
)

# Delta calculations
prev_cap = (
    state.capital_history[state.agents[0].name][-2]
    if len(state.capital_history[state.agents[0].name]) > 1
    else total_cap
)
cap_delta_pct = ((total_cap - prev_cap) / max(prev_cap, 0.01)) * 100 if prev_cap else 0

prev_phi = (
    state.metrics_history[-1]["phi"]
    if state.metrics_history
    else m_now["phi"]
)
phi_delta = m_now["phi"] - prev_phi

st.markdown('<div class="kpi-row">', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="label">💰 Total Capital</div>
            <div class="value">{total_cap:,.1f}</div>
            <div class="delta {'up' if cap_delta_pct >= 0 else 'down'}">
                {'▲' if cap_delta_pct >= 0 else '▼'} {abs(cap_delta_pct):.2f}% / tick
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    st.markdown(
        f"""
        <div class="kpi-card accent">
            <div class="label">🧠 Φ Integration</div>
            <div class="value">{m_now['phi']:.3f}</div>
            <div class="delta {'up' if phi_delta >= 0 else 'down'}">
                {'▲' if phi_delta >= 0 else '▼'} {abs(phi_delta):.3f} change
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    cl = m_now["cognitive_load"]
    st.markdown(
        f"""
        <div class="kpi-card success">
            <div class="label">⚙️ Cognitive Load</div>
            <div class="value">{cl if cl is not None else '—'}</div>
            <div class="delta up">avg path length</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    lr = m_now["learning_rate"]
    st.markdown(
        f"""
        <div class="kpi-card danger">
            <div class="label">📚 Learning Rate</div>
            <div class="value">{lr if lr is not None else '—'}</div>
            <div class="delta up">since last crisis</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k5:
    re_val = m_now["repr_efficiency"]
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="label">🔗 Repr. Efficiency</div>
            <div class="value">{re_val:.1%}</div>
            <div class="delta up">mediator volume</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────
# Main charts row: Capital Evolution | Heatmap
# ──────────────────────────────────────────
c1, c2 = st.columns([3, 2])

with c1:
    st.markdown("### 📈 Capital Evolution")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#3b82f6", "#f59e0b"]
    for i, (name, caps) in enumerate(state.capital_history.items()):
        ax.plot(
            range(len(caps)),
            caps,
            label=name,
            linewidth=2.2,
            color=colors[i % len(colors)],
        )
    for tick_num, _edges in state.crisis_marks:
        if tick_num < len(list(state.capital_history.values())[0]):
            ax.axvline(x=tick_num, color="#ef4444", linestyle="--", alpha=0.35, linewidth=1)
    ax.set_xlabel("Tick", fontsize=9)
    ax.set_ylabel("Capital", fontsize=9)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=8)
    ax.set_facecolor("#fafbfc")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    st.pyplot(fig)

with c2:
    st.markdown("### 🔥 Exchange Rate Heatmap")
    heat_mode = st.radio(
        "Mode",
        ["Rate", "Volume", "Deviation"],
        horizontal=True,
        key="heat_mode",
        label_visibility="collapsed",
    )
    market = state.market
    commodities = list(market.G.nodes())
    n = len(commodities)
    idx = {c: i for i, c in enumerate(commodities)}
    matrix = np.full((n, n), np.nan)
    for u, v, data in market.G.edges(data=True):
        if heat_mode == "Rate":
            val = data["rate"]
        elif heat_mode == "Volume":
            val = data["volume"] if data["volume"] > 0 else np.nan
        else:
            base = market.base_rates.get((u, v))
            val = math.log(data["rate"] / base) if base else np.nan
        matrix[idx[u], idx[v]] = val

    masked = np.ma.masked_invalid(matrix)
    finite = matrix[np.isfinite(matrix)]
    fig_h, ax_h = plt.subplots(figsize=(4, 3.5))
    if len(finite):
        if heat_mode == "Deviation":
            m_val = max(abs(finite.min()), abs(finite.max()), 0.05)
            norm = mcolors.TwoSlopeNorm(vmin=-m_val, vcenter=0.0, vmax=m_val)
            cmap = "RdBu_r"
        else:
            vmin, vmax = finite.min(), finite.max()
            if vmin == vmax:
                vmax = vmin * 1.1
            norm = mcolors.LogNorm(vmin=max(vmin, 1e-5), vmax=vmax)
            cmap = "YlOrRd"
        im = ax_h.imshow(masked, cmap=cmap, norm=norm)
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if np.isnan(val):
                    ax_h.text(j, i, "—", ha="center", va="center", color="lightgray", fontsize=7)
                else:
                    if heat_mode == "Rate":
                        txt = f"{val:.2f}"
                    elif heat_mode == "Volume":
                        txt = f"{val:,.0f}"
                    else:
                        txt = f"{val:+.2f}"
                    nr = norm.vmax - norm.vmin
                    frac = (val - norm.vmin) / nr if nr > 0 else 0.5
                    color = "white" if frac > 0.55 else "black"
                    ax_h.text(j, i, txt, ha="center", va="center", color=color, fontsize=6, fontweight="bold")
        fig_h.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04, label=heat_mode.lower())
    else:
        ax_h.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_h.transAxes)
    ax_h.set_xticks(range(n), [c[:4] for c in commodities], fontsize=7)
    ax_h.set_yticks(range(n), [c[:4] for c in commodities], fontsize=7)
    ax_h.set_facecolor("#fafbfc")
    fig_h.patch.set_facecolor("white")
    fig_h.tight_layout()
    st.pyplot(fig_h)

# ──────────────────────────────────────────
# Second row: Market Network | Intelligence Trend | Event Log
# ──────────────────────────────────────────
r1, r2, r3 = st.columns([2, 2, 2])

with r1:
    st.markdown("### 🕸️ Market Network")
    fig_n, ax_n = plt.subplots(figsize=(5, 4))
    G = state.market.G
    pos = nx.circular_layout(G)
    node_colors = []
    for node in G.nodes():
        if node in {"Wool", "Wheat", "Iron"}:
            node_colors.append("#f59e0b")
        elif node in {"Sheep"}:
            node_colors.append("#3b82f6")
        elif node in {"Wine"}:
            node_colors.append("#10b981")
        else:
            node_colors.append("#8b5cf6")
    edge_widths = [
        max(0.5, min(3.0, state.market.G[u][v].get("volume", 0) / 50 + 0.5))
        for u, v in G.edges()
    ]
    edge_colors = []
    for u, v in G.edges():
        rate = state.market.G[u][v].get("rate", 0)
        base = state.market.base_rates.get((u, v), rate)
        dev = (rate / base - 1) if base else 0
        if dev < -0.2:
            edge_colors.append("#ef4444")
        elif dev > 0.2:
            edge_colors.append("#10b981")
        else:
            edge_colors.append("#94a3b8")
    nx.draw_networkx_nodes(G, pos, ax=ax_n, node_color=node_colors, node_size=600, alpha=0.9)
    nx.draw_networkx_edges(
        G, pos, ax=ax_n, edge_color=edge_colors, width=edge_widths,
        arrows=True, arrowsize=14, alpha=0.6, connectionstyle="arc3,rad=0.15",
    )
    labels = {n: d["emoji"] for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax_n, font_size=11)
    ax_n.set_facecolor("#fafbfc")
    fig_n.patch.set_facecolor("white")
    ax_n.axis("off")
    fig_n.tight_layout()
    st.pyplot(fig_n)
    st.caption("🔵 Source | 🟡 Mediator | 🟢 Target | 🔴 Crisis edge | ⚪ Stable edge")

with r2:
    st.markdown("### 📊 Intelligence Metrics Trend")
    hist = state.metrics_history
    if len(hist) < 2:
        st.info("Run more steps to see metrics trend…")
    else:
        df_m = pd.DataFrame(hist).astype(float)
        fig_m, ax_m = plt.subplots(figsize=(5, 3))
        for col, color, label in [
            ("phi", "#3b82f6", "Φ Integration"),
            ("cognitive_load", "#f59e0b", "Cognitive Load"),
            ("learning_rate", "#10b981", "Learning Rate"),
            ("repr_efficiency", "#8b5cf6", "Repr. Efficiency"),
        ]:
            if col in df_m.columns:
                vals = df_m[col].dropna()
                if len(vals) > 0:
                    ax_m.plot(
                        df_m["tick"], df_m[col], label=label, linewidth=1.8, color=color
                    )
        ax_m.set_xlabel("Tick", fontsize=8)
        ax_m.legend(fontsize=7, loc="upper left", framealpha=0.9)
        ax_m.grid(True, alpha=0.15)
        ax_m.tick_params(labelsize=7)
        ax_m.set_facecolor("#fafbfc")
        fig_m.patch.set_facecolor("white")
        fig_m.tight_layout()
        st.pyplot(fig_m)

    # Edge state table
    st.markdown("#### 🔁 Edge State")
    rows = []
    for u, v, data in state.market.G.edges(data=True):
        base = state.market.base_rates.get((u, v), data["rate"])
        rows.append(
            {
                "Edge": f"{state.market.G.nodes[u]['emoji']}{u[:3]}→{v[:3]}",
                "Rate": round(data["rate"], 3),
                "Dev%": round((data["rate"] / base - 1) * 100, 1),
                "Vol": round(data["volume"], 1),
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        height=180,
    )

with r3:
    st.markdown("### 📜 Event Log")
    log_entries = state.logs[-20:] if state.logs else ["No events yet. Press Step to begin."]
    log_html = '<div class="event-log">'
    for entry in reversed(log_entries):
        cls = "tick"
        if "CRISIS" in entry:
            cls = "crisis"
        elif "arb" in entry or "Inject" in entry or "Drain" in entry:
            cls = "trade"
        log_html += f'<div class="entry {cls}">{entry}</div>'
    log_html += "</div>"
    st.markdown(log_html, unsafe_allow_html=True)

# ──────────────────────────────────────────
# Export row
# ──────────────────────────────────────────
st.markdown("---")
e1, e2, e3 = st.columns(3)
with e1:
    st.markdown("**📤 Export Data**")
    hist_df = pd.DataFrame(state.capital_history)
    hist_df.index.name = "tick"
    csv_cap = hist_df.to_csv().encode("utf-8")
    st.download_button(
        "📥 Capital CSV",
        data=csv_cap,
        file_name=f"capital_t{state.tick}.csv",
        mime="text/csv",
    )
    if state.metrics_history:
        metrics_csv = (
            pd.DataFrame(state.metrics_history).to_csv(index=False).encode("utf-8")
        )
        st.download_button(
            "📥 Metrics CSV",
            data=metrics_csv,
            file_name=f"metrics_t{state.tick}.csv",
            mime="text/csv",
        )

with e2:
    st.markdown("**🔗 Scenario Sharing**")
    share_params = {
        "crisis_prob": params["crisis_prob"],
        "profit_threshold": params["profit_threshold"],
        "recovery_rate": params["recovery_rate"],
        "trading_tax": params["trading_tax"],
    }
    encoded = urllib.parse.urlencode({k: str(v) for k, v in share_params.items()})
    st.code(f"?{encoded}", language="text")

with e3:
    st.markdown("**ℹ️ About**")
    st.caption(
        "Dashboard built from a design dialogue. "
        "Market ↔ Intelligence isomorphism: nodes=neurons, edges=synapses, "
        "rates=weights, arbitrage=signal propagation."
    )
    st.markdown(
        "Tutorial: [Emoji Market Tutorial](https://github.com/math4mad/Emoji-Market-Tutorial)"
    )
