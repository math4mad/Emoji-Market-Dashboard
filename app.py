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
    menu_items=None,
)

# Hide the default Streamlit toolbar (the white bar at the very top
# with "Deploy" / hamburger / fork icons) so our custom dark header
# sits flush against the viewport top.
st.markdown(
    """
<style>
[data-testid="stToolbar"]        { display: none !important; }
[data-testid="stDecoration"]     { display: none !important; }
#MainMenu                          { display: none !important; }
header[data-testid="stHeader"]    { display: none !important; }
</style>
""",
    unsafe_allow_html=True,
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
/* ========== Dark theme inspired by Agent Capital Simulator reference ========== */
:root {
    --bg: #0f1115;
    --panel: #171a21;
    --panel2: #1d212b;
    --line: #2a2f3a;
    --text: #dbe2ee;
    --muted: #8892a6;
    --cyan: #22d3ee;
    --blue: #3b82f6;
    --red: #ef4444;
    --green: #22c55e;
    --amber: #f59e0b;
    --purple: #a78bfa;
    --radius: 10px;
}
html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}
.block-container { padding-top: 0.6rem; padding-bottom: 1rem; max-width: 100%; }

/* ========== Top bar ========== */
.app-header {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 12px 18px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
}
.app-header h1 {
    font-size: 17px;
    font-weight: 600;
    letter-spacing: 0.3px;
    color: var(--text);
    margin: 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.app-header .header-spacer { flex: 1; }
.app-header .timeline-block {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--panel2);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 4px 10px;
}
.app-header .timeline-block .tick {
    color: var(--cyan);
    font-size: 12px;
    font-weight: 600;
    min-width: 32px;
    font-variant-numeric: tabular-nums;
}
.app-header .timeline-block input[type=range] {
    width: 180px;
    accent-color: var(--cyan);
}
.app-header .scenario-tag {
    background: var(--panel2);
    border: 1px solid var(--line);
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
    color: var(--muted);
}
.app-header .scenario-tag b { color: var(--cyan); }

/* ========== KPI cards (5 columns with gauge) ========== */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 14px;
}
.kpi-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 12px 14px;
    position: relative;
    overflow: hidden;
}
.kpi-card .kpi-head {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text);
    font-weight: 600;
    margin-bottom: 2px;
}
.kpi-card .kpi-dot {
    width: 8px; height: 8px; border-radius: 50%;
    display: inline-block;
}
.kpi-card .kpi-value {
    font-size: 22px;
    font-weight: 800;
    color: var(--text);
    line-height: 1.15;
    font-variant-numeric: tabular-nums;
    font-family: 'SF Mono', Consolas, monospace;
    margin-top: 2px;
}
.kpi-card .kpi-sub {
    font-size: 11px;
    color: var(--muted);
    margin-top: 2px;
}
.kpi-card .kpi-delta {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 4px;
    margin-top: 4px;
}
.kpi-card .kpi-delta.up   { background: rgba(34,197,94,.12); color: var(--green); }
.kpi-card .kpi-delta.down { background: rgba(239,68,68,.12); color: var(--red); }
.kpi-card .kpi-delta.warn { background: rgba(245,158,11,.12); color: var(--amber); }
.kpi-card .gauge {
    position: absolute;
    right: 8px; top: 10px;
    width: 44px; height: 26px;
    opacity: 0.9;
}

/* ========== Panel (chart card) ========== */
.panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: var(--radius);
    padding: 14px 16px;
    margin-bottom: 14px;
}
.panel h3, .panel h4 {
    color: var(--text);
    font-size: 14px;
    font-weight: 600;
    margin: 0 0 10px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.panel h3 .hint {
    font-size: 11px;
    color: var(--muted);
    font-weight: 400;
}
.panel h4 { font-size: 13px; margin-bottom: 8px; }

/* Heatmap tab (Rate / Volume / Deviation) */
.tabs-row {
    display: flex; gap: 4px;
    background: var(--panel2);
    border-radius: 6px;
    padding: 3px;
    margin-bottom: 8px;
}
.tabs-row label {
    color: var(--muted);
    font-size: 12px;
    cursor: pointer;
    padding: 2px 8px;
    border-radius: 4px;
}
.tabs-row label:has(input:checked) {
    background: var(--cyan);
    color: #0a0d12;
    font-weight: 600;
}

/* Event log rows */
.event-log {
    max-height: 220px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 12.5px;
}
.event-log .entry {
    display: flex;
    gap: 8px;
    align-items: baseline;
    padding: 5px 8px;
    border-radius: 6px;
    background: var(--panel2);
    border-left: 3px solid #6b7280;
    color: var(--muted);
}
.event-log .entry .t { color: var(--muted); font-size: 11px; min-width: 36px; font-variant-numeric: tabular-nums; }
.event-log .entry.crisis { border-left-color: var(--red); color: #fca5a5; }
.event-log .entry.trade  { border-left-color: var(--green); color: #86efac; }
.event-log .entry.policy { border-left-color: var(--blue); color: #93c5fd; }
.event-log .entry.sys    { border-left-color: #6b7280; color: var(--muted); }

/* Edge table */
.panel table { font-size: 12.5px; width: 100%; border-collapse: collapse; }
.panel table th { color: var(--muted); font-weight: 600; font-size: 11px; padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--line); }
.panel table td { color: var(--text); padding: 5px 8px; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; }
.cell-pos { color: var(--green); font-weight: 600; }
.cell-neg { color: var(--red); font-weight: 600; }

/* Streamlit element overrides for dark theme */
.stButton > button {
    background: var(--panel2);
    border: 1px solid var(--line);
    color: var(--text);
    border-radius: 6px;
    font-weight: 600;
    font-size: 13px;
}
.stButton > button:hover {
    border-color: var(--cyan);
    color: var(--cyan);
}
.stButton > button[kind="primary"] {
    background: var(--cyan);
    color: #0a0d12;
    border-color: var(--cyan);
}
.stSelectbox > div, .stNumberInput > div, .stSlider { color: var(--text); }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #171a21 0%, #1d212b 100%) !important;
    border-right: 1px solid var(--line) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--cyan) !important;
}
.stDataFrame, .stTable { color: var(--text) !important; }

/* Footer link row */
.footer-row {
    color: var(--muted);
    font-size: 12px;
    text-align: center;
    margin-top: 8px;
    padding-top: 12px;
    border-top: 1px solid var(--line);
}
.footer-row a { color: var(--cyan); text-decoration: none; }

/* Responsive: stack on small screens */
@media (max-width: 1100px) {
    .kpi-row { grid-template-columns: repeat(2, 1fr); }
}
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
# ── Top bar: logo + timeline + scenario + run controls ──
auto_run = st.toggle("▶ Auto Run", value=False, key="auto_run")
step_clicked = st.button("⏭ Step", use_container_width=False, key="step_btn")

scenario_emoji = {
    "🏔️ Highland Crisis": "🟠 Crisis Mode",
    "🤖 AI War": "🔴 High Volatility",
    "⚡ Stable Equilibrium": "🟢 Equilibrium",
}
current_scenario = st.session_state.get("preset_select", "(none)")
scenario_label = scenario_emoji.get(current_scenario, "🟦 Free Run")

st.markdown(
    f"""
    <div class="app-header">
        <h1>⬡ Emoji Market · System Intelligence</h1>
        <div class="timeline-block">
            <span class="tick">T{state.tick:03d}</span>
        </div>
        <div class="scenario-tag">SCENARIO: <b>{scenario_label}</b></div>
        <div class="header-spacer"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

# ── KPI cards: 5-column grid with small gauge sparkline (reference layout) ──
# Use columns for layout, then render each card's HTML inside
k1, k2, k3, k4, k5 = st.columns(5)

# Historical sparkline data
def _sparkline(caps, color):
    if not caps or len(caps) < 2:
        return ""
    pts = caps[-40:]
    vmin, vmax = min(pts), max(pts)
    rng = max(vmax - vmin, 1e-9)
    coords = []
    for i, v in enumerate(pts):
        x = i * (100 / max(len(pts) - 1, 1))
        y = 22 - (v - vmin) / rng * 20
        coords.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg class="gauge" viewBox="0 0 100 24" style="right:6px;top:36px;width:80px;height:24px">'
        f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="1.5"/>'
        f'</svg>'
    )

# Total Capital sparkline
agent_caps = list(state.capital_history.values())
# Sum across agents at each time step
if agent_caps and all(len(c) > 0 for c in agent_caps):
    min_len = min(len(c) for c in agent_caps)
    agg_caps = [sum(c[i] for c in agent_caps) for i in range(min_len)]
else:
    agg_caps = []
agg_spark = _sparkline(agg_caps, "#22d3ee")

with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-head"><span class="kpi-dot" style="background:#22d3ee"></span>Total Capital</div>
            <div class="kpi-value">{total_cap:,.0f}</div>
            <div class="kpi-sub">Capital Flow</div>
            <div class="kpi-delta {'up' if cap_delta_pct >= 0 else 'down'}">
                {'▲' if cap_delta_pct >= 0 else '▼'} {abs(cap_delta_pct):.2f}% / tick
            </div>
            {agg_spark}
        </div>
        """,
        unsafe_allow_html=True,
    )

# Φ Integration sparkline
phi_history = [m.get("phi", 0) for m in state.metrics_history] if state.metrics_history else []
phi_spark = _sparkline(phi_history, "#a78bfa")
with k2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-head"><span class="kpi-dot" style="background:#a78bfa"></span>Φ Integration</div>
            <div class="kpi-value">{m_now['phi']:.2f}</div>
            <div class="kpi-sub">System Coherence</div>
            <div class="kpi-delta {'up' if phi_delta >= 0 else 'down'}">
                {'▲' if phi_delta >= 0 else '▼'} {abs(phi_delta):.3f} change
            </div>
            {phi_spark}
        </div>
        """,
        unsafe_allow_html=True,
    )

# Cognitive Load
cl = m_now["cognitive_load"]
cl_history = [m.get("cognitive_load", 0) for m in state.metrics_history if m.get("cognitive_load") is not None]
cl_spark = _sparkline(cl_history, "#f59e0b")
cl_warn = (cl or 0) > 0.8
with k3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-head"><span class="kpi-dot" style="background:#f59e0b"></span>Cognitive Load</div>
            <div class="kpi-value">{cl if cl is not None else '—'}</div>
            <div class="kpi-sub">{'⚠ near threshold' if cl_warn else 'Exchange edges'}</div>
            <div class="kpi-delta {'warn' if cl_warn else 'up'}">
                {'▲ near threshold' if cl_warn else 'avg path length'}
            </div>
            {cl_spark}
        </div>
        """,
        unsafe_allow_html=True,
    )

# Learning Rate
lr = m_now["learning_rate"]
lr_history = [m.get("learning_rate", 0) for m in state.metrics_history if m.get("learning_rate") is not None]
lr_spark = _sparkline(lr_history, "#22c55e")
with k4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-head"><span class="kpi-dot" style="background:#22c55e"></span>Learning Rate</div>
            <div class="kpi-value">{lr if lr is not None else '—'}</div>
            <div class="kpi-sub">Market Stress</div>
            <div class="kpi-delta up">since last crisis</div>
            {lr_spark}
        </div>
        """,
        unsafe_allow_html=True,
    )

# Repr. Efficiency
re_val = m_now["repr_efficiency"]
re_history = [m.get("repr_efficiency", 0) for m in state.metrics_history if m.get("repr_efficiency") is not None]
re_spark = _sparkline(re_history, "#3b82f6")
with k5:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-head"><span class="kpi-dot" style="background:#3b82f6"></span>Repr. Efficiency</div>
            <div class="kpi-value">{re_val:.1%}</div>
            <div class="kpi-sub">Risk Exposure</div>
            <div class="kpi-delta up">mediator volume</div>
            {re_spark}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────
# Main charts row: Capital Evolution | Heatmap
# ──────────────────────────────────────────
# ── Main 3-col grid: Capital Evolution | Heatmap | Intelligence Trend ──
c1, c2, c3 = st.columns([1.1, 1, 1])

# Dark theme color palette for matplotlib
DARK_BG = "#171a21"
DARK_FG = "#dbe2ee"
DARK_MUTED = "#8892a6"

with c1:
    st.markdown(
        '<div class="panel"><h3>Capital Evolution <span class="hint">危机标记</span></h3>',
        unsafe_allow_html=True,
    )
    fig, ax = plt.subplots(figsize=(4.0, 2.5), dpi=110)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    colors = ["#22d3ee", "#f59e0b"]
    for i, (name, caps) in enumerate(state.capital_history.items()):
        ax.plot(
            range(len(caps)), caps, label=name,
            linewidth=1.8, color=colors[i % len(colors)],
        )
    for tick_num, _edges in state.crisis_marks:
        if tick_num < len(list(state.capital_history.values())[0]):
            ax.axvline(x=tick_num, color="#ef4444", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Tick", fontsize=8, color=DARK_MUTED)
    ax.set_ylabel("Capital", fontsize=8, color=DARK_MUTED)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.6,
              facecolor=DARK_BG, edgecolor=DARK_MUTED, labelcolor=DARK_FG)
    ax.grid(True, alpha=0.15, color=DARK_MUTED)
    ax.tick_params(labelsize=7, colors=DARK_MUTED)
    for spine in ax.spines.values():
        spine.set_color(DARK_MUTED)
        spine.set_alpha(0.3)
    fig.tight_layout()
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown(
        '<div class="panel"><h3>Exchange Rate <span class="hint">Rate / Volume / Deviation</span></h3>',
        unsafe_allow_html=True,
    )
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
    fig_h, ax_h = plt.subplots(figsize=(3.6, 2.5), dpi=110)
    fig_h.patch.set_facecolor(DARK_BG)
    ax_h.set_facecolor(DARK_BG)
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
                    ax_h.text(j, i, "—", ha="center", va="center", color=DARK_MUTED, fontsize=7)
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
        fig_h.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04)
    else:
        ax_h.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_h.transAxes, color=DARK_MUTED)
    ax_h.set_xticks(range(n))
    ax_h.set_xticklabels([c[:4] for c in commodities], fontsize=6, color=DARK_MUTED, rotation=45)
    ax_h.set_yticks(range(n))
    ax_h.set_yticklabels([c[:4] for c in commodities], fontsize=6, color=DARK_MUTED)
    for spine in ax_h.spines.values():
        spine.set_color(DARK_MUTED)
        spine.set_alpha(0.3)
    fig_h.tight_layout()
    st.pyplot(fig_h)
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown(
        '<div class="panel"><h3>Intelligence Metrics Trend <span class="hint">4 指标</span></h3>',
        unsafe_allow_html=True,
    )
    hist = state.metrics_history
    if len(hist) < 2:
        st.info("Run more steps to see metrics trend…")
    else:
        df_m = pd.DataFrame(hist).astype(float)
        fig_m, ax_m = plt.subplots(figsize=(3.8, 2.5), dpi=110)
        fig_m.patch.set_facecolor(DARK_BG)
        ax_m.set_facecolor(DARK_BG)
        for col, color, label in [
            ("phi", "#a78bfa", "Φ Integration"),
            ("cognitive_load", "#22d3ee", "Cognitive Load"),
            ("learning_rate", "#f59e0b", "Learning Rate"),
            ("repr_efficiency", "#22c55e", "Repr. Eff"),
        ]:
            if col in df_m.columns:
                vals = df_m[col].dropna()
                if len(vals) > 0:
                    ax_m.plot(
                        df_m["tick"], df_m[col], label=label,
                        linewidth=1.5, color=color,
                    )
        ax_m.set_xlabel("Tick", fontsize=8, color=DARK_MUTED)
        ax_m.legend(fontsize=7, loc="upper left", framealpha=0.6,
                    facecolor=DARK_BG, edgecolor=DARK_MUTED, labelcolor=DARK_FG)
        ax_m.grid(True, alpha=0.15, color=DARK_MUTED)
        ax_m.tick_params(labelsize=7, colors=DARK_MUTED)
        for spine in ax_m.spines.values():
            spine.set_color(DARK_MUTED)
            spine.set_alpha(0.3)
        fig_m.tight_layout()
        st.pyplot(fig_m)
    st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────
# Lower 2-col row (1:1): Edge State | Event Log
# ──────────────────────────────────────────
es_col, ev_col = st.columns([1, 1])

with es_col:
    st.markdown(
        '<div class="panel"><h3>Edge State <span class="hint">点击列头排序</span></h3>',
        unsafe_allow_html=True,
    )
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
        height=260,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with ev_col:
    st.markdown(
        '<div class="panel"><h3>Event Log <span class="hint">🔴 危机 · 🟢 交易 · 🔵 政策</span></h3>',
        unsafe_allow_html=True,
    )
    # Build a DataFrame so Event Log uses the same widget & height as Edge State,
    # giving the two panels a consistent visual treatment.
    if state.logs:
        log_entries = state.logs[-50:]  # keep up to 50 most recent
        log_df_rows = []
        for entry in reversed(log_entries):
            if "CRISIS" in entry:
                kind, marker = "Crisis", "🔴"
            elif (
                "Inject" in entry
                or "Drain" in entry
                or "Repaired" in entry
                or "[CB]" in entry
            ):
                kind, marker = "Policy", "🔵"
            elif "arb" in entry:
                kind, marker = "Trade", "🟢"
            else:
                kind, marker = "System", "⚪"
            log_df_rows.append({"Kind": f"{marker} {kind}", "Event": entry})
        st.dataframe(
            pd.DataFrame(log_df_rows),
            use_container_width=True,
            hide_index=True,
            height=260,
        )
    else:
        st.dataframe(
            pd.DataFrame([{"Kind": "—", "Event": "No events yet — press Step to begin."}]),
            use_container_width=True,
            hide_index=True,
            height=260,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer: export + about ──
st.markdown("---")
f1, f2, f3 = st.columns(3)
with f1:
    hist_df = pd.DataFrame(state.capital_history)
    hist_df.index.name = "tick"
    csv_cap = hist_df.to_csv().encode("utf-8")
    st.download_button(
        "📥 Capital CSV",
        data=csv_cap,
        file_name=f"capital_t{state.tick}.csv",
        mime="text/csv",
        use_container_width=True,
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
            use_container_width=True,
        )

with f2:
    share_params = {
        "crisis_prob": params["crisis_prob"],
        "profit_threshold": params["profit_threshold"],
        "recovery_rate": params["recovery_rate"],
        "trading_tax": params["trading_tax"],
    }
    encoded = urllib.parse.urlencode({k: str(v) for k, v in share_params.items()})
    st.code(f"?{encoded}", language="text")
    if st.button("📋 Copy share link", use_container_width=True):
        full_url = f"https://emoji-market-dashboard.streamlit.app/?{encoded}"
        st.toast(f"Link: {full_url}", icon="🔗")

with f3:
    st.markdown(
        """**ℹ️ About**

Dashboard built from a design dialogue.
Market ↔ Intelligence isomorphism: nodes=neurons, edges=synapses,
rates=weights, arbitrage=signal propagation."""
    )

# Footer
st.markdown(
    """
    <div class="footer-row">
        📖 <a href="https://math4mad.github.io/Emoji-Market-Tutorial/" target="_blank">Tutorial Website</a>
        · 💻 <a href="https://github.com/math4mad/Emoji-Market-Dashboard" target="_blank">Source Code</a>
        · 🌐 <a href="https://emoji-market-dashboard.streamlit.app" target="_blank">Live App</a>
    </div>
    """,
    unsafe_allow_html=True,
)
