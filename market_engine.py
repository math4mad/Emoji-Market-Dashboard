"""Market simulation engine for Emoji Market Dashboard."""

import math
import random
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional

COMMODITIES = [
    ("Sheep", "🐑"),
    ("Wool", "🧶"),
    ("Wheat", "🌾"),
    ("Iron", "🔨"),
    ("Wine", "🍷"),
    ("Energy", "⚡"),
]

BASE_RATES = [
    ("Sheep", "Wool", 5.0),
    ("Wool", "Wheat", 2.0),
    ("Wheat", "Iron", 0.05),
    ("Iron", "Wine", 0.5),
    ("Wine", "Sheep", 10.0),
    ("Wool", "Sheep", 0.2),
    ("Wheat", "Wool", 0.5),
    ("Iron", "Wheat", 20.0),
    ("Wine", "Iron", 2.0),
    ("Sheep", "Wine", 0.1),
    ("Energy", "Sheep", 20.0),
    ("Sheep", "Energy", 0.05),
    ("Energy", "Wine", 200.0),
    ("Wine", "Energy", 0.005),
    ("Energy", "Iron", 40.0),
    ("Iron", "Energy", 0.025),
]

MEDIATOR_NODES = {"Wool", "Wheat", "Iron"}


@dataclass
class Agent:
    name: str
    comm: str
    capital: float = 100.0
    target: str = "Wine"
    risk: float = 0.5
    volume_frac: float = 1.0
    trades: int = 0


class DynamicMarket:
    def __init__(self):
        self.G: nx.DiGraph = nx.DiGraph()
        self.base_rates: dict = {}
        self._init_graph()

    def _init_graph(self):
        for name, emoji in COMMODITIES:
            self.G.add_node(name, emoji=emoji)
        for base, target, rate in BASE_RATES:
            self.add_exchange_rate(base, target, rate)

    def add_exchange_rate(self, base, target, rate, loc="Global"):
        safe_rate = max(rate, 1e-10)
        weight = min(-math.log(safe_rate), 20.0)
        self.G.add_edge(base, target, weight=weight, rate=rate, loc=loc, volume=0.0)
        self.base_rates[(base, target)] = rate

    def execute_trade(self, path, amount):
        total_rate = 1.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge = self.G[u][v]
            impact = amount * 0.0001
            new_rate = max(edge["rate"] * (1 - impact), 0.01)
            edge["rate"] = new_rate
            edge["volume"] += amount
            edge["weight"] = min(-math.log(max(new_rate, 1e-10)), 20.0)
            total_rate *= new_rate
        return total_rate

    def market_recovery(self, recovery_rate=0.1):
        for u, v, data in self.G.edges(data=True):
            base = self.base_rates.get((u, v))
            if base and abs(data["rate"] - base) > 1e-9:
                diff = base - data["rate"]
                data["rate"] += diff * recovery_rate
                data["weight"] = min(-math.log(max(data["rate"], 1e-10)), 20.0)

    def inject_liquidity(self, agent, amount):
        agent.capital += amount

    def drain_liquidity(self, agent, amount):
        agent.capital = max(agent.capital - amount, 0)

    def reset_edge(self, edge_pair):
        u, v = edge_pair
        if self.G.has_edge(u, v):
            base = self.base_rates.get((u, v))
            if base:
                self.G[u][v]["rate"] = base
                self.G[u][v]["weight"] = min(-math.log(max(base, 1e-10)), 20.0)
                return True
        return False


@dataclass
class SimState:
    market: DynamicMarket = field(default_factory=DynamicMarket)
    agents: list = field(default_factory=list)
    tick: int = 0
    capital_history: dict = field(default_factory=dict)
    crisis_marks: list = field(default_factory=list)
    metrics_history: list = field(default_factory=list)
    logs: list = field(default_factory=list)

    def init_agents(self, configs=None):
        configs = configs or {}
        self.agents = [
            Agent(
                name="Alpha-Bot",
                comm="Sheep",
                capital=configs.get("Alpha-Bot", {}).get("capital", 100),
                target=configs.get("Alpha-Bot", {}).get("target", "Wine"),
                risk=configs.get("Alpha-Bot", {}).get("risk", 0.3),
                volume_frac=configs.get("Alpha-Bot", {}).get("volume_frac", 0.8),
            ),
            Agent(
                name="Beta-Bot",
                comm="Sheep",
                capital=configs.get("Beta-Bot", {}).get("capital", 100),
                target=configs.get("Beta-Bot", {}).get("target", "Wine"),
                risk=configs.get("Beta-Bot", {}).get("risk", 0.7),
                volume_frac=configs.get("Beta-Bot", {}).get("volume_frac", 1.0),
            ),
        ]
        self.capital_history = {a.name: [a.capital] for a in self.agents}


def step(state, params):
    """Advance simulation by one tick. Returns list of event strings."""
    events: list = []
    market = state.market

    # --- Crisis ---
    if random.random() < params["crisis_prob"]:
        edges = list(market.G.edges())
        if edges:
            u, v = random.choice(edges)
            market.G[u][v]["rate"] = max(market.G[u][v]["rate"] * 0.5, 1e-10)
            market.G[u][v]["weight"] = min(
                -math.log(max(market.G[u][v]["rate"], 1e-10)), 20.0
            )
            events.append(f"⚠️ CRISIS: {u}→{v} rate halved")

    # --- Agent evaluation & execution ---
    planned = []
    for agent in state.agents:
        agent.target = params["agent_targets"].get(agent.name, agent.target)
        agent.risk = params["agent_risks"].get(agent.name, agent.risk)
        agent.volume_frac = params["agent_volumes"].get(
            agent.name, agent.volume_frac
        )
        eff_threshold = 1 + (params["profit_threshold"] - 1) * (1 - 0.5 * agent.risk)
        try:
            path = nx.shortest_path(
                market.G, agent.comm, agent.target, weight="weight"
            )
            rate = 1.0
            for i in range(len(path) - 1):
                rate *= market.G[path[i]][path[i + 1]]["rate"]
            if market.G.has_edge(agent.target, agent.comm):
                cycle_rate = rate * market.G[agent.target][agent.comm]["rate"]
                full_path = list(path) + [agent.comm]
            else:
                cycle_rate = rate
                full_path = list(path)
            if cycle_rate > eff_threshold:
                planned.append((agent, full_path, cycle_rate))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

    for agent, full_path, cycle_rate in planned:
        trade_amount = agent.capital * agent.volume_frac
        market.execute_trade(full_path, trade_amount)
        gross_gain = agent.volume_frac * (cycle_rate - 1)
        tax = gross_gain * params["trading_tax"] if gross_gain > 0 else 0.0
        net_gain = gross_gain - tax
        agent.capital *= 1 + net_gain
        agent.trades += 1
        events.append(
            f"🤖 {agent.name} arb {agent.comm}→{agent.target} | "
            f"cycle={cycle_rate:.3f} | capital={agent.capital:.1f}"
        )

    # --- Crisis mark ---
    if any("CRISIS" in e for e in events):
        crisis_info = [
            e.split(": ")[1].split(" ")[0] for e in events if "CRISIS" in e
        ]
        state.crisis_marks.append((state.tick, crisis_info))

    # --- Market recovery ---
    market.market_recovery(params["recovery_rate"])

    # --- Record history ---
    for agent in state.agents:
        state.capital_history[agent.name].append(agent.capital)

    # --- Intelligence metrics ---
    m = compute_intelligence_metrics(market, state.tick, state.crisis_marks)
    state.metrics_history.append(
        {
            "tick": state.tick,
            **{k: float("nan") if v is None else float(v) for k, v in m.items()},
        }
    )

    state.tick += 1
    status = " | ".join(f"{a.name}: {a.capital:.1f}" for a in state.agents)
    events.append(f"📊 Tick {state.tick} | {status}")
    return events


def compute_intelligence_metrics(market, current_tick, crisis_marks=None):
    """Four neuroscience-inspired metrics (from design dialogue)."""
    G = market.G
    metrics: dict = {}

    # 1. Integration (Φ)
    if nx.number_of_nodes(G) > 1:
        try:
            base_reach = sum(
                len(paths) for _, paths in nx.all_pairs_shortest_path(G)
            )
        except Exception:
            base_reach = 0
        after = []
        for node in list(G.nodes()):
            G_copy = G.copy()
            G_copy.remove_node(node)
            try:
                after.append(
                    sum(len(paths) for _, paths in nx.all_pairs_shortest_path(G_copy))
                )
            except Exception:
                after.append(0)
        avg_reach = float(np.mean(after)) if after else 0
        metrics["phi"] = round(1.0 - avg_reach / max(base_reach, 1), 3)
    else:
        metrics["phi"] = 0.0

    # 2. Cognitive Load — use undirected version for path length
    try:
        G_und = G.to_undirected()
        if nx.is_connected(G_und):
            avg_path = nx.average_shortest_path_length(G_und)
            metrics["cognitive_load"] = round(min(float(avg_path) / 5.0, 10.0), 2)
        else:
            metrics["cognitive_load"] = None
    except (nx.NetworkXError, Exception):
        metrics["cognitive_load"] = None

    # 3. Learning Rate
    if crisis_marks and current_tick > crisis_marks[-1][0]:
        ticks_since = current_tick - crisis_marks[-1][0]
        metrics["learning_rate"] = round(10.0 / max(ticks_since, 1), 2)
    else:
        metrics["learning_rate"] = None

    # 4. Representation Efficiency
    total_vol = sum(d.get("volume", 0) for _, _, d in G.edges(data=True))
    mediator_vol = sum(
        d.get("volume", 0)
        for u, v, d in G.edges(data=True)
        if u in MEDIATOR_NODES or v in MEDIATOR_NODES
    )
    metrics["repr_efficiency"] = round(mediator_vol / max(total_vol, 1), 3)

    return metrics
