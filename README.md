# Emoji Market Dashboard 🌍

> 系统智能仪表盘 — A beautiful Streamlit dashboard for the Emoji Market simulation.

## Live Demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features

- **KPI Cards** — Total Capital, Φ Integration, Cognitive Load, Learning Rate, Representation Efficiency
- **Capital Evolution Chart** — Agent capital over time with crisis markers
- **Exchange Rate Heatmap** — Rate / Volume / Deviation-from-base modes
- **Market Network Graph** — Visualize commodity exchange graph with crisis edge coloring
- **Intelligence Metrics Trend** — Track four neuroscience-inspired metrics over time
- **Edge State Table** — Live rates, deviations, and volumes per edge
- **Event Log** — Color-coded simulation events (crisis, trades, policy)
- **Policy Sandbox** — Central bank: inject/drain liquidity, trading tax, targeted edge repair
- **Preset Scenarios** — Highland Crisis, AI War, Stable Equilibrium
- **CSV Export** — Download capital history and intelligence metrics
- **Scenario Sharing** — URL-encoded parameter strings

## Intelligence Metrics

| Metric | Formula | Meaning |
|---|---|---|
| Φ Integration | $1 - \text{avg reach after node removal} / \text{base reach}$ | System irreducibility |
| Cognitive Load | $\text{avg path length} / 5$ | Exchange chain complexity |
| Learning Rate | $10 / \text{ticks since last crisis}$ | Recovery speed |
| Repr. Efficiency | $\text{mediator volume} / \text{total volume}$ | Abstraction maturity |

## Design

The dashboard follows a clean, modern layout:
- Top header bar with gradient branding
- KPI card row with colored top borders and hover effects
- Left sidebar for all controls (simulation params, agent config, policy, presets)
- Main area: charts in white panels with soft shadows

## Tech Stack

| Component | Technology |
|---|---|
| Web Framework | Streamlit |
| Graph Algorithms | NetworkX |
| Visualization | Matplotlib |
| Data Handling | NumPy, Pandas |

## Project Structure

```
app.py                  # Main Streamlit dashboard
market_engine.py       # Market simulation engine (DynamicMarket, Agent, SimState)
.streamlit/config.toml # Theme configuration
requirements.txt        # Python dependencies
```

## Companion Project

- **Tutorial Website**: [Emoji-Market-Tutorial](https://github.com/math4mad/Emoji-Market-Tutorial)
- **Original Product**: [Emoji-Market-qwen](https://github.com/math4mad/Emoji-Market-qwen)

## License

MIT
