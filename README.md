# FIU ChainTrace

Single-page AML investigation dashboard: unified datasets, wallet graph, pattern engine, optional XGBoost, live chain polling, OFAC screening, and RSS intelligence. The center view is a **Three.js** globe embedded in Streamlit (`st.components.v1.html`).

## Run

```bash
cd FIU_Detection
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard.py
```

## Environment

| Variable | Purpose |
|----------|---------|
| `ETHERSCAN_API_KEY` | Free key from [etherscan.io](https://etherscan.io/apis) for Ethereum live samples |
| `FIU_BTC_WATCH_ADDR` | Optional Bitcoin address for `blockchain.info` sample (default: genesis) |

Streamlit Cloud: add the same keys under **Secrets** as `ETHERSCAN_API_KEY`.

## Data layout

Place optional CSV files under `data/`:

| Source | Files |
|--------|--------|
| Elliptic (primary) | `elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv` |
| Ethereum fraud (Kaggle) | `ethereum_fraud.csv` (or names in `data_loader.ETH_FRAUD_FILES`) |
| IBM AML (Kaggle) | `ibm_aml.csv` (or `IBM_AML_FILES`) |
| Panama Papers | `panama_papers_edges.csv` (or `PANAMA_FILES`) |

**Demo mode** (default): if Elliptic files are missing, the app uses an internal synthetic graph so the UI works immediately.

**Live mode**: turn off demo mode and enable **Live mode**; requires network access. APIs are polled every ~60s; failures fall back silently to dataset-only behavior.

## OFAC and watch list

- SDN list is downloaded to `data/ofac_sdn_download.csv` on first successful fetch (crypto-like strings extracted from text).
- Extra addresses: `known_bad_addresses.csv` in the project root.

## Modules

| File | Role |
|------|------|
| `dashboard.py` | Layout, globe HTML, Streamlit wiring |
| `data_loader.py` | Normalize all sources to `tx_id, from_address, to_address, amount, timestamp, chain, label` |
| `graph_builder.py` | `networkx.DiGraph` of wallets |
| `pattern_detector.py` | Rules + network clustering for globe |
| `ml_model.py` | Optional XGBoost |
| `live_api.py` | Bitcoin / Ethereum / Tron polling |
| `rss_monitor.py` | FinCEN, FATF, OFAC RSS |

## Notes for judges

- Elliptic is transaction-native; this build maps edges to **synthetic `bc1sim…` addresses** so wallet-centric rules apply consistently.
- Globe selection for the right-hand panel is driven from the **left list**; the iframe shows tooltips and arcs on click inside the canvas.
- Full **100vh / no scroll** is approximated with CSS (`overflow: hidden` on the app); Streamlit still injects its own DOM, so very small viewports may need minor zoom adjustments.
