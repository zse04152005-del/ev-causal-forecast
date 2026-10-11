# Causally anchored price-aware forecasting of urban electric vehicle charging demand

Code, configurations and result files for the manuscript

> S. Zhang. *Causally anchored price-aware forecasting of urban electric vehicle charging demand.* Manuscript submitted to *Applied Energy*, 2026.

The repository contains everything needed to reproduce the results of the paper from the public UrbanEV dataset: the causal estimation of the price elasticity from price-switch events, the proposed forecaster **CPA-STGNN**, the baselines, the semi-synthetic counterfactual benchmark, the price adjustment scenarios, and the scripts that produce every table and figure of the paper. The summary results of all reported experiments are included, so the tables and figures can be regenerated without retraining.

## Method in brief

When charging prices follow a fixed time-of-use schedule, a forecasting loss cannot tell how much of the daily demand pattern is caused by the price. CPA-STGNN therefore separates the two tasks:

1. **Causal estimation module** (`src/causal/`): the short-run own-price elasticity is estimated outside the network from price-switch events in five-minute station data, with a within-station design (station-by-slot and date-by-slot fixed effects, Poisson pseudo-maximum likelihood). The main estimate is −0.21 (s.e. 0.20).
2. **Price-blind spatio-temporal module** (`src/models/backbones/graph_wavenet.py`, `src/models/heads/`): a Graph WaveNet encoder with an adaptive graph and a non-crossing quantile head produces baseline quantiles without seeing any price.
3. **Anchored price response** (`src/models/heads/price.py`, `src/models/cpastgnn.py`): the estimated elasticity enters as a fixed quantity that receives no gradient (cut feedback), so that counterfactual forecasts under other tariffs follow in closed form; intervals are calibrated online by adaptive conformal inference (`src/conformal/aci.py`).

## Repository structure

```
assets/            fixed zone attributes (zone_static.csv) and SHA-256 checksums of the UrbanEV files
configs/           default.yaml; experiment/ (final_main.yaml = main setting, ess.yaml = benchmark, ...);
                   anchor/ (elasticity files; fivemin_main.json is the anchor used in the paper)
src/
  data/            loading, calendar, data-quality mask, price features, graphs, splits, windows
  causal/          switch-event designs on hourly and five-minute data, PPML, panel models, anchor files
  models/          CPA-STGNN, Graph WaveNet and STAEformer backbones, quantile and price heads, losses,
                   baselines/ (FCNN, LSTM, GCN, GCN-LSTM, STGCN, ASTGCN, AGCRN, PAG, PIAST)
  conformal/       adaptive conformal inference with masks
  eval/            metrics, Diebold-Mariano tests, price-response read-out, semi-synthetic benchmark
  simulate/        price adjustment scenarios
scripts/           command-line entry points (see "Reproducing the results")
tests/             unit tests (python -m unittest)
results/           summary results of all experiments reported in the paper (see "Included results")
figures/
  scripts/         plotting code for the figures of the paper
  fig_overview.svg editable source of Fig. 1
  paper/           the nine figures as they appear in the manuscript (PDF)
```

Code comments and log messages are partly written in Chinese; identifiers, configuration keys and file formats are in English.

## Installation

Python 3.10 or 3.11.

```bash
git clone https://github.com/zse04152005-del/ev-causal-forecast.git
cd ev-causal-forecast
pip install -r requirements.txt
```

For training on an NVIDIA GPU, install the CUDA build of PyTorch first (see https://pytorch.org/get-started/locally/) and then `pip install -r requirements_windows_cuda.txt`. The experiments of the paper were run with PyTorch 2.11 (CUDA 12.8) on Windows with an NVIDIA RTX 4060 (8 GB); one training run of CPA-STGNN takes about 11 minutes. The causal analysis, the evaluation, the scenarios, the tables and the figures need only the CPU packages in `requirements.txt` (PyTorch is not required for them).

## Data

The study uses **UrbanEV** (Li et al., *Scientific Data* 12:523, 2025, https://doi.org/10.1038/s41597-025-04874-4), released under CC0 1.0. The data are not redistributed here. All paths below are relative to the repository root; `data/` is excluded from version control.

| Data | Source | Location expected by the code |
|---|---|---|
| Hourly zone-level data (275 zones), weather, POI, zone boundaries | `git clone https://github.com/IntelligentSystemsLab/UrbanEV data/raw/urbanev_github` | `data/raw/urbanev_github/data/` |
| Five-minute station-level data (causal analysis only) | `UrbanEVDataset.zip` linked in the UrbanEV README (contains `station-processed/`) and `UrbanEVDataset.zip` on Dryad, https://doi.org/10.5061/dryad.np5hqc04z | converted to `data/interim/fiveMin/` by `scripts/convert_fivemin.py` |

```bash
python scripts/verify_data.py        # checks the hourly files against assets/urbanev_github_SHA256SUMS.txt
python scripts/convert_fivemin.py --station-zip data/raw/urbanev_station/UrbanEVDataset.zip \
                                  --dryad-zip   data/raw/urbanev_dryad/UrbanEVDataset.zip
```

## Reproducing the results

All commands are run from the repository root. Each training run writes to `results/runs/<run name>/`.

**1. Data description and causal estimation** (CPU, Section 3.2 and Section 4.5 of the paper)

```bash
python scripts/eda.py                                   # results/EDA/
python scripts/estimate_causal_5min.py --stages all     # results/causal/5min/, configs/anchor/fivemin_*.json
python scripts/semisynthetic_5min.py --means 30,60 --seeds 3 --tag m30_60
python scripts/summarize_semisynth_5min.py              # designs C and D on synthetic arrivals; arrival elasticity
python scripts/boot_cd_diff.py --reps 100               # bootstrap of the difference between designs C and D
python scripts/estimate_panel_supp.py                   # hourly panel PPML and double machine learning
```

**2. Forecasting experiments** (GPU, Sections 4.3, 4.4 and 4.6)

```bash
python scripts/train.py --config configs/experiment/final_main.yaml train.seed=0     # CPA-STGNN, one seed
python scripts/train_baseline.py baseline.name=agcrn                                  # one baseline
```

The complete batch of the paper (main model, baselines, ablations, loss profile, prior sensitivity of PAG and PIAST, semi-synthetic benchmark; 415 runs in total) is scripted in two PowerShell files; they skip runs that are already finished:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_final.ps1 -Phase pilot,tune,main,baselines,ablations,eid,eps,ess
powershell -ExecutionPolicy Bypass -File scripts\run_final_v2.ps1      # second round: adaptive graph only (main setting of the paper)
```

**3. Summaries, tests and scenarios** (CPU)

```bash
python scripts/summarize_all.py                  # results/comparisons/all_runs_flat.csv, groups_mean.csv, eid/eps summaries
python scripts/summarize_ess.py                  # results/comparisons/ess_summary*.csv
python scripts/collect_by_step.py --runs results/runs --out results/comparisons
python scripts/loss_series.py --runs results/runs --out results/comparisons/dm/loss_a.npz <groups>   # see script header
python scripts/significance_tests.py --dir results/comparisons
python scripts/simulate_scenarios.py --runs results/runs --out results/scenarios
```

**4. Tables and figures of the paper** (CPU; these steps use only the files in `results/` and the UrbanEV data)

```bash
python scripts/make_paper_tables.py --out results/tables --causal-dir results/causal/5min       # Tables 1 and 4
python scripts/make_tables.py --dir results/comparisons --out results/tables                    # Table 2
python scripts/make_exp_tables.py --cmp results/comparisons --scen results/scenarios --out results/tables   # Table 3
cd figures/scripts
python fig4_modules.py       # Figs. 2-4  -> figures/modules/
python fig_results.py        # Figs. 5-9  -> figures/results/
node render_svg.cjs ../fig_overview.svg     # Fig. 1 (needs Node.js and Playwright)
```

With the included results these commands reproduce the tables in `results/tables/` byte for byte.

**Unit tests**: `python -m unittest discover -s tests -t .` (tests that need PyTorch or the data are skipped when these are missing).

## Included results

| Paper | File(s) | Produced by |
|---|---|---|
| Table 1 (data) | `results/tables/tab_data.*` | `make_paper_tables.py` |
| Table 2 (comparison) | `results/tables/tab_main_results.*`; `results/comparisons/groups_mean.csv`, `dm_wilcoxon*.csv`, `dm/*.npz` | `make_tables.py`, `significance_tests.py` |
| Table 3 (ablation) | `results/tables/tab_ablation.*` | `make_exp_tables.py` |
| Table 4 (elasticity) | `results/tables/tab_causal.*`; `results/causal/5min/` | `make_paper_tables.py`, `estimate_causal_5min.py` |
| Fig. 1 (overview) | `figures/fig_overview.svg` | hand-drawn vector source |
| Figs. 2-4 (modules) | `figures/paper/fig_rings.pdf`, `fig_encoder.pdf`, `fig_calibration.pdf` | `fig4_modules.py` (panels c, a, d) |
| Fig. 5 (data context) | `figures/paper/fig_context.pdf` | `fig_results.py` (`Fig_R_context`) |
| Fig. 6 (accuracy by horizon) | `figures/paper/fig_horizon.pdf`; `results/comparisons/metrics_by_step.csv`, `conformal_by_step.csv` | `fig_results.py` (`Fig_R_horizon`) |
| Fig. 7 (identifiability) | `figures/paper/fig_validation.pdf`; `results/comparisons/eid_summary.csv`, `eps_summary.csv` | `fig_results.py` (`Fig_R_validation`) |
| Fig. 8 (semi-synthetic benchmark) | `figures/paper/fig_ess.pdf`; `results/comparisons/ess_summary*.csv` | `fig_results.py` (`Fig_R_ess`) |
| Fig. 9 (price adjustment) | `figures/paper/fig_scenario.pdf`; `results/scenarios/` | `fig_results.py` (`Fig_R_scenario`) |
| Hyperparameter tuning | `results/comparisons/runs_tune.csv` | `summarize_runs.py --prefix tune_` |
| Reproduction of the UrbanEV benchmark | `results/baselines/urbanev_repro/` | `repro_urbanev_table3.py` |

`results/comparisons/all_runs_flat.csv` lists the test metrics of every individual run (model, seed, MAE, RMSE, CRPS, coverage). Per-run logs and point forecasts (about 24 MB per run) are not included because of their size; they are available from the author on request.

## License

The code in this repository is released under the MIT License (see `LICENSE`). Parts of `src/models/baselines/` are adapted from the official implementations of the baseline models, which keep their own licenses; see `THIRD_PARTY_NOTICES.md`. The UrbanEV data are not part of this repository and are subject to their own terms (CC0 1.0).

## Citation

If you use this code, please cite the paper (see `CITATION.cff`; the reference will be updated on publication).

## Contact

Shien Zhang, https://github.com/zse04152005-del
