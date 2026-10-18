# Causally anchored price-aware forecasting of urban electric vehicle charging demand

Code, configurations and result files for the manuscript

> S. Zhang. *Causally anchored price-aware forecasting of urban electric vehicle charging demand.* Manuscript submitted to *Applied Energy*, 2026.

The repository reproduces every table and figure of the paper and of its Supplementary Material from the public UrbanEV dataset: the checks of the theory, the causal estimation of the price elasticity from price-switch events, the proposed forecaster **CPA-STGNN** with adaptive anchoring, the eighteen baselines, the ablations, the semi-synthetic counterfactual benchmark and the price adjustment scenarios. The summary results of all reported experiments are included, so the tables and figures can be regenerated without retraining.

## Method in brief

When charging prices follow a fixed time-of-use (TOU) schedule, the forecasting loss cannot tell how much of the daily demand pattern is caused by the price (Theorem 1 of the paper). CPA-STGNN therefore identifies the price response outside the forecaster and inserts it as a fixed quantity:

1. **Causal estimation** (`src/causal/`, `scripts/estimate_causal_5min.py`, `scripts/causal_round2.py`): the short-run own-price elasticity is estimated from price switches in five-minute station data with a within-station design (design D: date-by-slot and station-by-slot fixed effects, Poisson pseudo-maximum likelihood). The anchor is −0.21, with a zone-clustered standard error of 0.20 and a zone-bootstrap standard error of 0.31.
2. **Price-blind forecaster** (`src/models/backbones/graph_wavenet.py`, `src/models/heads/quantile.py`): a Graph WaveNet encoder with an adaptive graph and a non-crossing quantile head predicts baseline quantiles without seeing any price.
3. **Anchored price response** (`src/models/heads/price.py`, `src/models/cpastgnn.py`): the baseline is multiplied by `exp(beta * log-price deviation)` with the anchor as a constant that receives no gradient (cut feedback), so counterfactual forecasts under other tariffs follow in closed form. The 90% interval is calibrated online by adaptive conformal inference (`src/conformal/aci.py`); counterfactual intervals combine it with the confidence interval of the elasticity (Theorem 3).
4. **Adaptive anchoring** (Theorem 2 and Eq. 10 of the paper): a second network learns a sign-constrained elasticity from zero, and the anchor and the learned elasticity are combined with the plug-in weight `w = e / (e + s_A^2)`, where `e = max((beta_L - beta_A)^2 - s_A^2, 0)` (`scripts/summarize_ess_v3.py` for the benchmark; Section 4.5 of the paper for the observed data).

## Repository structure

```
assets/            fixed zone attributes (zone_static.csv) and SHA-256 checksums of the UrbanEV files
configs/
  default.yaml     all settings with their defaults
  experiment/      final_main.yaml (main setting), ess.yaml (semi-synthetic benchmark), round2_best.json (tuned
                   settings of the baselines, written by select_round2.py), smoke.yaml and sens_lag.yaml (tests)
  anchor/          elasticity files; fivemin_main.json is the anchor of the paper
src/
  data/            loading, calendar, data-quality mask, price features, graphs, splits, windows
  causal/          switch-event designs on hourly and five-minute data, PPML, panel models, anchor files
  models/          CPA-STGNN, Graph WaveNet and STAEformer backbones, quantile and price heads, losses, gradient
                   boosting, reference methods; baselines/ (FCNN, LSTM, STID, DLinear, GCN, GCN-LSTM, STGCN,
                   ASTGCN, AGCRN, PAG, PIAST)
  conformal/       adaptive conformal inference with masks
  eval/            metrics, Diebold-Mariano tests, price-response read-out, semi-synthetic benchmark, counterfactuals
  simulate/        price adjustment scenarios
scripts/           command-line entry points (see "Reproducing the results")
tests/             unit tests
results/           summary results of every experiment in the paper (see "Included results")
figures/
  fig_overview.svg editable source of Fig. 1
  paper/           the figures of the manuscript and of the Supplementary Material (PDF)
  scripts/         plotting code
```

Code comments and log messages are partly written in Chinese; identifiers, configuration keys and file formats are in English.

## Installation

Python 3.10 or 3.11.

```bash
git clone https://github.com/zse04152005-del/ev-causal-forecast.git
cd ev-causal-forecast
pip install -r requirements.txt
```

For training on an NVIDIA GPU, install the CUDA build of PyTorch first (https://pytorch.org/get-started/locally/) and then `pip install -r requirements_windows_cuda.txt`. The experiments of the paper were run with PyTorch 2.11 (CUDA 12.8) on Windows with an NVIDIA RTX 4060 (8 GB); one training run of CPA-STGNN takes about 11 minutes. The theory checks, the causal analysis, the evaluation, the scenarios, the tables and the figures need only the CPU packages in `requirements.txt`.

## Data

The study uses **UrbanEV** (Li et al., *Scientific Data* 12:523, 2025, https://doi.org/10.1038/s41597-025-04874-4), released under CC0 1.0. The data are not redistributed here; `data/` is excluded from version control.

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

All commands are run from the repository root. Each training run writes to `results/runs/<run name>/`. The batch scripts skip runs that already have a `summary.json`, so an interrupted batch is resumed by running the same command again.

### 1. Data description, theory checks and causal estimation (CPU)

```bash
python scripts/eda.py                                    # results/EDA/                          (Section 4.1, Fig. 5)
python scripts/theory_data.py                            # results/theory/t1_data_numbers.json   (Section 3.1, Table S1)
python scripts/theory_montecarlo.py                      # results/theory/mc_t1_t2.json          (Tables S2 and S3)
python scripts/estimate_causal_5min.py --stages all      # results/causal/5min/, configs/anchor/ (Table 4)
python scripts/causal_round2.py --stages all --reps 200  # results/causal/round2/: zone bootstrap, randomisation test,
                                                         #   direction, event study, identifying variation, periods
python scripts/causal_round2_influence.py                # influence of single zones on the bootstrap
python scripts/estimate_panel_supp.py                    # hourly panel PPML and double machine learning (Table 4)
python scripts/semisynthetic_5min.py --means 30,60 --seeds 3 --tag m30_60   # designs C and D on synthetic arrivals
python scripts/summarize_semisynth_5min.py               # results/causal/5min/semisynth_5min_*.csv (Section 4.5)
```

`results/causal/round2_w12/` and `round2_w24/` hold the 60 and 120 min half-windows of `causal_round2.py` (Table S4).

### 2. Forecasting experiments (GPU)

Single runs:

```bash
python scripts/train.py --config configs/experiment/final_main.yaml train.seed=0                       # CPA-STGNN
python scripts/train_baseline.py --config configs/experiment/final_main.yaml baseline.name=stid train.seed=0
python scripts/run_baseline.py model.name=profile          # reference methods (naive_last, naive_seasonal, profile), gbdt
```

The runs of the paper are scripted in four PowerShell files (`-Phase` takes a comma-separated list):

| Script and phases | Runs | Used in |
|---|---|---|
| `run_final.ps1 -Phase tune` | tuning grid of CPA-STGNN (12 configurations, seed 0) | Section 4.2, `runs_tune.csv` |
| `run_final.ps1 -Phase baselines` | STGCN and AGCRN with price inputs (`base_*_price`, 5 seeds) | Table 2 |
| `run_final.ps1 -Phase ablations,eps` | predefined graphs (`main_A0`, `abl_A11_predefined`); prior sensitivity of PAG and PIAST (`eps_*`) | Table 3, Fig. 7b |
| `run_final_v2.ps1 -Phase main,ablations,eid,stae` | CPA-STGNN and Graph WaveNet, seeds 0-4 (`v2_main_A0`, `v2_gwn_blind`); ablations (`v2_abl_*`); loss profile (`v2_eid_profile*`) | Tables 2 and 3, Fig. 7a |
| `run_round2_baselines.ps1 -Phase tune`, then `python scripts/select_round2.py` | tuning grid of every deep baseline (12 configurations, 4 for STAEformer, seed 0) | Section 4.2, `round2_tuning.csv`, `configs/experiment/round2_best.json` |
| `run_round2_baselines.ps1 -Phase seeds,pag,t2,gbdt` | baselines with ten seeds (`r2_*`), seeds 5-9 of CPA-STGNN and Graph WaveNet, model without a graph, PAG on utilisation, sign-constrained elasticity learned from zero (`r2_proj0`), gradient boosting | Tables 2 and 3, Section 4.5 |
| `run_round2_baselines.ps1 -Phase dm` | loss series for the Diebold-Mariano tests (`results/comparisons/dm/loss_r2.npz`) | Tables 2 and 3 |
| `run_ess_v3.ps1 -Phase main,oracle,endo` | semi-synthetic benchmark: 2 variation settings x 2 true elasticities x 5 alignments x 5 models x 5 seeds, anchors at the true value, endogenous deviations (570 runs) | Section 4.6, Fig. 8, Table S5 |

### 3. Summaries, tests and scenarios (CPU)

```bash
python scripts/summarize_all.py --runs-dir results/runs             # all_runs_flat.csv, groups_mean.csv, eid/eps summaries
python scripts/collect_by_step.py --runs results/runs --out results/comparisons   # metrics_by_step.csv, conformal_by_step.csv
python scripts/learned_elasticity.py --out results/comparisons/learned_elasticity.csv results/runs/r2_proj0_s* \
       results/runs/v2_abl_A1_free_s* results/runs/v2_abl_A2_none_s* results/runs/v2_abl_A13_soft1_s*
python scripts/summarize_ess_v3.py                                  # ess_v3_runs.csv, ess_v3_summary.csv (with adaptive anchoring)
python scripts/merge_loss_series.py --first results/comparisons/dm/loss_a.npz \
       --second results/comparisons/dm/loss_r2.npz --out results/comparisons/dm/loss_r2m.npz v2_main_A0 v2_gwn_blind
python scripts/significance_tests.py --dir results/comparisons      # dm_wilcoxon.csv, dm_wilcoxon_overall.csv
python scripts/real_deviation_test.py --runs-dir results/runs   # realdev_slopes.csv: errors on real price departures (Table S6)
python scripts/simulate_scenarios.py --runs results/runs --out results/scenarios --se 0.3131                      # anchor
python scripts/simulate_scenarios.py --runs results/runs --out results/scenarios/adaptive --beta -0.343 --se 0.3131
python scripts/simulate_scenarios.py --runs results/runs --out results/scenarios/steady_state_mu60 --beta -0.3119 --se 0.4582
```

`loss_a.npz` holds the loss series of seeds 0-4 of CPA-STGNN and Graph WaveNet from the first batch; `merge_loss_series.py` combines them with seeds 5-9, so that the tests use all ten seeds.

### 4. Tables and figures (CPU; these steps use only `results/` and the UrbanEV data)

```bash
python scripts/make_paper_tables.py --out results/tables --causal-dir results/causal/5min      # Tables 1 and 4
python scripts/make_tables.py --dir results/comparisons --out results/tables                   # Table 2
python scripts/make_exp_tables.py --cmp results/comparisons --scen results/scenarios --out results/tables   # Table 3
python figures/scripts/fig_ess_v3.py                     # Fig. 8   -> figures/results/
python figures/scripts/fig_event_direction.py            # Fig. S1  -> figures/results/
cd figures/scripts
python fig4_modules.py       # Figs. 2-4            -> figures/modules/
python fig_results.py        # Figs. 5, 6, 7 and 9  -> figures/results/
node render_svg.cjs ../fig_overview.svg     # Fig. 1 (needs Node.js and Playwright)
```

**Unit tests**: `python -m unittest discover -s tests -t .` (tests that need PyTorch or the data are skipped when these are missing).

## Included results

| Paper | File(s) | Produced by |
|---|---|---|
| Table 1 (data) | `results/tables/tab_data.*` | `make_paper_tables.py` |
| Table 2 (comparison) | `results/tables/tab_main_results.*`; `results/comparisons/groups_mean.csv`, `dm_wilcoxon*.csv`, `dm/*.npz` | `make_tables.py`, `significance_tests.py` |
| Table 3 (ablation) | `results/tables/tab_ablation.*`; `results/comparisons/learned_elasticity.csv` | `make_exp_tables.py`, `learned_elasticity.py` |
| Table 4 (elasticity) | `results/tables/tab_causal.*`; `results/causal/5min/`, `results/causal/round2/summary.json` | `make_paper_tables.py`, `estimate_causal_5min.py`, `causal_round2.py` |
| Fig. 1 (overview) | `figures/fig_overview.svg`, `figures/paper/fig_overview.pdf` | vector source |
| Figs. 2-4 (modules) | `figures/paper/fig_rings.pdf`, `fig_encoder.pdf`, `fig_calibration.pdf` | `fig4_modules.py` |
| Fig. 5 (data) | `figures/paper/fig_context.pdf`; `results/EDA/` | `fig_results.py` |
| Fig. 6 (accuracy by horizon) | `figures/paper/fig_horizon.pdf`; `results/comparisons/metrics_by_step.csv`, `conformal_by_step.csv` | `fig_results.py` |
| Fig. 7 (loss profile, injected priors) | `figures/paper/fig_validation.pdf`; `results/comparisons/eid_summary.csv`, `eps_summary.csv` | `fig_results.py` |
| Fig. 8 (semi-synthetic benchmark) | `figures/paper/fig_ess.pdf`; `results/comparisons/ess_v3_runs.csv`, `ess_v3_summary.csv` | `fig_ess_v3.py` |
| Fig. 9 (price adjustment) | `figures/paper/fig_scenario.pdf`; `results/scenarios/` (with `adaptive/`, `steady_state_mu60/`) | `fig_results.py` |
| Table S1 (Theorem 1 on the data) | `results/theory/t1_data_numbers.json` | `theory_data.py` |
| Tables S2 and S3 (Monte Carlo) | `results/theory/mc_t1_t2.json` | `theory_montecarlo.py` |
| Table S4, Fig. S1 (robustness, event study) | `results/causal/round2*/`; `figures/paper/fig_S_event_direction.pdf` | `causal_round2.py`, `causal_round2_influence.py`, `fig_event_direction.py` |
| Table S5 (benchmark, all combinations) | `results/comparisons/ess_v3_summary.csv` | `summarize_ess_v3.py` |
| Table S6 (real price departures) | `results/comparisons/realdev_slopes.csv` | `real_deviation_test.py` |
| Tuning | `results/comparisons/runs_tune.csv` (CPA-STGNN), `round2_tuning.csv` (baselines) | `summarize_runs.py --prefix tune_`, `select_round2.py` |

`results/comparisons/all_runs_flat.csv` lists the test metrics of every individual run (model, seed, MAE, RMSE, CRPS, coverage). Per-run logs, checkpoints and point forecasts (about 24 MB per run) are not included because of their size; they are available from the author on request.

## License

The code in this repository is released under the MIT License (see `LICENSE`). Parts of `src/models/baselines/` are adapted from the official implementations of the baseline models, which keep their own licenses; see `THIRD_PARTY_NOTICES.md`. The UrbanEV data are not part of this repository and are subject to their own terms (CC0 1.0).

## Citation

If you use this code, please cite the paper (see `CITATION.cff`; the reference will be updated on publication).

## Contact

Shien Zhang, https://github.com/zse04152005-del
