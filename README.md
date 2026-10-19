# Shared Temporal Models and Cross-Link Graphs for LoRaWAN RSSI Forecasting

Companion code for the paper by Lorenzo Moreira Donatti and Carlos Henrique
Barriquello, Federal University of Santa Maria (UFSM).

**Publication status:** the manuscript has been submitted to *Internet of Things*
(Elsevier). It has not yet been accepted for publication.

This repository reproduces next-hour mean RSSI forecasting for nine links at
UVA gateway A and six links at gateway C. It contains **four Python files**, prepared
hourly data, frozen paper configurations, and numerical reference results.
No manuscript sources or figure-generation scripts are required.

## Citation

If you use this code or its results in your research, please acknowledge this
repository. If the manuscript is accepted, please also cite the article using
its final publication details and DOI when available. This section and
[`CITATION.cff`](CITATION.cff) will be updated with the published reference.

The submitted manuscript is:

> Lorenzo Moreira Donatti and Carlos Henrique Barriquello.
> *Shared Temporal Models and Cross-Link Graphs for LoRaWAN RSSI Forecasting*.
> Manuscript submitted to *Internet of Things* (Elsevier).

When using the UVA data, please also cite the original dataset listed under
[Data and reference provenance](#data-and-reference-provenance).

## Repository layout

```text
run.py                    Command-line entry point and experiment execution
models.py                 Statistical, recurrent, linear and graph models
data.py                   Hourly windows, scaling, topology and configuration loading
evaluation.py             Per-link metrics, macro summaries and paired bootstrap
requirements.txt
configs/
  A.csv, C.csv             Hyperparameters selected for each gateway
data/
  A/, C/                   hourly.csv and topology.csv
reference/
  A/, C/                   per_link.csv, macro.csv, paired.csv,
                           predictions.csv.gz and environment.txt
```

`predictions.csv.gz` is an ordinary compressed CSV, not a model checkpoint.
It contains target time, forecast origin, link, observed RSSI and all 12 predictions
in dBm. Compression keeps the repository small while allowing numerical results
to be recalculated without fitting models. There are no distributed JSON or NPZ files.

## Installation

Python 3.12 and Linux are recommended. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py check
```

The check validates topology order, configuration coverage and complete-window
counts:

| Gateway | Links | Train | Validation | Test |
|---|---:|---:|---:|---:|
| A | 9 | 7145 | 1369 | 1015 |
| C | 6 | 4967 | 1052 | 1161 |

## Recalculate the manuscript's numerical results

```bash
python run.py metrics
```

This reads the archived CSV predictions and writes `per_link.csv`, `macro.csv`
and `paired.csv` to `outputs/metrics/A/` and `outputs/metrics/C/`. It verifies the
recomputed values against the reference CSVs. Archived per-link metrics have four
decimal places, so their comparison allows rounding; macros and paired effects
are checked at full precision with a numerical tolerance of 1e-12.

## Refit the paper models

```bash
python run.py train --gateway A
python run.py train --gateway C
```

Or run both tasks sequentially with `python run.py train`.

Training uses the selected configurations in `configs/`, CPU with six threads,
seed 42, a 24-hour history, one-hour horizon, 64-epoch cap and early stopping with
patience ten. Each graph ablation inherits the hybrid configuration and is fitted
from scratch. Missing configurations cause an error rather than a fallback to
defaults. Full training may take substantial time.

Outputs are written under `outputs/train/A/` and `outputs/train/C/`:

- `per_link.csv`: MAE and RMSE by link, in dB (the primary outcomes).
- `macro.csv`: unweighted means of link-level MAE and RMSE.
- `paired.csv`: candidate-minus-reference MAE differences and unadjusted 95% intervals.
- `configuration.csv` and `environment.txt`: fitting configuration and environment.
- `predictions.npz`: **one generated archive per gateway**, containing truth,
  every completed model prediction, link names, target timestamps and forecast-origin
  timestamps. This preserves the ability to calculate new metrics without refitting.

To recalculate metrics from a new completed training run:

```bash
python run.py metrics --source outputs/train --output outputs/recomputed
```

Existing gateway output directories are never overwritten. Choose another
`--output` for a new run. Completed predictions are saved after each model, but
this compact implementation does not resume interrupted training automatically.
Input paths are relative to the script directory; an explicitly supplied relative
`--source` or `--output` is relative to the working directory.

## Models and protocol

| Code name | Paper interpretation |
|---|---|
| `ARIMAX` / `VARX` | Dedicated / joint statistical controls |
| `SingleNode_Seq2Seq` / `MultiNode_Seq2Seq` | Dedicated / joint attention Seq2Seq |
| `SingleNode_NLinear` / `SingleNode_DLinear` | Dedicated linear controls |
| `MultiNode_NLinear` / `MultiNode_DLinear` | Shared-weight, channel-independent linear controls |
| `PhysicalAdaptive_STGNN` | Hybrid STGNN |
| `STGNN_NoGraph` | Shared temporal control without cross-link propagation |
| `STGNN_PhysicalOnly` / `STGNN_AdaptiveOnly` | Geographic / learned adjacency |

`MultiNode` in the linear model names denotes parameter sharing, **not cross-channel
inputs**. All experiments retain chronological 70/15/15 partitions, training-only
scalers, historical weather, strictly hourly complete windows, and no imputation.
A and C are separate tasks, not independent deployments. Gateway B is excluded
from the paper's model comparisons because only two links satisfy eligibility.

The macro-RMSE is the mean of link-level RMSEs, not pooled RMSE. The four paired
comparisons use 2,000 circular block-bootstrap replicates, 24-hour blocks, seed
2026, and resampling within contiguous observed runs. Negative differences favor
the candidate. Block length can be changed through `block_intervals()` in
`evaluation.py` when conducting a separate sensitivity analysis.

## Data and reference provenance

Prepared hourly inputs derive from:

Fateme Nikseresht, Victor Ariel Leal Sobral, Jonathan L. Goodall and Bradford
Campbell (2025). *Long-term LoRaWAN Communication Metadata from an Urban
Deployment*. University of Virginia Dataverse.
[DOI: 10.18130/V3/RFTICK](https://doi.org/10.18130/V3/RFTICK).
The dataset is distributed under **CC BY 4.0**; retain this attribution when
redistributing its prepared derivatives. Sensor selection uses at least 70%
hourly RSSI availability in both training and validation, never test availability.

The A reference is from `optimized_seed42_64ep_final_v2`; C is from
`optimized_seed42_64ep_final`. Their recorded environments are preserved in
`reference/A/environment.txt` and `reference/C/environment.txt`. Reference predictions
were converted losslessly from the original archives to compressed CSV.

This repository starts from the prepared hourly data and refits the frozen paper
configurations. It does **not** repeat raw-packet preprocessing or hyperparameter
search. The requirements specify compatible minimum versions rather than an exact
historical lockfile. Retraining under different library versions or hardware may
change numerical results. Archived-prediction recalculation reproduces the reported
results independently of training variation.

The model and window-processing code was consolidated from the original benchmark.
Validation covered input-array equality, short neural training comparisons and
recalculation of all reference metrics and paired intervals. Full paper training
was not repeated while preparing this repository.
