# ART-RAN: Exposing Adversarial Vulnerabilities in Agentic O-RAN rApps

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/IEEE%20ICC-2027-1f6feb.svg)](#citation)

**ART-RAN** stands for **Adversarial Reasoning Threats in Radio Access Networks**. It is a threat-analysis and reproducibility framework for studying how adversarial compromise of agentic Non-RT O-RAN rApps can propagate through downstream Near-RT control and affect RAN-level QoS.

> **Paper:** *ART-RAN: Exposing Adversarial Vulnerabilities in Agentic O-RAN rApps*  
> **Authors:** Yared Abera Ergu, Van-Linh Nguyen, Po-Ching Lin, and Ren-Hung Hwang  
> **Target venue:** IEEE ICC 2027

<p align="center">
  <img src="docs/figures/art-ran-architectu<img width="1376" height="451" alt="model-architecture" src="https://github.com/user-attachments/assets/a6361404-6f3c-4f09-8846-ac6a78c9c8b7" />
re.png" width="92%" alt="ART-RAN architecture">
</p>

> Add the final paper architecture figure as `docs/figures/art-ran-architecture.png`.

## Overview

ART-RAN shifts adversarial analysis upstream from conventional attacks on a downstream ML/DRL controller to the **agentic rApp context–reasoning–policy pipeline**. The framework models bounded compromise of rApp inputs, lifecycle/execution dependencies, and policy outputs, then traces the effect through a trusted/fixed Near-RT actuator to measurable RAN behavior.

The current reproducible implementation uses a **`ReferenceReasoner`** as the rApp reasoning component. It should not be interpreted as evidence from a deployed LLM unless an LLM-backed reasoner is explicitly added and evaluated.

### Threat surfaces

- **A1 — Context/input manipulation:** corruption of telemetry, retrieval/knowledge, memory, intent, or other context used by the Non-RT rApp.
- **A2 — Lifecycle/supply-chain compromise:** manipulation of rApp software/model/configuration or its execution dependencies in the SMO/O-Cloud lifecycle.
- **A3 — Reasoning/output-integrity compromise:** generation of harmful but operationally valid high-level rApp policies.

The downstream Near-RT controller is treated as the trusted actuator during adversarial evaluation; the primary experimental question is how far upstream compromise can steer that fixed controller and degrade service-level behavior.

## Experimental basis

ART-RAN uses the public [Colosseum O-RAN COMMAG dataset](https://github.com/wineslab/colosseum-oran-commag-dataset) as the trace basis. The public release reports a 4-BS, 3-MHz/15-PRB setup with three slices and 40 UEs.

The ART-RAN preprocessing/evaluation pipeline uses:

- 250-ms trace aggregation;
- three slices: eMBB, mMTC, and URLLC;
- experiment-level split: `exp1-exp4` train, `exp5` validation, `exp6` test;
- removal of non-positive requested-PRB reports **before aggregation**;
- an empirical action catalog derived from observed slice-PRB allocations;
- an ExtraTrees-based **semi-counterfactual response model** for immediate throughput and grant-ratio response;
- queue evolution from a physical balance equation with trace-inferred exogenous arrivals;
- a 29-D Near-RT observation: 22-D RAN state + 7-D rApp policy;
- MaskablePPO as the downstream Near-RT actuator.

For scenario-specific evaluation, the scenario label constrains the **episode initial state**; subsequent states follow the recorded COMMAG trajectory naturally. Episode starts are sampled uniformly over all eligible start states.

## Repository layout

```text
ART-RAN/
├── art_ran_commag/          # Core environment, response model, reasoner, calibration
├── scripts/                 # Preprocessing, training, attacks, evaluation, audits
├── configs/                 # Reproducible experiment configurations
├── docs/                    # Threat model, dataset, and reproduction notes
│   └── figures/             # Paper/repository figures
├── results/                 # Lightweight final tables/plots only
├── artifacts/               # Pointers/checksums for large model artifacts
├── .github/workflows/       # CI checks
├── README.md
├── CITATION.cff
├── LICENSE
└── requirements.txt
```

Raw COMMAG data, prepared parquet files, empirical-twin binaries, virtual environments, and full training runs are intentionally **not committed**.

## Quick start

### 1. Clone ART-RAN

```bash
git clone https://github.com/Jaredabera/ART-RAN.git
cd ART-RAN
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For exact replication, use `requirements-lock.txt` when it is included in a tagged release.

### 2. Obtain COMMAG

```bash
mkdir -p data
git clone https://github.com/wineslab/colosseum-oran-commag-dataset.git \
  data/colosseum-oran-commag-dataset
```

The dataset is not redistributed by ART-RAN. Please cite the COMMAG paper/repository when using it.

### 3. Prepare trace data

```bash
export DATA_ROOT="$PWD/data/colosseum-oran-commag-dataset"

python scripts/prepare_commag.py \
  --data-root "$DATA_ROOT" \
  --out prepared_v062/commag_wide.parquet
```

### 4. Validate preprocessing and the empirical response model

Run the supplied audit scripts before policy training:

```bash
python scripts/audit_scenario_support_v062.py --help
python scripts/audit_twin_identity_v062.py --help
python scripts/audit_start_sampling_v062.py --help
python scripts/feasibility_audit.py --help
```

A release should pass, at minimum:

- no negative/non-positive aggregated requested-PRB values;
- observed actions represented in the empirical action catalog;
- observed-action response-model fidelity reported by split/scenario;
- representative episode-start sampling;
- action-level QoS feasibility characterized before PPO training.

### 5. Train the Near-RT controller

Use the training script and configuration included in `scripts/` / `configs/` for the tagged release. The trained Near-RT controller is frozen during adversarial evaluation.

### 6. Run ART-RAN attacks and evaluation

The release scripts expose the attack surfaces and paired clean/adversarial evaluation used by the paper. Final commands and seeds are documented in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Reproducibility policy

We separate **code**, **third-party data**, **large generated artifacts**, and **paper results**:

- Source code and lightweight configs: committed to GitHub.
- COMMAG raw data: downloaded from the upstream repository.
- Large empirical-twin/checkpoint files: regenerated locally or hosted as release artifacts with checksums.
- Final paper tables/figures: stored under `results/` with scripts that reproduce them.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and [`artifacts/README.md`](artifacts/README.md).

## Results

Paper results will be added after the final multi-seed training/evaluation pipeline is frozen. We do not publish intermediate diagnostic runs as final paper evidence.

See [`results/README.md`](results/README.md).

## Security and responsible use

ART-RAN is intended for **defensive research, robustness evaluation, and secure O-RAN design**. Experiments should be conducted only on datasets, simulators, emulators, testbeds, or networks for which the researcher has authorization. See [`SECURITY.md`](SECURITY.md).

## Citation

If you use ART-RAN, please cite the repository and the accompanying paper when available:

```bibtex
@misc{ergu2027artran,
  title        = {ART-RAN: Exposing Adversarial Vulnerabilities in Agentic O-RAN rApps},
  author       = {Ergu, Yared Abera and Nguyen, Van-Linh and Lin, Po-Ching and Hwang, Ren-Hung},
  year         = {2027},
  howpublished = {GitHub repository},
  url          = {https://github.com/Jaredabera/ART-RAN},
  note         = {Companion code for the IEEE ICC 2027 manuscript}
}
```

## License

The code in this repository is released under the [MIT License](LICENSE), unless a file states otherwise. Third-party datasets and dependencies remain subject to their own licenses.

## Contact

For research questions, reproducibility issues, or bug reports, please use GitHub Issues.
