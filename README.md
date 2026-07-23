# Code and data accompanying the paper *A Resolution Framework for Finite-Sample Model Selection*

This repository contains the implementation of the Δ_min diagnostic framework, experiment scripts, analysis code, and supplementary data for the paper.

## Repository structure

```
├── main.tex                       # Paper source (LaTeX)
├── refs.bib                       # Bibliography
├── sn-jnl.cls                     # Springer Nature journal class
├── sn-mathphys-ay.bst             # Bibliography style
├── LICENSE                        # Apache 2.0
│
├── experiments/                   # Experiment scripts
│   ├── decision_rule_monte_carlo.py  # Decision Rule 1 contour (Fig. 7)
│   └── mistral_seeds.py              # Mistral-7B LoRA training (seeds 43,44)
│
├── figures/                       # Generated figures
│   ├── fig_decision_rule_contour.pdf/png
│   ├── fig_synthetic_validation.png
│   ├── fig_simulation_power_curve.png
│   ├── fig_bootstrap_selection.png
│   └── fig_literature_survey.png
│
├── data/                          # Supplementary data
│   ├── loss_deltamin_results.json     # Loss-based Δ_min extraction (§7)
│   ├── checkpoint_accuracies.csv      # Per-checkpoint accuracy tables
│   ├── bootstrap_results.csv          # Bootstrap selection probabilities
│   ├── literature_survey.csv          # Full extraction table (§9, Appendix E)
│   └── mnist_results.csv              # MNIST random forest results (§8.3)
│
├── bst/                           # Bibliography style files
├── docs/                          # Supplementary documentation
├── paas_code/                     # Training and evaluation pipeline
└── _submit/                       # Submission archive
```

## Key results

- **Δ_min diagnostic**: `compute_delta_min.py` implements the threshold calculator
- **Decision Rule 1**: `experiments/decision_rule_monte_carlo.py` generates the regret contour
- **Mistral replication**: `experiments/mistral_seeds.py` reproduces the B2 experiment
- **Loss-based Δ_min**: `compute_loss_deltamin.py` (§7 continuous metric extension)

## Reproducibility

All experiment configurations are documented in the paper (Appendices A–E). Model
checkpoints are not redistributed due to size; the training scripts and hyperparameters
needed to reproduce them are included.

## License

Apache 2.0 — see LICENSE for details.
