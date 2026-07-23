#!/bin/bash
# PAAS AutoDL setup script
# Run on AutoDL instance after selecting image:
#   Framework: PyTorch   Version: 2.5.1   Python: 3.12(ubuntu22.04)   CUDA: 12.4

set -e

cd /root/paas || { echo "Upload paas/ to /root/paas first"; exit 1; }

# Use 50GB data disk for model cache (not 30GB system disk)
export HF_HOME=/root/autodl-tmp/.hf
mkdir -p $HF_HOME

# Install project dependencies (torch+cu124 already pre-installed on AutoDL)
pip install transformers==4.44.2 peft==0.12.0 trl==0.11.0 datasets accelerate bitsandbytes

# Verify GPU
python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')
print(f'torch: {torch.__version__}')
"

# Run 1 seed × cosine schedule (test run)
python3 -c "
from config.schemas import ExperimentConfig, TrainConfig, ProxyConfig, ValidationConfig, MonitorConfig, SelectionConfig, OracleConfig
from pipeline.runner import run_experiment

config = ExperimentConfig(
    experiment_name='autodl_sft_cosine_seed42',
    train=TrainConfig(total_steps=1000, save_every_k_steps=100, schedule='cosine', seed=42),
    proxy=ProxyConfig(win_rate_n=200),
    validation=ValidationConfig(use_human_labels=False),
    monitor=MonitorConfig(rho_window=3, threshold=0.5, min_safe_count=2, min_points=3),
    selection=SelectionConfig(window_frac=0.2, aggregation='uniform'),
    oracle=OracleConfig(benchmarks=['mmlu', 'humaneval'], batch_size=8),
    baselines=['proxy_best', 'uniform_ensemble'],
    seed=42,
    output_dir='results',
)

run_experiment(config, force_trigger_test=False)
"

# Package results for download
tar czf results.tar.gz results/
echo '=== DONE ==='
echo 'Download results: scp root@INSTANCE_IP:/root/paas/results.tar.gz ./'
