"""AutoDL cloud GPU submission script.

Uploads the paas/ project to AutoDL instance, installs dependencies,
and runs training with the specified configuration.

Usage:
    python cloud/submit_autodl.py --config config/experiments/sft_cosine.yaml

Requires AutoDL API key set in environment: AUTODL_API_KEY
or configured in ~/.autodl/config.json

AutoDL web: https://autodl.com
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Submit PAAS training to AutoDL")
    parser.add_argument("--config", default="config/experiments/sft_cosine.yaml",
                        help="Experiment config file")
    parser.add_argument("--instance-id", help="AutoDL instance ID (overrides config)")
    parser.add_argument("--upload-only", action="store_true",
                        help="Only upload code, don't start training")
    args = parser.parse_args()

    # Locate project root
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(str(project_root))

    # Pack all code into a tarball
    print("[autodl] Packing paas/ project...")
    subprocess.run(
        "tar czf /tmp/paas_code.tar.gz --exclude='__pycache__' "
        "--exclude='*.pyc' --exclude='.git' --exclude='results' "
        "--exclude='data/*.json' .",
        shell=True, check=True,
    )
    print("[autodl] Package created: /tmp/paas_code.tar.gz")

    # Instructions for manual upload
    print()
    print("=" * 60)
    print("AutoDL Submission Instructions")
    print("=" * 60)
    print()
    print("1. Open AutoDL console: https://autodl.com/console")
    print("2. Start your GPU instance")
    print("3. Use JupyterLab terminal or SSH to connect")
    print("4. Upload the package:")
    print()
    print("   # On your local machine:")
    print("   scp /tmp/paas_code.tar.gz root@INSTANCE_IP:/root/")
    print()
    print("5. On the AutoDL instance:")
    print()
    print("   cd /root/")
    print("   tar xzf paas_code.tar.gz")
    print("   pip install torch transformers accelerate peft datasets trl bitsandbytes")
    print("   python -c \"from pipeline.runner import run_experiment; ...\"")
    print()
    print("   # Or use the provided script:")
    print("   bash cloud/autodl_setup.sh")
    print()

    # Generate setup script
    setup_script = """#!/bin/bash
# AutoDL setup script for PAAS
# Run on AutoDL instance after uploading code

set -e

cd /root/paas

# Install Python dependencies
pip install torch transformers accelerate peft datasets trl bitsandbytes
pip install lm_eval  # for oracle evaluation

# Verify GPU
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Run training (override with your config)
python3 -c "
from config.schemas import ExperimentConfig
from pipeline.runner import run_experiment

config = ExperimentConfig(
    experiment_name='autodl_sft_cosine',
    seed=42,
)
config.train.schedule = 'cosine'
config.train.total_steps = 1000
config.train.save_every_k_steps = 100

run_experiment(config, force_trigger_test=False)
"

# Package results for download
tar czf results.tar.gz results/
echo 'Results ready: results.tar.gz'
"""
    script_path = project_root / "cloud" / "autodl_setup.sh"
    script_path.write_text(setup_script)
    os.chmod(str(script_path), 0o755)
    print(f"[autodl] Setup script written: {script_path}")


if __name__ == "__main__":
    main()
