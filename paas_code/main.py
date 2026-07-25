"""PAAS: Proxy-Alignment-Aware Selection.

Entry point for running experiments.

Usage:
    python main.py --config config/experiments/sft_cosine.yaml
    python main.py --pilot           # runs pilot experiment
    python main.py --download-data   # downloads and prepares datasets
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="PAAS: Proxy-Alignment-Aware Selection")
    parser.add_argument("--config", help="Path to experiment config file")
    parser.add_argument("--pilot", action="store_true", help="Run pilot experiment")
    parser.add_argument("--download-data", action="store_true", help="Download datasets")
    args = parser.parse_args()

    if args.download_data:
        from scripts.download_data import download_all
        download_all()
        return

    if args.pilot:
        from pilot.run_pilot import run_pilot
        run_pilot()
        return

    if args.config:
        # Full-scale experiment from config
        # TODO: implement YAML config loading
        print(f"Loading config: {args.config}")
        print("Full-scale experiment execution coming soon.")
        print("For now, run: python main.py --pilot")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
