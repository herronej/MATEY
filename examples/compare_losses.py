#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.
#
# Compare loss curves from DDP baseline vs pipeline-parallel training.
#
# Usage:
#   python compare_losses.py \
#       --ddp_dir ./Dev_JHUTDB3D_TTMG/basic_config/demo \
#       --pp_dir  ./Dev_JHUTDB3D_TTMG/basic_config/pp_train
#
# Reads train_log_epoch*.json from both directories, prints a comparison
# table, and optionally generates a matplotlib plot.

import argparse
import json
import os
import glob
import sys


def load_logs(exp_dir):
    """Load train_log_epoch*.json files from an experiment directory."""
    pattern = os.path.join(exp_dir, 'train_log_epoch*.json')
    files = sorted(glob.glob(pattern),
                   key=lambda f: int(f.split('epoch')[-1].split('.')[0]))
    logs = []
    for f in files:
        with open(f) as fp:
            logs.append(json.load(fp))
    return logs


def print_comparison(ddp_logs, pp_logs):
    """Print side-by-side comparison table."""
    print(f"{'Epoch':>6} | {'DDP train':>12} {'DDP valid':>12} | "
          f"{'PP train':>12} {'PP valid':>12} | "
          f"{'Train diff':>11} {'Valid diff':>11}")
    print("-" * 95)

    n = min(len(ddp_logs), len(pp_logs))
    for i in range(n):
        d = ddp_logs[i]
        p = pp_logs[i]
        dt = d.get('train_nrmse', d.get('train_nrmse', float('nan')))
        dv = d.get('valid_nrmse', d.get('valid_nrmse', float('nan')))
        pt = p.get('train_nrmse', float('nan'))
        pv = p.get('valid_nrmse', float('nan'))
        tdiff = (pt - dt) / max(abs(dt), 1e-8) * 100
        vdiff = (pv - dv) / max(abs(dv), 1e-8) * 100
        epoch = d.get('epoch', p.get('epoch', i + 1))
        print(f"{epoch:>6} | {dt:>12.6f} {dv:>12.6f} | "
              f"{pt:>12.6f} {pv:>12.6f} | "
              f"{tdiff:>+10.1f}% {vdiff:>+10.1f}%")

    # Summary
    if n > 0:
        ddp_final_v = ddp_logs[n-1].get('valid_nrmse', float('nan'))
        pp_final_v = pp_logs[n-1].get('valid_nrmse', float('nan'))
        print(f"\nFinal valid NRMSE: DDP={ddp_final_v:.6f}, PP={pp_final_v:.6f}, "
              f"diff={(pp_final_v - ddp_final_v) / max(abs(ddp_final_v), 1e-8) * 100:+.1f}%")


def try_plot(ddp_logs, pp_logs, save_path=None):
    """Generate matplotlib comparison plot if available."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot generation")
        return

    n = min(len(ddp_logs), len(pp_logs))
    epochs = list(range(1, n + 1))

    ddp_train = [d.get('train_nrmse', float('nan')) for d in ddp_logs[:n]]
    ddp_valid = [d.get('valid_nrmse', float('nan')) for d in ddp_logs[:n]]
    pp_train = [p.get('train_nrmse', float('nan')) for p in pp_logs[:n]]
    pp_valid = [p.get('valid_nrmse', float('nan')) for p in pp_logs[:n]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, ddp_train, 'b-o', label='DDP train', markersize=4)
    ax1.plot(epochs, pp_train, 'r--s', label='PP train', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('NRMSE')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, ddp_valid, 'b-o', label='DDP valid', markersize=4)
    ax2.plot(epochs, pp_valid, 'r--s', label='PP valid', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('NRMSE')
    ax2.set_title('Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        out = os.path.join(os.path.dirname(ddp_logs[0].get('_path', '.')),
                           'ddp_vs_pp_comparison.png')
        plt.savefig('ddp_vs_pp_comparison.png', dpi=150)
        print(f"Plot saved to ddp_vs_pp_comparison.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare DDP vs PP loss curves")
    parser.add_argument("--ddp_dir", required=True, type=str,
                        help="DDP experiment directory (contains train_log_epoch*.json)")
    parser.add_argument("--pp_dir", required=True, type=str,
                        help="PP experiment directory (contains train_log_epoch*.json)")
    parser.add_argument("--plot", default=None, type=str,
                        help="Save plot to this path (optional)")
    args = parser.parse_args()

    ddp_logs = load_logs(args.ddp_dir)
    pp_logs = load_logs(args.pp_dir)

    if not ddp_logs:
        print(f"No logs found in {args.ddp_dir}")
        sys.exit(1)
    if not pp_logs:
        print(f"No logs found in {args.pp_dir}")
        sys.exit(1)

    print(f"DDP: {len(ddp_logs)} epochs from {args.ddp_dir}")
    print(f"PP:  {len(pp_logs)} epochs from {args.pp_dir}")
    print()

    print_comparison(ddp_logs, pp_logs)
    try_plot(ddp_logs, pp_logs, save_path=args.plot)
