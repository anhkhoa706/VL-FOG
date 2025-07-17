#!/usr/bin/env python3
"""
Simple script to launch TensorBoard for monitoring training.
Usage: python launch_tensorboard.py [--port 6006] [--runs_dir runs]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def find_latest_run():
    """Find the most recent run directory with TensorBoard logs."""
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return None
    
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
    
    # Sort by modification time and get the latest
    latest_run = max(run_dirs, key=lambda d: d.stat().st_mtime)
    tensorboard_dir = latest_run / "tensorboard"
    
    if tensorboard_dir.exists():
        return str(tensorboard_dir)
    return None

def main():
    parser = argparse.ArgumentParser(description="Launch TensorBoard for training monitoring")
    parser.add_argument("--port", type=int, default=6006, help="TensorBoard port (default: 6006)")
    parser.add_argument("--runs_dir", type=str, default="runs", help="Runs directory (default: runs)")
    parser.add_argument("--latest", action="store_true", help="Automatically find latest run")
    
    args = parser.parse_args()
    
    if args.latest:
        log_dir = find_latest_run()
        if log_dir is None:
            print("❌ No TensorBoard logs found in recent runs")
            sys.exit(1)
        print(f"📊 Found latest run: {log_dir}")
    else:
        log_dir = args.runs_dir
        if not os.path.exists(log_dir):
            print(f"❌ Directory {log_dir} does not exist")
            sys.exit(1)
    
    print(f"🚀 Starting TensorBoard on port {args.port}")
    print(f"📂 Log directory: {log_dir}")
    print(f"🌐 Open your browser to: http://localhost:{args.port}")
    print(f"⏹️ Press Ctrl+C to stop TensorBoard")
    
    try:
        # Launch TensorBoard
        cmd = ["tensorboard", "--logdir", log_dir, "--port", str(args.port), "--reload_interval", "1"]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n⏹️ TensorBoard stopped")
    except FileNotFoundError:
        print("❌ TensorBoard not found. Install with: pip install tensorboard")
        sys.exit(1)

if __name__ == "__main__":
    main() 