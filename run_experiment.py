#!/usr/bin/env python3

"""Run experiment for different seeds."""

import argparse
import datetime
import os
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument(
        "--experiment-dir", default="experiments", help="experiment dir"
    )
    parser.add_argument("--log-dir", default="logs", help="log dir")
    parser.add_argument("algorithm", help="algorithm")
    parser.add_argument("task", help="MuJoCo task")
    parser.add_argument("--epoch", default=200, help="number of epochs")
    parser.add_argument("--seeds", nargs="+", default=[0, 1, 2, 3, 4], help="seeds")
    args = parser.parse_args()
    for seed in args.seeds:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(
            args.log_dir,
            args.task,
            f"{args.algorithm}_{current_time}_{seed}.txt",
        )
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        subprocess.Popen(
            "nohup python "
            f"{args.algorithm}.py "
            f"--task {args.task} "
            f"--epoch {args.epoch} "
            f"--seed {seed} "
            f"--experiment-dir {args.experiment_dir} "
            f"> {log_file} 2>&1 &",
            shell=True,
            stdin=None,
            stdout=None,
            stderr=None,
            close_fds=True,
        )
    print("Experiments started.")
