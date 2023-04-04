#!/usr/bin/env python3

"""Run experiment."""

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
    parser.add_argument("--epoch", default=1, help="number of epochs")
    args = parser.parse_args()
    for seed in range(2):
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
            f"--logdir {args.experiment_dir} "
            f"> {log_file} 2>&1 &",
            shell=True,
            stdin=None,
            stdout=None,
            stderr=None,
            close_fds=True,
        )
    print("Experiments started.")
