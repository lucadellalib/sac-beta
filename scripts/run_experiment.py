#!/usr/bin/env python3

# ==============================================================================
# Copyright 2024 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Run experiment for different seeds."""

import argparse
import datetime
import os
import subprocess


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument(
        "--experiment-dir",
        default=os.path.join(ROOT_DIR, "experiments"),
        help="experiment dir",
    )
    parser.add_argument(
        "--log-dir", default=os.path.join(ROOT_DIR, "logs"), help="log dir"
    )
    parser.add_argument("algorithm", help="algorithm")
    parser.add_argument("task", help="MuJoCo task")
    parser.add_argument("--epoch", default=200, help="number of epochs")
    parser.add_argument("--seeds", nargs="+", default=[0, 1, 2, 3, 4], help="seeds")
    args = parser.parse_args()

    processes = []
    try:
        for seed in args.seeds:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file = os.path.join(
                args.log_dir,
                args.task,
                f"{args.algorithm}_{current_time}_{seed}.txt",
            )
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            process = subprocess.Popen(
                "python "
                f"{args.algorithm}.py "
                f"--task {args.task} "
                f"--epoch {args.epoch} "
                f"--seed {seed} "
                f"--experiment-dir {args.experiment_dir} "
                f"> {log_file} 2>&1",
                shell=True,
                stdin=None,
                stdout=None,
                stderr=None,
                close_fds=True,
            )
            processes.append(process)
        print("Experiments running...")
        for process in processes:
            process.communicate()
        print("Done!")
    except KeyboardInterrupt:
        for process in processes:
            process.kill()
        print("Stopped.")
