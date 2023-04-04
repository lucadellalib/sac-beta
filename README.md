# Soft Actor-Critic with Beta Distribution

The goal of this project is to explore the use of [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) with Beta distribution.
We evaluate the proposed methods on a variety of [MuJoCo](https://gymnasium.farama.org/environments/mujoco/) continuous control tasks.

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

First of all, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
Clone or download and extract the repository, navigate to `<path-to-repository>`, open a terminal and run:

```bash
conda env create -f environment.yml
```

Project dependencies (pinned to a specific version to reduce compatibility and reproducibility issues)
will be installed in a [Conda](https://www.anaconda.com/) virtual environment named `sac-beta`.

To activate it, run:

```bash
conda activate sac-beta
```

To deactivate it, run:

```bash
conda deactivate
```

To permanently delete it, run:

```bash
conda remove --n sac-beta --all
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

### Running an experiment

To train one of the available algorithms on a MuJoCo task open a terminal and run:

```bash
conda activate sac-beta
python run_experiment.py <algorithm> <task>
```

Logs and experimental results (metrics, checkpoints, etc.) can be found in the auto-generated `logs`
and `experiments` directory, respectively.

### Reproducing the experimental results

The experiments were run on an Ubuntu 20.04.5 LTS machine with an Intel i7-10875H CPU
with 8 cores @ 2.30 GHz, 32 GB RAM and an NVIDIA GeForce RTX 3070 GPU @ 8 GB with CUDA Toolkit 11.3.1.
To reproduce the experimental results, open a terminal and run:

```bash
conda activate sac-beta

python run_experiment.py sac_tanh_normal Ant-v4
python run_experiment.py sac_tanh_normal HalfCheetah-v4
python run_experiment.py sac_tanh_normal Walker2d-v4
python run_experiment.py sac_tanh_normal Swimmer-v4

python run_experiment.py sac_beta_omt Ant-v4
python run_experiment.py sac_beta_omt HalfCheetah-v4
python run_experiment.py sac_beta_omt Walker2d-v4
python run_experiment.py sac_beta_omt Swimmer-v4
```

Wait for the experiments to finish. To plot the results, open a terminal and run:

```bash
python plotter.py --root-dir experiments --shaded-std --legend-pattern "\\w+" -u --output-path benchmark.pdf
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

Luca Della Libera <[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)>

---------------------------------------------------------------------------------------------------------
