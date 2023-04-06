# Soft Actor-Critic with Beta Policy via Implicit Reparameterization

The goal of this project is to explore the use of [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290v2) with beta policy,
which, compared to the standard normal policy, does not suffer from boundary effect bias and provides faster convergence
(see [Chou et al.](https://proceedings.mlr.press/v70/chou17a.html)).
[Implicit reparameterization](https://arxiv.org/abs/1805.08498v4) is used to draw samples from the policy in a differentiable
manner, as required by SAC. For the experimental evaluation we use a subset of [MuJoCo](https://gymnasium.farama.org/environments/mujoco/) continuous
control tasks.

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

To train one of the available algorithms on a MuJoCo task, open a terminal and run:

```bash
conda activate sac-beta
python <algorithm>.py --task <task>
```

Logs and experimental results (metrics, checkpoints, etc.) can be found in the auto-generated `logs`
and `experiments` directory, respectively.

### Reproducing the experimental results

The experiments were run on an Ubuntu 20.04.5 LTS machine with an Intel i7-10875H CPU
with 8 cores @ 2.30 GHz, 32 GB RAM and an NVIDIA GeForce RTX 3070 GPU @ 8 GB with CUDA Toolkit 11.3.1.
To reproduce the experimental results, open a terminal and run:

```bash
conda activate sac-beta

python run_experiment.py sac_normal Ant-v4
python run_experiment.py sac_normal HalfCheetah-v4
python run_experiment.py sac_normal Walker2d-v4
python run_experiment.py sac_normal Swimmer-v4

python run_experiment.py sac_tanh_normal Ant-v4
python run_experiment.py sac_tanh_normal HalfCheetah-v4
python run_experiment.py sac_tanh_normal Walker2d-v4
python run_experiment.py sac_tanh_normal Swimmer-v4

python run_experiment.py sac_beta_omt Ant-v4
python run_experiment.py sac_beta_omt HalfCheetah-v4
python run_experiment.py sac_beta_omt Walker2d-v4
python run_experiment.py sac_beta_omt Swimmer-v4

python run_experiment.py sac_beta_ad Ant-v4
python run_experiment.py sac_beta_ad HalfCheetah-v4
python run_experiment.py sac_beta_ad Walker2d-v4
python run_experiment.py sac_beta_ad Swimmer-v4
```

Wait for the experiments to finish. To plot the results, open a terminal and run:

```bash
python plotter.py --root-dir experiments/Ant-v4 --shaded-std --legend-pattern "^([\w-]+)" -u --output-path Ant-v4.pdf
python plotter.py --root-dir experiments/HalfCheetah-v4 --shaded-std --legend-pattern "^([\w-]+)" -u --output-path HalfCheetah-v4.pdf
python plotter.py --root-dir experiments/Walker2d-v4 --shaded-std --legend-pattern "^([\w-]+)" -u --output-path Walker2d-v4.pdf
python plotter.py --root-dir experiments/Swimmer-v4 --shaded-std --legend-pattern "^([\w-]+)" -u --output-path Swimmer-v4.pdf
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

Luca Della Libera <[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)>

---------------------------------------------------------------------------------------------------------
