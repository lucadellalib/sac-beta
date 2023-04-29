# Soft Actor-Critic with Beta Policy via Implicit Reparameterization Gradients

This project investigates the use of [soft actor-critic (SAC)](https://arxiv.org/abs/1801.01290v2) with a beta
policy, which, compared to a normal policy, does not suffer from boundary effect bias and [has been shown to
convergence faster](https://proceedings.mlr.press/v70/chou17a.html). Implicit reparameterization approaches based
on [automatic differentiation](https://arxiv.org/abs/1805.08498v4) and [optimal mass transport](https://arxiv.org/abs/1806.01851v2)
are used to draw samples from the policy in a differentiable manner, as required by SAC. For the experimental
evaluation we use a subset of [MuJoCo](https://gymnasium.farama.org/environments/mujoco/) continuous control tasks.

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

The experiments were run on a CentOS Linux 7 machine with an Intel Xeon Gold 6148 Skylake CPU with 20 cores
@ 2.40 GHz, 32 GB RAM and an NVIDIA Tesla V100 SXM2 @ 16GB with CUDA Toolkit 11.4.2.

#### Performance comparison

To reproduce the experimental results, open a terminal and run:

```bash
conda activate sac-beta

python run_experiment.py sac_normal Ant-v4
python run_experiment.py sac_tanh_normal Ant-v4
python run_experiment.py sac_beta_ad Ant-v4
python run_experiment.py sac_beta_omt Ant-v4

python run_experiment.py sac_normal HalfCheetah-v4
python run_experiment.py sac_tanh_normal HalfCheetah-v4
python run_experiment.py sac_beta_ad HalfCheetah-v4
python run_experiment.py sac_beta_omt HalfCheetah-v4

python run_experiment.py sac_normal Hopper-v4
python run_experiment.py sac_tanh_normal Hopper-v4
python run_experiment.py sac_beta_ad Hopper-v4
python run_experiment.py sac_beta_omt Hopper-v4

python run_experiment.py sac_normal Walker2d-v4
python run_experiment.py sac_tanh_normal Walker2d-v4
python run_experiment.py sac_beta_ad Walker2d-v4
python run_experiment.py sac_beta_omt Walker2d-v4
```

Wait for the experiments to finish. To plot the results, open a terminal and run:

```bash
python plotter.py --root-dir experiments/Ant-v4 --smooth 1 --shaded-std --legend-pattern "^([\w-]+)" --title Ant-v4 -u --output-path Ant-v4.pdf
python plotter.py --root-dir experiments/HalfCheetah-v4 --smooth 1 --shaded-std --legend-pattern "$^" --title HalfCheetah-v4 --ylabel "" -u --output-path HalfCheetah-v4.pdf
python plotter.py --root-dir experiments/Hopper-v4 --smooth 1 --shaded-std --legend-pattern "$^" --title Hopper-v4 --ylabel "" -u --output-path Hopper-v4.pdf
python plotter.py --root-dir experiments/Walker2d-v4 --smooth 1 --shaded-std --legend-pattern "$^" --title Walker2d-v4 --ylabel "" -u --output-path Walker2d-v4.pdf
```

#### Ablation study

To reproduce the experimental results, open a terminal and run:

```bash
conda activate sac-beta

python run_experiment.py sac_beta_omt Ant-v4 --experiment-dir experiments/ablation
python run_experiment.py sac_beta_omt_no_clip Ant-v4 --experiment-dir experiments/ablation
python run_experiment.py sac_beta_omt_non-concave Ant-v4 --experiment-dir experiments/ablation
python run_experiment.py sac_beta_omt_softplus Ant-v4 --experiment-dir experiments/ablation
```

Wait for the experiments to finish. To plot the results, open a terminal and run:

```bash
python plotter.py --root-dir experiments/ablation/Ant-v4 --smooth 1 --shaded-std --legend-pattern "^([\w-]+)" -u --output-path ablation.pdf
```


---------------------------------------------------------------------------------------------------------

## 📧 Contact

Luca Della Libera <[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)>

---------------------------------------------------------------------------------------------------------
