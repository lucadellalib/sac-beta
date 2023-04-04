#!/usr/bin/env python3

"""Plot results."""

# Adapted from:
# https://github.com/thu-ml/tianshou/blob/v0.5.0/examples/mujoco/plotter.py
# https://github.com/thu-ml/tianshou/blob/v0.5.0/examples/mujoco/tools.py

import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib
import numpy as np
import tqdm
from matplotlib import pyplot as plt
from matplotlib import rc
from tensorboard.backend.event_processing import event_accumulator


matplotlib.use("pdf")


COLORS = [
    # deepmind style
    "#0072B2",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    # "#F0E442",
    "#d73027",  # RED
    # built-in color
    "blue",
    "red",
    "pink",
    "cyan",
    "magenta",
    "yellow",
    "black",
    "purple",
    "brown",
    "orange",
    "teal",
    "lightblue",
    "lime",
    "lavender",
    "turquoise",
    "darkgreen",
    "tan",
    "salmon",
    "gold",
    "darkred",
    "darkblue",
    "green",
    # personal color
    "#313695",  # DARK BLUE
    "#74add1",  # LIGHT BLUE
    "#f46d43",  # ORANGE
    "#4daf4a",  # GREEN
    "#984ea3",  # PURPLE
    "#f781bf",  # PINK
    "#ffc832",  # YELLOW
    "#000000",  # BLACK
]


def find_all_files(root_dir, pattern):
    """Find all files under root_dir according to relative pattern."""
    file_list = []
    for dirname, _, files in os.walk(root_dir):
        for f in files:
            absolute_path = os.path.join(dirname, f)
            if re.match(pattern, absolute_path):
                file_list.append(absolute_path)
    return file_list


def group_files(file_list, pattern):
    res = defaultdict(list)
    for f in file_list:
        match = re.search(pattern, f)
        key = match.group() if match else ""
        res[key].append(f)
    return res


def csv2numpy(csv_file):
    csv_dict = defaultdict(list)
    reader = csv.DictReader(open(csv_file))
    for row in reader:
        for k, v in row.items():
            csv_dict[k].append(eval(v))
    return {k: np.array(v) for k, v in csv_dict.items()}


def convert_tfevents_to_csv(root_dir, refresh=False):
    """Recursively convert test/reward from all tfevent file under root_dir to csv.

    This function assumes that there is at most one tfevents file in each directory
    and will add suffix to that directory.

    :param bool refresh: re-create csv file under any condition.
    """
    tfevent_files = find_all_files(root_dir, re.compile(r"^.*tfevents.*$"))
    print(f"Converting {len(tfevent_files)} tfevents files under {root_dir} ...")
    result = {}
    with tqdm.tqdm(tfevent_files) as t:
        for tfevent_file in t:
            t.set_postfix(file=tfevent_file)
            output_file = os.path.join(
                os.path.split(tfevent_file)[0], "test_reward.csv"
            )
            if os.path.exists(output_file) and not refresh:
                content = list(csv.reader(open(output_file, "r")))
                if content[0] == ["env_step", "reward", "time"]:
                    for i in range(1, len(content)):
                        content[i] = list(map(eval, content[i]))
                    result[output_file] = content
                    continue
            ea = event_accumulator.EventAccumulator(tfevent_file)
            ea.Reload()
            initial_time = ea._first_event_timestamp
            content = [["env_step", "reward", "time"]]
            for test_reward in ea.scalars.Items("test/reward"):
                content.append(
                    [
                        round(test_reward.step, 4),
                        round(test_reward.value, 4),
                        round(test_reward.wall_time - initial_time, 4),
                    ]
                )
            csv.writer(open(output_file, "w")).writerows(content)
            result[output_file] = content
    return result


def merge_csv(csv_files, root_dir, remove_zero=False):
    """Merge result in csv_files into a single csv file."""
    assert len(csv_files) > 0
    if remove_zero:
        for v in csv_files.values():
            if v[1][0] == 0:
                v.pop(1)
    sorted_keys = sorted(csv_files.keys())
    sorted_values = [csv_files[k][1:] for k in sorted_keys]
    content = [
        ["env_step", "reward", "reward:shaded"]
        + list(map(lambda f: "reward:" + os.path.relpath(f, root_dir), sorted_keys))
    ]
    for rows in zip(*sorted_values):
        array = np.array(rows)
        assert len(set(array[:, 0])) == 1, (set(array[:, 0]), array[:, 0])
        line = [rows[0][0], round(array[:, 1].mean(), 4), round(array[:, 1].std(), 4)]
        line += array[:, 1].tolist()
        content.append(line)
    output_path = os.path.join(root_dir, f"test_reward_{len(csv_files)}seeds.csv")
    print(f"Output merged csv file to {output_path} with {len(content[1:])} lines.")
    csv.writer(open(output_path, "w")).writerows(content)


def smooth(y, radius, mode="two_sided", valid_only=False):
    """Smooth signal y, where radius is determines the size of the window.

    mode="twosided":
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode="causal":
        average over the window [max(index - radius, 0), index]
    valid_only: put nan in entries where the full-sized window is not available

    """
    assert mode in ("two_sided", "causal")
    if len(y) < 2 * radius + 1:
        return np.ones_like(y) * y.mean()
    elif mode == "two_sided":
        convkernel = np.ones(2 * radius + 1)
        out = np.convolve(y, convkernel, mode="same") / np.convolve(
            np.ones_like(y), convkernel, mode="same"
        )
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == "causal":
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel, mode="full") / np.convolve(
            np.ones_like(y), convkernel, mode="full"
        )
        out = out[: -radius + 1]
        if valid_only:
            out[:radius] = np.nan
    return out


def plot_ax(
    ax,
    file_lists,
    legend_pattern=".*",
    xlabel=None,
    ylabel=None,
    title=None,
    xlim=None,
    xkey="env_step",
    ykey="reward",
    smooth_radius=0,
    shaded_std=True,
    legend_outside=False,
):
    def legend_fn(x):
        # return os.path.split(os.path.join(
        #     args.root_dir, x))[0].replace("/", "_") + " (10)"
        return re.search(legend_pattern, x).group(0)

    legends = map(legend_fn, file_lists)
    # sort filelist according to legends
    file_lists = [f for _, f in sorted(zip(legends, file_lists))]
    legends = list(map(legend_fn, file_lists))

    for index, csv_file in enumerate(file_lists):
        csv_dict = csv2numpy(csv_file)
        x, y = csv_dict[xkey], csv_dict[ykey]
        y = smooth(y, radius=smooth_radius)
        color = COLORS[index % len(COLORS)]
        ax.plot(x, y, color=color)
        if shaded_std and ykey + ":shaded" in csv_dict:
            y_shaded = smooth(csv_dict[ykey + ":shaded"], radius=smooth_radius)
            ax.fill_between(x, y - y_shaded, y + y_shaded, color=color, alpha=0.2)

    ax.legend(
        legends,
        loc=2 if legend_outside else None,
        bbox_to_anchor=(1, 1) if legend_outside else None,
    )
    if xlim is not None:
        ax.set_xlim(xmin=0, xmax=xlim)
    # add title
    ax.set_title(title)
    # add labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    # add grid
    plt.grid()


def plot_figure(
    file_lists,
    group_pattern=None,
    fig_length=6,
    fig_width=6,
    sharex=False,
    sharey=False,
    title=None,
    usetex=False,
    **kwargs,
):
    rc("text", usetex=usetex)
    rc("font", family="serif", serif=["Computer Modern"], size=14)
    rc("axes", labelsize=15)
    rc("legend", fontsize=12)
    rc("xtick", direction="out")
    rc("ytick", direction="out")

    if not group_pattern:
        fig, ax = plt.subplots(figsize=(fig_length, fig_width))
        plot_ax(ax, file_lists, title=title, **kwargs)
    else:
        res = group_files(file_lists, group_pattern)
        row_n = int(np.ceil(len(res) / 3))
        col_n = min(len(res), 3)
        fig, axes = plt.subplots(
            row_n,
            col_n,
            sharex=sharex,
            sharey=sharey,
            figsize=(fig_length * col_n, fig_width * row_n),
            squeeze=False,
        )
        axes = axes.flatten()
        for i, (k, v) in enumerate(res.items()):
            plot_ax(axes[i], v, title=k, **kwargs)
    if title:  # add title
        fig.suptitle(title, fontsize=20)
    fig.tight_layout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot experimental results")
    parser.add_argument("--root-dir", default="./", help="root dir (default: ./)")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-generate all csv files instead of using existing one.",
    )
    parser.add_argument(
        "--remove-zero",
        action="store_true",
        help="Remove the data point of env_step == 0.",
    )
    parser.add_argument(
        "--fig-length",
        type=int,
        default=6,
        help="matplotlib figure length (default: 6)",
    )
    parser.add_argument(
        "--fig-width", type=int, default=6, help="matplotlib figure width (default: 6)"
    )
    parser.add_argument(
        "--style", default="classic", help="matplotlib figure style (default: seaborn)"
    )
    parser.add_argument(
        "--title", default=None, help="matplotlib figure title (default: None)"
    )
    parser.add_argument(
        "--xkey", default="env_step", help="x-axis key in csv file (default: env_step)"
    )
    parser.add_argument(
        "--ykey", default="reward", help="y-axis key in csv file (default: reward)"
    )
    parser.add_argument(
        "--smooth", type=int, default=0, help="smooth radius of y axis (default: 0)"
    )
    parser.add_argument(
        "--xlabel", default="Timesteps", help="matplotlib figure xlabel"
    )
    parser.add_argument("--ylabel", default="Return", help="matplotlib figure ylabel")
    parser.add_argument(
        "--shaded-std",
        action="store_true",
        help="shaded region corresponding to standard deviation of the group",
    )
    parser.add_argument(
        "--sharex",
        action="store_true",
        help="whether to share x axis within multiple sub-figures",
    )
    parser.add_argument(
        "--sharey",
        action="store_true",
        help="whether to share y axis within multiple sub-figures",
    )
    parser.add_argument(
        "--legend-outside",
        action="store_true",
        help="place the legend outside of the figure",
    )
    parser.add_argument(
        "--xlim", type=int, default=None, help="x-axis limitation (default: None)"
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default=r".*/test_reward_\d+seeds.csv$",
        help="regular expression to determine whether or not to include target csv "
        "file, default to including all test_reward_{num}seeds.csv file under rootdir",
    )
    parser.add_argument(
        "--group-pattern",
        type=str,
        default=r"(/|^)\w*?\-v(\d|$)",
        help="regular expression to group files in sub-figure, default to grouping "
        "according to env_name dir, "
        " means no grouping",
    )
    parser.add_argument(
        "--legend-pattern",
        type=str,
        default=r".*",
        help="regular expression to extract legend from csv file path, default to "
        "using file path as legend name.",
    )
    parser.add_argument("--show", action="store_true", help="show figure")
    parser.add_argument(
        "--output-path", type=str, help="figure save path", default="./figure.png"
    )
    parser.add_argument(
        "--dpi", type=int, default=200, help="figure dpi (default: 200)"
    )
    parser.add_argument(
        "-u",
        "--usetex",
        action="store_true",
        help="render text with LaTeX",
    )
    args = parser.parse_args()
    csv_files = convert_tfevents_to_csv(args.root_dir, args.refresh)
    merge_csv(csv_files, args.root_dir, args.remove_zero)
    file_lists = find_all_files(args.root_dir, re.compile(args.file_pattern))
    file_lists = [os.path.relpath(f, args.root_dir) for f in file_lists]
    if args.style:
        plt.style.use(args.style)
    os.chdir(args.root_dir)
    plot_figure(
        file_lists,
        group_pattern=args.group_pattern,
        legend_pattern=args.legend_pattern,
        fig_length=args.fig_length,
        fig_width=args.fig_width,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        xkey=args.xkey,
        ykey=args.ykey,
        xlim=args.xlim,
        sharex=args.sharex,
        sharey=args.sharey,
        smooth_radius=args.smooth,
        shaded_std=args.shaded_std,
        legend_outside=args.legend_outside,
        usetex=args.usetex,
    )
    if args.output_path:
        plt.savefig(args.output_path, dpi=args.dpi, bbox_inches="tight")
    if args.show:
        plt.show()
