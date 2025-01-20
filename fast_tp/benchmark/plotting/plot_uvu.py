import numpy as np
import matplotlib.pyplot as plt
import os, json, pathlib, sys

from plotting_utils import (
    BENCHMARK_FOLDER,
    FIGURES_FOLDER, 
    Project, 
    impl_to_project_func, 
    project_to_color_map,
    project_to_display_order_map,
    get_latest_experiment_path,
    sort_impls_by_display_order,
    calculate_tp_per_sec,
    set_grid
    )

sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
from fast_tp.benchmark.analysis_utils import load_benchmarks, grouped_barchart, filter

plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 11})

def plot_uvu_benchmark(experiment_path : pathlib.Path) -> None: 
    benchmarks, metadata = load_benchmarks(BENCHMARK_FOLDER, experiment_path.name)

    config_labels = metadata["config_labels"]
    implementations =  [
        "E3NNTensorProduct", 
        "CUETensorProduct", 
        "LoopUnrollTP"
        ]

    labelmap = impl_to_project_func
    colormap = project_to_color_map


    data = {"forward": {}, "backward": {}}
    for config_label in enumerate(config_labels):
        for direction in ["forward", "backward"]:
            data[direction][config_label] = {}
            for impl in implementations:
                if direction == "forward" or impl != "CUETensorProduct":
                    exp = filter(benchmarks, {"config_label": config_label, 
                                            "direction": direction, 
                                            "implementation_name": impl}, match_one=True)
                    data[direction][config_label][labelmap(impl)] = calculate_tp_per_sec(exp)
        
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(5.0, 7.0))
    axs[0].set_title("Node-Edge Tensor Products, No Fusion")
    fig.supylabel("Throughput (# tensor products / s)", x=0.03, y=0.56)
    grouped_barchart(data["forward"], axs[0], bar_height_fontsize=0, xticklabel=False, colormap=colormap)
    grouped_barchart(data["backward"], axs[1], bar_height_fontsize=0, colormap=colormap)

    set_grid(axs[0])
    set_grid(axs[1])
    axs[0].xaxis.set_ticklabels([])

    axs[0].set_ylabel("Forward")
    axs[1].set_ylabel("Backward")

    handles, labels = axs[0].get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    axs[0].legend(*zip(*unique))

    fig.show()
    fig.tight_layout()
    fig.savefig(str(FIGURES_FOLDER / "uvu_throughput_comparison.pdf"))

if __name__ == "__main__":
    latest_experiment_path = get_latest_experiment_path()
    plot_uvu_benchmark(latest_experiment_path)