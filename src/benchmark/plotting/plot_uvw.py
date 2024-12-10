import numpy as np
import matplotlib.pyplot as plt
import os, json, pathlib, sys, re

from plotting_utils import (
    BENCHMARK_FOLDER,
    FIGURES_FOLDER, 
    Project, 
    impl_to_project_map, 
    project_to_color_map,
    sort_impls_by_display_order,
    get_latest_experiment_path,
    set_grid, 
    calculate_tp_per_sec,
    )

sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
from src.benchmark.analysis_utils import load_benchmarks, grouped_barchart, filter
from src.benchmark.logging_utils import getLogger




def plot_uvw_benchmark(experiment_path : pathlib.Path) -> None:
    
    benchmarks, metadata = load_benchmarks(BENCHMARK_FOLDER, latest_experiment_path.name)

    configs = metadata['config_strs']
    implementations = metadata['implementations']
    directions = metadata['directions']

    sort_impls_by_display_order(implementations)

    labelmap = impl_to_project_map
    colormap = project_to_color_map

    def calculate_tp_per_sec(exp):
        return exp["benchmark results"]["batch_size"] / (np.mean(exp["benchmark results"]["time_millis"]) * 0.001)

    data = {"forward": {}, "backward": {}}
    for direction in directions:
        data[direction] = {}
        for config in configs: 
            data[direction][config] = {}
            for impl in implementations:
                exp = filter(benchmarks, {"config_str": config, 
                                            "direction": direction, 
                                            "implementation_name": impl}, match_one=True)
                
                data[direction][config][labelmap[impl]] = calculate_tp_per_sec(exp)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({'font.size': 11})

    
        
    fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(5.0, 7.0))
    axs[0].set_title("Node Update Tensor Products")
    fig.supylabel("Throughput (# tensor products / s)", x=0.03, y=0.56)
    grouped_barchart(data["forward"], axs[0], bar_height_fontsize=0, xticklabel=False, colormap=colormap)
    grouped_barchart(data["backward"], axs[1], bar_height_fontsize=0, xticklabel=True, colormap=colormap)

    set_grid(axs[0])
    set_grid(axs[1])
    axs[0].xaxis.set_ticklabels([])

    xtick_labels = axs[1].get_xticklabels()

    for tick in xtick_labels:
        text = tick.get_text()
        modified_text = text[text.index('('):]
        tick.set_text(modified_text)
        tick.set_fontsize(8)

    axs[1].set_xticklabels(xtick_labels)

    axs[0].set_ylabel("Forward")
    axs[1].set_ylabel("Backward")

    handles, labels = axs[0].get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    axs[0].legend(*zip(*unique))

    fig.show()
    fig.tight_layout()
    fig.savefig(str(FIGURES_FOLDER / "uvw_throughput_comparison.pdf"))

if __name__ == "__main__":
    latest_experiment_path = get_latest_experiment_path()
    plot_uvw_benchmark(latest_experiment_path)