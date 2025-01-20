import os
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt


from plotting_utils import (
    BENCHMARK_FOLDER,
    FIGURES_FOLDER, 
    Project, 
    impl_to_project_func, 
    project_to_color_map,
    sort_impls_by_display_order,
    get_latest_experiment_path,
    set_grid, 
    calculate_tp_per_sec,
    )

sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
from fast_tp.benchmark.analysis_utils import load_benchmarks, grouped_barchart, filter
from fast_tp.benchmark.logging_utils import getLogger




def plot_uvw_benchmark(experiment_path : pathlib.Path) -> None:
    
    benchmarks, metadata = load_benchmarks(BENCHMARK_FOLDER, latest_experiment_path.name)

    configs = metadata['config_labels']
    # config_labels = metadata['config_labels']
    implementations = metadata['implementations']
    directions = metadata['directions']

    # sort_impls_by_display_order(implementations)

    labelfunc = impl_to_project_func
    colormap = project_to_color_map

    dataf32 = {"forward": {}, "backward": {}}
    for i, desc in enumerate(configs):
        for direction in ["forward", "backward"]:
            dataf32[direction][desc] = {}
            for impl in implementations:
                if True: # direction == "forward" or impl != "CUETensorProduct" or 'mace' in desc:
                    f32_benches = [b for b in benchmarks if b["benchmark results"]["rep_dtype"] == "<class 'numpy.float32'>"]
                    exp = filter(f32_benches, {"config_label": desc, 
                                            "direction": direction, 
                                            "implementation_name": impl
                                            }, match_one=True)
                    dataf32[direction][desc][labelfunc(impl)] = calculate_tp_per_sec(exp)

    dataf64 = {"forward": {}, "backward": {}}
    for i, desc in enumerate(configs):
        for direction in ["forward", "backward"]:
            dataf64[direction][desc] = {}
            for impl in implementations:
                if True: # direction == "forward" or impl != "CUETensorProduct" or 'mace' in desc:
                    f64_benches = [b for b in benchmarks if b["benchmark results"]["rep_dtype"] == "<class 'numpy.float64'>"]
                    exp = filter(f64_benches, {"config_label": desc, 
                                            "direction": direction, 
                                            "implementation_name": impl
                                            }, match_one=True)
                    dataf64[direction][desc][labelfunc(impl)] = calculate_tp_per_sec(exp)               

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({'font.size': 11})
        
    fig = plt.figure(figsize=(7, 7))
    gs = fig.add_gridspec(2, 2)
    axs = gs.subplots(sharex=True, sharey='row')

    grouped_barchart(dataf32["forward"], axs[0][0], bar_height_fontsize=0, xticklabel=False, colormap=colormap, group_spacing=6.0)
    grouped_barchart(dataf32["backward"], axs[1][0], bar_height_fontsize=0,xticklabel=True, colormap=colormap, group_spacing=6.0)

    grouped_barchart(dataf64["forward"], axs[0][1], bar_height_fontsize=0, xticklabel=False, colormap=colormap, group_spacing=6.0)
    grouped_barchart(dataf64["backward"], axs[1][1], bar_height_fontsize=0,xticklabel=True, colormap=colormap, group_spacing=6.0)

    for i in range(2):
        for j in range(2):
            set_grid(axs[i][j])

    fig.supylabel("Throughput (# tensor products / s)", x=0.03, y=0.56)

    axs[0][0].set_ylabel("Forward")
    axs[1][0].set_ylabel("Backward")

    axs[1][0].set_xlabel("float32")
    axs[1][1].set_xlabel("float64")

    handles, labels = axs[0][1].get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    axs[0][1].legend(*zip(*unique))

    fig.show()
    fig.tight_layout()
    fig.savefig(str(FIGURES_FOLDER / "uvw_throughput_comparison.pdf"))

if __name__ == "__main__":
    latest_experiment_path = get_latest_experiment_path()
    plot_uvw_benchmark(latest_experiment_path)