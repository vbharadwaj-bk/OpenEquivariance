import numpy as np
import matplotlib.pyplot as plt
import os, json, pathlib, sys, logging

from plotting_utils import (
    BENCHMARK_FOLDER,
    FIGURES_FOLDER, 
    Project, 
    impl_to_project_map, 
    project_to_color_map,
    get_latest_experiment_path
    )

sys.path.insert(1, os.path.join(sys.path[0], '../../../'))
from src.benchmark.analysis_utils import load_benchmarks, grouped_barchart, filter
from src.benchmark.logging_utils import getLogger

logger = getLogger()
logger.setLevel(logging.DEBUG)

plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 11})

latest_experiment_path = get_latest_experiment_path()

benchmarks, metadata = load_benchmarks(BENCHMARK_FOLDER, latest_experiment_path.name)

configs = metadata['configs']
implementations = metadata['implementations']

labelmap = impl_to_project_map
colormap = project_to_color_map

def calculate_tp_per_sec(exp):
    return exp["benchmark results"]["batch_size"] / (np.mean(exp["benchmark results"]["time_millis"]) * 0.001)

data = {"forward": {}, "backward": {}}
for i, config in enumerate(configs):
    for direction in ["forward"]:
        data[direction][config] = {}
        for impl in implementations:
            if True: # direction == "forward":
                exp = filter(benchmarks, {"config": config, 
                                          "direction": direction, 
                                          "implementation_name": impl}, match_one=True)
                
                data[direction][config][labelmap[impl]] = calculate_tp_per_sec(exp)


def set_grid(ax):
    ax.set_axisbelow(True)
    ax.grid(True)
    
fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(5.0, 7.0))
axs[0].set_title("Node Update Tensor Products")
fig.supylabel("Throughput (# tensor products / s)", x=0.03, y=0.56)
grouped_barchart(data["forward"], axs[0], bar_height_fontsize=0, xticklabel=False, colormap=colormap)
# grouped_barchart(data["backward"], axs[1], bar_height_fontsize=0, colormap=colormap)

set_grid(axs[0])
set_grid(axs[1])
axs[0].xaxis.set_ticklabels([])

#axs[1].set_xlabel("Input Configuration")

axs[0].set_ylabel("Forward")
axs[1].set_ylabel("Backward")

handles, labels = axs[0].get_legend_handles_labels()
unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
axs[0].legend(*zip(*unique))

fig.show()
fig.tight_layout()
fig.savefig(str(FIGURES_FOLDER / "uvw_throughput_comparison.pdf"))