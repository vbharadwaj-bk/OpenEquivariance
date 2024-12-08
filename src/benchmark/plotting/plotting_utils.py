import pathlib
from typing import Literal
from collections import defaultdict

BENCHMARK_FOLDER    =pathlib.Path(__file__).parent.parent.parent.parent / "outputs" 
FIGURES_FOLDER      =pathlib.Path(__file__).parent.parent.parent.parent / "figures"

Project = Literal[
    'e3nn',
    'cuE', 
    'ours'
]

impl_to_project_map : dict[str, Project] = defaultdict(
    lambda: 'ours', 
    { 
    'E3NNTensorProduct' : 'e3nn',
    'CUETensorProduct' : 'cuE',
    }
)

project_to_color_map : dict[Project, str] = {
    'e3nn' : 'lightblue',
    'cuE' : 'orange',
    'ours' : 'green'
}

project_to_display_order_map : dict[Project, int] = {
    'e3nn' : 0, 
    'cuE'  : 1,
    'ours' : 2, 
}

def set_grid(ax):
        ax.set_axisbelow(True)
        ax.grid(True)

def calculate_tp_per_sec(exp):
    return exp["benchmark results"]["batch_size"] / (np.mean(exp["benchmark results"]["time_millis"]) * 0.001)

def sort_impls_by_display_order(implementations : list[str]) -> None :
    implementations.sort(key=lambda x : project_to_display_order_map[impl_to_project_map[x]])  

def get_latest_experiment_path() -> pathlib.Path:
    latest_experiment = max(
        (folder for folder in BENCHMARK_FOLDER.iterdir() if folder.is_dir() and folder.name.isdigit()),
        key=lambda x: int(x.name)
    )
    return latest_experiment