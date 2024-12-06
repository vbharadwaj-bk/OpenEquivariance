import pathlib
from typing import Literal
from collections import defaultdict

BENCHMARK_FOLDER    =pathlib.Path("/global/homes/a/aglover/equivariant_spmm/outputs")
FIGURES_FOLDER      =pathlib.Path("/global/homes/a/aglover/equivariant_spmm/figures")

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

def get_latest_experiment_path() -> pathlib.Path:
    latest_experiment = max(
        (folder for folder in BENCHMARK_FOLDER.iterdir() if folder.is_dir() and folder.name.isdigit()),
        key=lambda x: int(x.name)
    )
    return latest_experiment