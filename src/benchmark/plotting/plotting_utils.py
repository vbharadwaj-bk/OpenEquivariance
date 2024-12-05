import pathlib
from typing import Literal
from collections import defaultdict

BENCHMARK_FOLDER    =pathlib.Path("/global/homes/a/aglover/equivariant_spmm/outputs")
FIGURES_FOLDER      =pathlib.Path("/global/homes/a/aglover/equivariant_spmm/figures")

Project = Literal[
    'e3nn',
    'cuEquivariance', 
    'ours'
]

impl_to_project_map : dict[str, Project] = defaultdict(
    lambda: 'ours', 
    { 
    'E3NNTensorProduct' : 'e3nn',
    'CUETensorProduct' : 'cuEquivariance',
    }
)

project_to_color_map : dict[Project, str] = {
    'e3nn' : 'lightblue',
    'cuEquivariance' : 'orange',
    'ours' : 'green'
}

def get_latest_experiment_path() -> pathlib.Path:
    latest_experiment = max(
        (folder for folder in BENCHMARK_FOLDER.iterdir() if folder.is_dir() and folder.name.isdigit()),
        key=lambda x: int(x.name)
    )
    return latest_experiment