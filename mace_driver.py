import sys, json, time, pathlib
sys.path.append('mace_dev')

import argparse
import logging
from pathlib import Path

import ase.io
import numpy as np
import torch
from e3nn import o3
from mace import data, modules, tools
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.tools import torch_geometric
from torch.utils.benchmark import Timer
from mace.calculators import mace_mp
from torch.profiler import profile, record_function, ProfilerActivity

import warnings
warnings.filterwarnings("ignore")

try:
    import cuequivariance as cue  # pylint: disable=unused-import
    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

def analyze_trace(trace_file):
    trace = None
    with open(trace_file, "r") as f:
        trace = json.load(f)

    total = 0
    cgtp_fwd_bwd = 0
    reduce_by_key = 0
    other_kernels = 0

    for event in trace["traceEvents"]:
        if "args" in event and "stream" in event["args"]:
            total += event["dur"]

            if "forward" in event["name"] \
                or "backward" in event["name"] \
                or "TensorProductUniform1dKernel" in event["name"]:
                cgtp_fwd_bwd += event["dur"]

            elif "_scatter_gather_elementwise_kernel" in event["name"]:
                reduce_by_key += event["dur"]
            else:
                other_kernels += event["dur"]

    return { 
        "total_cuda_ms": total / 1000.,
        "cgtp_fwd_bwd_ms": cgtp_fwd_bwd / 1000.,
        "reduce_by_key_ms": reduce_by_key / 1000.,
        "other_kernels_ms": other_kernels / 1000.
    }

def create_model(hidden_irreps, max_ell, cueq_config=None):
    table = tools.AtomicNumberTable([8, 82, 53, 55])
    model_config = {
        "r_max": 6.0,
        "num_bessel": 8,
        "num_polynomial_cutoff": 6,
        "max_ell": max_ell,
        "interaction_cls": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "interaction_cls_first": modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        "num_interactions": 2,
        "num_elements": len(table),
        "hidden_irreps": o3.Irreps(hidden_irreps),
        "MLP_irreps": o3.Irreps("16x0e"),
        "gate": torch.nn.functional.silu,
        "atomic_energies": torch.ones(len(table)),
        "avg_num_neighbors": 8,
        "atomic_numbers": table.zs,
        "correlation": 3,
        "radial_type": "bessel",
        "num_elements": 4,
        "cueq_config": cueq_config,
        "atomic_inter_scale": 1.0,
        "atomic_inter_shift": 0.0,
    }
    return modules.ScaleShiftMACE(**model_config)

def benchmark_model(model, batch, num_iterations=100, warmup=100, label=None, output_folder=None):
    def run_inference():
        out = model(batch,training=True)
        torch.cuda.synchronize()
        return out

    # Warmup
    for _ in range(warmup):
        run_inference()

    # Benchmark
    timer = Timer(
        stmt="run_inference()",
        globals={
            "run_inference": run_inference,
        },
    )
    warm_up_measurement = timer.timeit(num_iterations)
    measurement = timer.timeit(num_iterations)

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            run_inference() 

    trace_file = str(output_folder / f"traces/{label}_trace.json")
    prof.export_chrome_trace(trace_file)

    with open(output_folder / f"{label}.json", "w") as f:
        json.dump({
            "time_ms_mean": measurement.mean * 1000, 
            "label": label,
            "cuda_time_profile": analyze_trace(trace_file)
        }, f, indent=4) 

    print(run_inference())

    return measurement

def load_fast_tp(source_model, device):
    from mace.tools.scripts_utils import extract_config_mace_model
    config = extract_config_mace_model(source_model)
    config["fast_tp_config"] = {"enabled": True, "conv_fusion": "deterministic"}
    target_model = source_model.__class__(**config).to(device)

    source_dict = source_model.state_dict()
    target_dict = target_model.state_dict()

    # To migrate fast_tp, we should transfer all keys
    for key in target_dict:
        if key in source_dict:
            target_dict[key] = source_dict[key]

    target_model.load_state_dict(target_dict)
    return target_model.to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xyz_file", type=str, help="Path to xyz file")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument("--max_ell", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_irreps", type=str, default="128x0e + 128x1o + 128x2e")
    parser.add_argument("--output_folder", type=str, default=None)
    args = parser.parse_args()

    output_folder = args.output_folder

    if output_folder is None:
        millis_since_epoch = round(time.time() * 1000)
        output_folder = pathlib.Path(f'outputs/{millis_since_epoch}')
    else:
        output_folder = pathlib.Path(output_folder)

    for dtype_str, dtype in [   ("f32", torch.float32),
                                ("f64", torch.float64)]:
        torch.set_default_dtype(dtype)
        device = torch.device(args.device)
        hidden_irreps = o3.Irreps(args.hidden_irreps)

        # Create dataset
        atoms_list = ase.io.read(args.xyz_file, index=":")
        #table = tools.AtomicNumberTable(list(set(np.concatenate([atoms.numbers for atoms in atoms_list]))))
        table = tools.AtomicNumberTable([6, 82, 53, 55])
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[data.AtomicData.from_config(
                data.config_from_atoms(atoms),
                z_table=table,
                cutoff=6.0
            ) for atoms in atoms_list],
            batch_size=min(len(atoms_list), args.batch_size),
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(device)
        batch_dict = batch.to_dict()

        output_folder.mkdir(parents=True, exist_ok=True)

        traces_folder = output_folder / "traces"
        traces_folder.mkdir(parents=True, exist_ok=True) 

        print("\nBenchmarking Configuration:")
        print(f"Number of atoms: {len(atoms_list[0])}")
        print(f"Number of edges: {batch['edge_index'].shape[1]}")
        print(f"Batch size: {min(len(atoms_list), args.batch_size)}")
        print(f"Device: {args.device}")
        print(f"Hidden irreps: {hidden_irreps}")
        print(f"Number of iterations: {args.num_iters}\n")

        # Test without CUET
        model_e3nn = create_model(hidden_irreps, args.max_ell).to(device)
        #model_e3nn = mace_mp(model="large", device="cuda", default_dtype="float64")
        measurement_e3nn = benchmark_model(model_e3nn, batch_dict, args.num_iters, label=f"e3nn_{dtype_str}", output_folder=output_folder)
        print(f"E3NN Measurement:\n{measurement_e3nn}")

        model_fast_tp = load_fast_tp(model_e3nn, device)  
        measurement_fast_tp = benchmark_model(model_fast_tp, batch_dict, args.num_iters, label=f"ours_{dtype_str}", output_folder=output_folder)
        print(f"\nFast TP (ours) Measurement:\n{measurement_fast_tp}")
        print(f"\nSpeedup: {measurement_e3nn.mean / measurement_fast_tp.mean:.2f}x")

        model_cueq = run_e3nn_to_cueq(model_e3nn)
        model_cueq = model_cueq.to(device)
        measurement_cueq = benchmark_model(model_cueq, batch_dict, args.num_iters, label=f"cuE_{dtype_str}", output_folder=output_folder)
        print(f"\nCUET Measurement:\n{measurement_cueq}")
        print(f"\nSpeedup: {measurement_e3nn.mean / measurement_cueq.mean:.2f}x")

if __name__ == "__main__":
    main()