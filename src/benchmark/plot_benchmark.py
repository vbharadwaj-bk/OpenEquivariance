import os, pathlib, importlib
import numpy as np 
import matplotlib.pyplot as plt
from benchmark_utils import load_benchmarks, filter

def main():
    outputs_path = pathlib.Path(__file__).parent.parent.parent / "outputs"

    figures_path = pathlib.Path(__file__).parent.parent.parent / "figures"

    benchmarks, metadata = load_benchmarks(outputs_path)

    implementations = metadata["implementations"]
    configs = metadata["configs"]
    throughputs = []
    throughput_errs = []

    e_bandwidths = []
    e_bandwidth_errs = []

    for implementation in implementations:
        throughputs = []
        throughput_errs = []
        e_bandwidths = []
        e_bandwidth_errs = []
        for config in configs:
            filter_criteria = {"config": config, "name":implementation}
            result = filter(benchmarks, filter_criteria)
            throughputs.append(np.mean(result['benchmark']["throughputs_gflops"]))
            throughput_errs.append(np.std(result['benchmark']["throughputs_gflops"]) * 3)
            e_bandwidths.append(np.mean(result['benchmark']['bandwidth_gbps_rough']))
            e_bandwidth_errs.append(np.std(result['benchmark']['bandwidth_gbps_rough']) * 3)
    
        plt.figure()
        fig_title = f"Throughput, {implementation}"
        plt.bar([str(config) for config in configs], throughputs)
        plt.xlabel("L1, L2, L3")
        plt.ylabel("GFLOPs")
        plt.grid(True)
        plt.title(fig_title)
        plt.show()
        plt.savefig(figures_path / fig_title.lower().replace(" ","_"))

        plt.figure()
        fig_title = f"Effective Bandwidth, {implementation}"
        plt.bar([str(config) for config in configs], e_bandwidths)
        plt.xlabel("L1, L2, L3")
        plt.ylabel("GBPs")
        plt.grid(True)
        plt.title(fig_title)
        plt.show()
        plt.savefig(figures_path / fig_title.lower().replace(" ","_"))

if __name__ == '__main__':
    main()