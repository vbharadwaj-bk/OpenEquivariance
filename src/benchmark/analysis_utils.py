import json, os, pathlib
import matplotlib.pyplot as plt

def load_benchmarks(path : pathlib.Path, subfolder=None):
    if subfolder is None:
        folders = os.listdir(path)
        subfolder = sorted(folders)[-1]
        
    benchmarks = []
    metadata = None
    
    files = os.listdir(path / subfolder)
    for file in files:
        with open( path / subfolder / file , "r") as f:
            if file != "metadata.json":
                benchmarks.append(json.load(f))
                benchmarks[-1]["filename"] = file
            else:
                metadata = json.load(f)
                metadata["folder"] = subfolder
                
    return benchmarks, metadata

def filter(benchmarks, base, match_one=True):
    filtered_results = []
    for benchmark in benchmarks:
        matched = True
        for key in base:
            if benchmark[key] != base[key]:
                matched = False
        
        if matched:
            filtered_results.append(benchmark)
        
    if len(filtered_results) == 0:
        print("WARNING: Filter matched no experiments")
        return None
    
    if len(filtered_results) > 1 and match_one:
        print("Error, matched more than one experiment:")
        for experiment in filtered_results:
            print(experiment["filename"])
        assert(False)
    
    if match_one:
        filtered_results = filtered_results[0]
    
    return filtered_results

def grouped_barchart(data: dict, ax, bar_width=1.0, group_spacing=3.0, 
        rotate_xlabels=True, 
        colormap=None, 
        label=True,
        edgecolor='k',
        edgewidth=1.0,
        bar_height_fontsize=7,
        xticklabel=True):
    '''
    data is a dictionary with the following structure:
    xtick_label -> dict(bar_label->value)
    
    Example Use:
        
        data = {
            "Adelie": {
                'Bill Depth': 18.35, 'Bill Length': 38.79, 'Flipper Length': 89.95
            },
            "Chinstrap": {
                'Bill Depth': 18.43, 'Bill Length': 48.83, 'Flipper Length': 195.82
            },
             "Gentoo": {
                'Bill Depth': 14.98, 'Bill Length': 47.50, 'Flipper Length': 217.19
            }
        }
    
        fig, ax = plt.subplots()
        grouped_barchart(data, ax)
        ax.legend()
    '''
    xtick_labels = list(data.keys())
    color_keys = {} # Maps bars to colors
    
    coord = 0.0
    xticks = []
    
    if colormap is None:
        colormap = plt.get_cmap('tab10')
    
    for bar_group in data:
        bars = data[bar_group]
        xticks.append(coord + len(bars) * bar_width / 2.0)
        
        for i, bar_label in enumerate(bars):
            is_first_label = False
            if bar_label not in color_keys:
                if isinstance(colormap, dict):
                    color_keys[bar_label] = colormap[bar_label]
                else:
                    color_keys[bar_label] = colormap(len(color_keys))
                is_first_label = True

            rects = None    
            if is_first_label:      
                rects = ax.bar(coord + bar_width * (i + 0.5), bars[bar_label], label=bar_label, width=bar_width, 
                        color=color_keys[bar_label], edgecolor=edgecolor, linewidth=edgewidth)
            else:
                rects = ax.bar(coord + bar_width * (i + 0.5), bars[bar_label], width=bar_width, 
                        color=color_keys[bar_label],
                        edgecolor=edgecolor, linewidth=edgewidth)

            if bar_height_fontsize > 0:
                ax.bar_label(rects, padding=3, fontsize=bar_height_fontsize) 

        coord += len(bars) * bar_width + group_spacing

    if xticklabel: 
        if rotate_xlabels:
            ax.set_xticks(xticks, labels=xtick_labels, rotation=45, ha='right')
        else:
            ax.set_xticks(xticks, labels=xtick_labels)
    else: 
        ax.set_xticks(xticks) 

def barchart(xlabels, heights, ax, bar_width=1.0, spacing=3.0, rotate_xlabels=True, colormap=None, data_label="_", 
        edgecolor='k', edgewidth=1.0, bar_height_fontsize=10):
    '''
    Usage:

    fig, ax = plt.subplots()
    barchart(["alpha", "beta", "gamma"], [5, 7, 3], ax, data_label="Test")
    '''
    assert(len(xlabels) == len(heights))
    
    data = {}
    for i, xlabel in enumerate(xlabels):
        data[xlabel] = {data_label: heights[i]}
        
    grouped_barchart(data, ax, bar_width, spacing, rotate_xlabels, colormap, edgecolor, edgewidth, bar_height_fontsize, label=True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def roofline_plot(draw_bounds, cpu_roofs, mem_bottlenecks, AI_v, datapoints, compute_unit="TFLOP/s", mem_unit="TB/s", fig_ratio=2, fig_dimension=7):
    '''
    Example Usage:

    # Architecture-specific roofs
    cpu_roofs = {"A100 FP32 Peak": 19.5}
    mem_bottlenecks = {"HBM2": 1.555}
    AI_v = {"": 12.54}

    # Datapoints
    datapoints = [{"AI": 7.24, "throughput": 5.4, "label": "I'm pretty cool", "marker": "x", "color": "r"}]
    draw_bounds = {"xmin": 1.0, "xmax": 25, "ymin": 0.4, "ymax": 25}
    fig, ax = roofline_plot(draw_bounds, cpu_roofs, mem_bottlenecks, AI_v, datapoints, fig_ratio=1.8, fig_dimension=5)

    '''
    xmin, xmax, ymin, ymax = draw_bounds["xmin"], draw_bounds["xmax"], draw_bounds["ymin"], draw_bounds["ymax"]
    
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)

    ax.grid(color="#dddddd", zorder=-1)
    ax.set_xlabel("Arithmetic Intensity [FLOP/Byte]", fontsize=15)
    ax.set_ylabel(f"Performance [{compute_unit}]", fontsize=15)

    ##########################################################
    # Set size for explicitly setting axes widths/heights
    def set_size(w, h, ax=None):
        """ w, h: width, height in inches """
        if not ax: ax = plt.gca()
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w) / (r - l)
        figh = float(h) / (t - b)
        ax.figure.set_size_inches(figw, figh)

    ##########################################################

    # Axis sizes
    xlogsize = float(np.log10(xmax / xmin))
    ylogsize = float(np.log10(ymax / ymin))
    m = xlogsize / ylogsize

    print(f"Axis limits: 10^[({np.log10(xmin)} -> {np.log10(xmax)}) x ({np.log10(ymin)} -> {np.log10(ymax)})]")
    print(f"Plot logarithmic ratio: {m}\n")

    max_roof = max([throughput for throughput in cpu_roofs.values()])
    max_slope = max([slope for slope in mem_bottlenecks.values()])

    # Draw slopes
    for mem_roof, slope in mem_bottlenecks.items():
        print(f"slope\t\"{mem_roof}\"\t\t{slope} {mem_unit}")
        y = [0, max_roof]
        x = [float(yy) / slope for yy in y]
        ax.loglog(x, y, linewidth=1.0, linestyle='-.', color="grey", zorder=10)

        # Label
        xpos =  xmin * (10 ** (xlogsize * 0.04))
        ypos = 1.05 * xpos * slope
        if ypos < ymin:
            ypos = ymin * (10 ** (ylogsize * 0.02))
            xpos = ypos / slope
        ax.annotate(
            f"{mem_roof}: {slope} {mem_unit}", (xpos, ypos),
            rotation=np.arctan(m / fig_ratio) * 180 / np.pi,
            fontsize=11, ha="left", va='bottom', color="grey"
        )

    # Draw roofs
    for roof, value in cpu_roofs.items():
        print(f"roof\t\"{roof}\"\t\t{value} {compute_unit}")
        x = [value / max_slope, xmax * 10]
        ax.loglog(x, [value] * len(x), linewidth=1.0, linestyle='-.', color="grey", zorder=10)
        ax.text(
            xmax / (10 ** (xlogsize * 0.01)), value * (10 ** (ylogsize * 0.01)),
            f"{roof}: {value} {compute_unit}", ha="right", fontsize=11, color="grey"
        )

    # Benchmarks
    for benchmark, AI in AI_v.items():
        print(f"benchmark\t\"{benchmark}\"\t\t{AI} FLOPs/Byte")
        plt.axvline(x=AI, dashes=[10, 10, 3, 10], linewidth=0.4, color="#aaaaaa")
        ax.text(AI / 1.15, ymin * 1.24, benchmark, fontsize=12, rotation=90, va="bottom", color="#888888")

    # Datapoints
    for point in datapoints:
        AI = point["AI"]
        if isinstance(AI, str):
            AI = AI_v[AI]
        ax.scatter(AI, point["throughput"], label=point["label"], marker=point["marker"], zorder=100, c=point["color"])

    # Set axes limits and layout
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    set_size(fig_dimension * fig_ratio, fig_dimension)
    
    return fig, ax
