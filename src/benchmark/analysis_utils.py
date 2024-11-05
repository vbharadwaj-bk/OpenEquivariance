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
        bar_height_fontsize=7):
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
        
    if rotate_xlabels:
        ax.set_xticks(xticks, labels=xtick_labels, rotation=45, ha='right')
    else:
         ax.set_xticks(xticks, labels=xtick_labels)
            

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
