import json, os, pathlib

def load_benchmarks(path : pathlib.Path, subfolder=None):
    assert isinstance(path, pathlib.Path)
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
