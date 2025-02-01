import os, warnings, logging
from pathlib import Path

oeq_root = str(Path(__file__).parent.parent)

build_ext = True 
candidates = [f for f in os.listdir(oeq_root + '/extlib') 
                if f.startswith('kernel_wrapper')]

if len(candidates) == 1:
    build_ext = False 

kernel_wrapper = None
if not build_ext: 
    from openequivariance.extlib.kernel_wrapper import * 
else:
    from setuptools import setup
    from torch.utils.cpp_extension import library_paths, include_paths

    sources = [
        'hip_wrapper.cpp'
    ]
    include_dirs = ['util']

    sources = [oeq_root + '/extension/' + src for src in sources]
    include_dirs = [oeq_root + '/extension/' + d for d in include_dirs] + include_paths('cuda')

    extra_link_args = ['-Wl,--no-as-needed', 
                    '-lhiprtc']
    
    try:
        cuda_libs = library_paths('cuda')[1]
        if os.path.exists(cuda_libs + '/stubs'):
            extra_link_args.append('-L' + cuda_libs + '/stubs')
    except Exception as e:
        logging.info(str(e))

    global torch
    import torch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kernel_wrapper = torch.utils.cpp_extension.load("hip_wrapper",
            sources,
            extra_cflags=["-O3"],
            extra_include_paths=include_dirs,
            extra_ldflags=extra_link_args)

    from hip_wrapper import *
