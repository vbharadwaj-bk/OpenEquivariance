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
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, library_paths

    sources = [
        'util/buffer.cpp',
        'util/jit.cpp',
        'util/device_prop.cpp',
        'tensorproducts/jit_tp.cpp',
        'convolution/jit_conv.cpp',
        'kernel_wrapper.cpp'
    ]
    include_dirs = ['util', 
                    'tensorproducts', 
                    'convolution']

    sources = [oeq_root + '/extension/' + src for src in sources]
    include_dirs = [oeq_root + '/extension/' + d for d in include_dirs]

    extra_link_args = ['-Wl,--no-as-needed', 
                    '-lcuda',
                    '-lnvrtc',
                    '-lcudart']
    
    try:
        cuda_libs = library_paths(cuda=True)[1]
        if os.path.exists(cuda_libs + '/stubs'):
            extra_link_args.append('-L' + cuda_libs + '/stubs')
    except Exception as e:
        logging.info(str(e))

    global torch
    import torch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kernel_wrapper = torch.utils.cpp_extension.load("kernel_wrapper",
            sources,
            extra_cflags=["-O3"],
            extra_include_paths=include_dirs,
            with_cuda=True,
            extra_ldflags=extra_link_args)

    from kernel_wrapper import *