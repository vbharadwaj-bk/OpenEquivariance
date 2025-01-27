import os, tempfile
from pathlib import Path
from openequivariance.benchmark.logging_utils import getLogger
logger = getLogger()

package_root = str(Path(__file__).parent.parent)
oeq_root = str(Path(__file__).parent)

build_ext = False 
candidates = [f for f in os.listdir(oeq_root + '/extlib') 
                if f.startswith('kernel_wrapper')]

if len(candidates) == 0:
    logger.info("No extension found. Building extension.")
    build_ext = True
elif len(candidates) > 1:
    logger.info("Multiple extensions found. Deleting all and rebuilding extension.")
    for c in candidates:
        os.remove(oeq_root + '/extlib/' + c)
    build_ext = True

if build_ext:
    with tempfile.TemporaryDirectory() as tmpdirname:
        from setuptools import setup
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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

        setup(name='kernel_wrapper',
            ext_modules=[
                CUDAExtension(
                        name='kernel_wrapper',
                        include_dirs=include_dirs,
                        sources=sources,
                        extra_compile_args={'cxx': ['-O3']},
                        extra_link_args=['-Wl,--no-as-needed', 
                                            '-lcuda',
                                            '-lnvrtc',
                                            '-lcudart'])
            ],
            cmdclass={
                'build_ext': BuildExtension 
            },
            script_args=['build_ext', 
                '--build-lib', oeq_root + '/extlib',
                '--build-temp', tmpdirname])

        print("Finished compiling extension!")

import torch # Needed for libc10.so for torch compiled extensions
from openequivariance.implementations.e3nn_lite import TPProblem, Irreps
from openequivariance.implementations.LoopUnrollTP import LoopUnrollTP 
from openequivariance.implementations.convolution.LoopUnrollConv import LoopUnrollConv
from openequivariance.implementations.e3nn_lite import TPProblem, Irreps