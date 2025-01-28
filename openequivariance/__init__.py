import openequivariance.extlib
from pathlib import Path

dp = openequivariance.extlib.DeviceProp(0)

package_root = str(Path(__file__).parent.parent)

from openequivariance.implementations.e3nn_lite import TPProblem, Irreps
from openequivariance.implementations.TensorProduct import TensorProduct 
from openequivariance.implementations.convolution.TensorProductConv import TensorProductConv 

# For compatibility with first version of README
from openequivariance.implementations.convolution.LoopUnrollConv import LoopUnrollTP
from openequivariance.implementations.convolution.LoopUnrollConv import LoopUnrollConv