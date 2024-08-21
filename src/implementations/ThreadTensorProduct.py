import numpy as np

class ThreadTensorProduct:
    def __init__(self, L1, L2, L3):
        self.internal = TensorProductInternal(L1, L2, L3)

    def execute_tensor_product(self, L1_in, L2_in, L3_out):
        self..exec_tensor_product_cpu(L1_in, L2_in, L3_out) 

