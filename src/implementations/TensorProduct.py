class TensorProduct:
    '''
    Each class implementation of a TensorProduct uses
    a different internal representation, which it can
    initialize uniquely. 
    '''
    def __init__(self, L1, L2, L3):
        self.internal = None
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3

    def exec_tensor_product_cpu(self, L1_in, L2_in, L3_out):
        self.internal.exec_tensor_product_cpu(L1_in, L2_in, L3_out) 

    def get_row_length(self, mode):
        return self.internal.get_row_length(mode)