import numpy as np
import matplotlib.pyplot as plt

def create_nonzero_pattern(L1, L2, L3, slice_dim):
    tensor = np.zeros(2 * L1 + 1, 2 * L2 + 1, 2 * L3 + 1)

    for slice_idx in range(tensor.shape[slice_dim]):
        if slice_dim == 0:
            slice = tensor[slice_idx, :, :]
        elif slice_dim == 1: 
            slice = tensor[:, slice_idx, :]
        else:
            slice = tensor[:, :, slice_idx]

        slice += np.diag(1.0, 0)

    return tensor

if __name__=='__main__':
    slice_dim = 0
    slice_idx = 2

    pattern_tensor = create_nonzero_pattern(5, 5, 5, slice_dim)

    if slice_dim == 0:
        slice = pattern_tensor[slice_idx, :, :]
    elif slice_dim == 1: 
        slice = pattern_tensor[:, slice_idx, :]
    else:
        slice = pattern_tensor[:, :, slice_idx]

    plt.imshow(slice)

