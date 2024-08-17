import numpy as np
import matplotlib.pyplot as plt

def fill_diagonal(slice, offset, direction):
    assert(direction == 1 or direction == -1)
    rows, cols = slice.shape
    center = rows // 2 + offset, cols // 2

    cur = center
    while 0 <= cur[0] < rows and 0 <= cur[1] < cols:
        slice[cur[0], cur[1]] = 1.0
        cur = cur[0] + 1, cur[1] + direction

    cur = center
    while 0 <= cur[0] < rows and 0 <= cur[1] < cols:
        slice[cur[0], cur[1]] = 1.0
        cur = cur[0] - 1, cur[1] - direction 



def create_nonzero_pattern(L1, L2, L3, slice_dim):
    L = [L1, L2, L3]
    tensor = np.zeros((2 * L1 + 1, 2 * L2 + 1, 2 * L3 + 1))

    for slice_idx in range(tensor.shape[slice_dim]):
        if slice_dim == 0:
            slice = tensor[slice_idx, :, :]
        elif slice_dim == 1: 
            slice = tensor[:, slice_idx, :]
        else:
            slice = tensor[:, :, slice_idx]

        m = slice_idx - L[slice_dim]
        fill_diagonal(slice, m, 1)
        fill_diagonal(slice, -m, 1)


    return tensor

if __name__=='__main__':
    slice_dim = 0
    slice_idx = 3

    pattern_tensor = create_nonzero_pattern(L1=5, L2=5, L3=2, slice_dim=slice_dim)

    if slice_dim == 0:
        slice = pattern_tensor[slice_idx, :, :]
    elif slice_dim == 1: 
        slice = pattern_tensor[:, slice_idx, :]
    else:
        slice = pattern_tensor[:, :, slice_idx]

    plt.imshow(slice)
    plt.show()
