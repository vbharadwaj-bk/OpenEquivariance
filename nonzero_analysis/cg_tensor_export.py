import pickle
import e3nn
LMAX = 10
cg_tensor_dict = {}
for l1 in range(LMAX + 1):
    for l2 in range(LMAX + 1):
        for l3 in range(abs(l1-l2), l1 + l2 + 1):
            cg_tensor_dict[(l1, l2, l3)] = e3nn.o3.wigner_3j(l1,l2,l3).numpy()

with open('../data/CG_tensors.pickle', 'wb') as handle:
    pickle.dump(cg_tensor_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)