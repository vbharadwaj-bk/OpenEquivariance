import pickle
import numpy as np
from sklearn.neighbors import radius_neighbors_graph

def cif_to_molecular_graph(cif_file, cp, radius=3.5):
    with open(f'../../data/cif_files/{cif_file}', 'r') as f:
        print("Started reading file...")
        lines = f.readlines()
        print("Finished reading file!")

        coords = []
        for line in lines:
            if line.startswith('ATOM'):
                parts = line.split()
                coords.append([float(parts[cp[0]]), float(parts[cp[1]]), float(parts[cp[2]])])

        coords = np.array(coords)

        print("Starting radius neighbors calculation...")
        A = radius_neighbors_graph(coords, radius, mode='connectivity',
                           include_self=False) 
        print(f"Finished radius neighbors calculation, found {A.nnz} nonzeros.") 


if __name__=='__main__':
    cif_to_molecular_graph('hiv_capsid.cif', (10, 11, 12), radius=3.5)  
