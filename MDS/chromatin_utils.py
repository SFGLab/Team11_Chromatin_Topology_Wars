
import numpy as np


from scipy.spatial.distance import squareform, pdist
from scipy.linalg import eigh
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances

mmcif_atomhead = """data_nucsim
# 
_entry.id nucsim
# 
_audit_conform.dict_name       mmcif_pdbx.dic 
_audit_conform.dict_version    5.296 
_audit_conform.dict_location   http://mmcif.pdb.org/dictionaries/ascii/mmcif_pdbx.dic 
# ----------- ATOMS ----------------
loop_
_atom_site.group_PDB 
_atom_site.id 
_atom_site.type_symbol 
_atom_site.label_atom_id 
_atom_site.label_alt_id 
_atom_site.label_comp_id 
_atom_site.label_asym_id 
_atom_site.label_entity_id 
_atom_site.label_seq_id 
_atom_site.pdbx_PDB_ins_code 
_atom_site.Cartn_x 
_atom_site.Cartn_y 
_atom_site.Cartn_z
"""

mmcif_connecthead = """#
loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_label_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_label_seq_id
_struct_conn.ptnr2_label_atom_id
"""

def write_mmcif(points,cif_file_name='LE_init_struct.cif'):
    atoms = ''
    n = len(points)
    for i in range(0,n):
        x = points[i][0]
        y = points[i][1]
        try:
            z = points[i][2]
        except IndexError:
            z = 0.0
        atoms += ('{0:} {1:} {2:} {3:} {4:} {5:} {6:}  {7:} {8:} '
                '{9:} {10:.3f} {11:.3f} {12:.3f}\n'.format('ATOM', i+1, 'D', 'CA',\
                                                            '.', 'ALA', 'A', 1, i+1, '?',\
                                                            x, y, z))

    connects = ''
    for i in range(0,n-1):
        connects += f'C{i+1} covale ALA A {i+1} CA ALA A {i+2} CA\n'

    # Save files
    ## .pdb
    cif_file_content = mmcif_atomhead+atoms+mmcif_connecthead+connects

    with open(cif_file_name, 'w') as f:
        f.write(cif_file_content)


def get_coordinates_cif(file):
    '''
    It returns the corrdinate matrix V (N,3) of a .pdb file.
    The main problem of this function is that coordiantes are not always in 
    the same column position of a .pdb file. Do changes appropriatelly,
    in case that the data aren't stored correctly. 
    
    Input:
    file (str): the path of the .cif file.
    
    Output:
    V (np.array): the matrix of coordinates
    '''
    V = list()
    
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("ATOM"):
                columns = line.split()
                x = eval(columns[10])
                y = eval(columns[11])
                z = eval(columns[12])
                V.append([x, y, z])
    
    return np.array(V)


# D is a (n x n) symmetric distance matrix
def CLASSICAL_MDS(D, n_components = 3):

    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K = -0.5 * H @ (D ** 2) @ H

    # Compute top n_components eigenvalues and eigenvectors
    eigvals, eigvecs = eigh(K, eigvals=(n - n_components, n - 1))

    # Coordinates in lower-dimensional space
    X = eigvecs * np.sqrt(eigvals)
    
    return X

def AtomAtomDistance(coords1, coords2):

    return pow(sum((c1 - c2)**2
                   for c1, c2 in zip(coords1, coords2)), 0.5)



def SMACOF(mat):

    embedding = MDS(n_components = 3,
                            dissimilarity='precomputed')
    transform = embedding.fit_transform(mat)

    return transform

def _smacof_single(dissimilarities, init=None, weights=None, max_iter=300, verbose=0, eps=1e-3, random_seed=None,
                   n_components = 3):
    
    """SMACOF algorithm implementation derived from scipy version but corrected for handling missing values

    Parameters
    ----------
    dissimilarities : ndarray, shape (n_samples, n_samples)
        Pairwise dissimilarities between the points. Can contains np.nan values
        to indicate missing values. Must be symmetric.
    init : ndarray, shape (n_samples, 2)
        initial configuration
    weights : ndarray, shape (n_samples, n_samples)
        Pairwise weights of dissimilarities (0 indiciates that this dissimilariyt
        shouldn't be taken into consideration, wheras non-zero positive value w_ij indicates
        strength of dissimilarity d_ij
    max_iter : int, default 300
        Maximum number of iterations of the SMACOF algorithm for a single run.
    verbose : int, optional, default: 0
        Level of verbosity.
    eps : float, optional, default: 1e-3
        Relative tolerance with respect to stress at which to declare
        convergence.
    random_seed : int
        Seed for randomness

    Returns
    -------
    X : ndarray, shape (n_samples, n_components)
        Coordinates of the points in a ``n_components``-space.
    stress : float
        The final value of the stress (sum of squared distance of the
        disparities and the distances for all constrained points).
    n_iter : int
        The number of iterations corresponding to the best stress."""


    n_samples = dissimilarities.shape[0]

    if random_seed is None:
        np.random.seed(random_seed)

    if weights is None:
        weights = np.ones(shape=(n_samples, n_samples), dtype=np.float)
        weights[np.isnan(dissimilarities)] = 0.0
        weights[np.diag_indices_from(weights)] = 1.0 #just ensure that we have ones on diagonal

    V = -weights
    V[np.diag_indices_from(V)] = np.sum(weights, 1) - weights[np.diag_indices_from(weights)]

    V_p = np.linalg.pinv(V)

    #n_components = 2
    X = init

    old_stress = None
    dis = euclidean_distances(X)
    stress = np.nansum(weights.ravel() * ((dis.ravel() - dissimilarities.ravel()) ** 2)) / 2

    for it in range(max_iter):
        # Update X using the Guttman transform
        dis[dis == 0] = 1e-8
        ratio = dissimilarities / dis
        B = -ratio
        B[np.arange(len(B)), np.arange(len(B))] += np.nansum(ratio, axis=1)

        # this is weighted formula 8.29 (page 191)
        X = V_p @ np.nan_to_num(B) @ np.nan_to_num(X)

        dis = euclidean_distances(X)
        stress = np.nansum(weights.ravel() * ((dis.ravel() - dissimilarities.ravel()) ** 2)) / 2

        if verbose >= 2:
            print('it: %d, stress %s' % (it, stress))

        if old_stress is not None:
            if(old_stress - stress) < eps:
                if verbose:
                    print('breaking at iteration %d with stress %s' % (it,
                                                                       stress))
                break

        old_stress = stress
    return X
