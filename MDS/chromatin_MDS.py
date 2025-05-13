
import time
import numpy as np

from scipy.sparse.csgraph import floyd_warshall

from chromatin_utils import CLASSICAL_MDS, AtomAtomDistance
from chromatin_utils import write_mmcif, SMACOF, _smacof_single

CONSECUTIVE = 1.0 #angstrom
LOOP = 1.0 #angstrom

MAXW = 10**6

RAD = {-2:20,
       -1:20,
       0:40,
       1:40,
       2:40}

Ms_file = "RepliSage_Tinit_1.2_ht0/metadata/Ms.npy"
Ns_file = "RepliSage_Tinit_1.2_ht0/metadata/Ns.npy"
Cs_file = "RepliSage_Tinit_1.2_ht0/metadata/spins.npy"


Ms = np.load(Ms_file)
Ns = np.load(Ns_file)
Cs = np.load(Cs_file)

N  = Cs.shape[0]
Nt = Ms.shape[1] 
Nc = 5
Dc = {x-2:N+x for x in range(5)}

X = np.random.random((N, 3))

for t in range(Nt):

    t0 = time.time()
    D = np.zeros((N, N))
    W = np.ones((N, N))

    #D = np.zeros((N, N))
    #W = np.ones((N, N))

    #Consecutive (i,i+1) pairs
    for i in range(N-1):
        D[i,i+1] = CONSECUTIVE
        D[i+1,i] = CONSECUTIVE
        W[i,i+1] = 1000000
        W[i+1,i] = 1000000

    #D[0,N-1] = CONSECUTIVE
    #D[N-1,0] = CONSECUTIVE
    #W[0,N-1] = 1000
    #W[N-1,0] = 1000

    for v,w in zip(Ms[:,t],Ns[:,t]):
        if v >= 0 and w >= 0: 
            D[v,w] = LOOP
            D[w,v] = LOOP
            W[v,w] = 1000
            W[w,v] = 1000

    D = floyd_warshall(D, directed = False)

    for i in range(N):
        for j in range(N):
            if i != j:
                if Cs[i,t] == Cs[j,t]:
                    if D[i, j] > RAD[Cs[i,t]]:
                        D[i, j] = RAD[Cs[i,t]]
                        W[i, j] = 0.001

    D = floyd_warshall(D, directed = False)

    #for i,c in enumerate(Cs[:,t]):
    #
    #    D[i,Dc[c]] = RAD[c]
    #    D[Dc[c],i] = RAD[c]

    print(np.max(D))
    print(list(D[0]))
    #print(D)
    #print(W)
    #X = CLASSICAL_MDS(D)

    for _ in range(10):
        X = _smacof_single(D,init=X,weights=W)#_smacof_single(D,init=X,weights=W) #SMACOF(D)[:N] # CLASSICAL_MDS(D)[:N]

    print(t, round(time.time() - t0, 3))

    write_mmcif(X[:N], cif_file_name = "cif5/test_{}.cif".format(t))

    #print(X[:20])

    #for i in range(N):
    #    print(AtomAtomDistance(X[i],X[i+1]))
    
    break

        
