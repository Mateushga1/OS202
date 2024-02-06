import numpy as np
from mpi4py import MPI

# Initialisation de l'environnement MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Dimension du problème (peut être changé)
dim = 120

# Calcul du nombre de colonnes par tâche
Nloc = dim // size

# Initialisation de la matrice sur chaque tâche
local_A = np.zeros((dim, Nloc))

# Calcul de la partie de la matrice pour chaque tâche
for i in range(Nloc):
    local_A[:, i] = [(j + rank * Nloc + i) % dim + 1. for j in range(dim)]

# Initialisation du vecteur u sur chaque tâche
u = np.array([i + 1. for i in range(dim)])

# Calcul local du produit matrice-vecteur
local_v = np.dot(local_A.T, u)

# Collecte des résultats de toutes les tâches
v = np.zeros(dim)
comm.Allgather(local_v, v)

# Affichage du résultat sur chaque tâche
print(f"Process {rank}: v = {local_v}")

# Affichage du vecteur résultat sur la première tâche
if rank == 0:
    print(f"\nv (Résultat global) = {v}")
