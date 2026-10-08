"""
Copied from tp1.ipynb + OBC
"""

from itertools import combinations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ================= bitstring primitives: one site index, one state ================
# Every one is a one-liner, but naming them fixes the argument order (site first,
# state second) and lets the Hamiltonian below read as physics, not bit twiddling.


def basis_states(L, N):
    """Occupation-number basis of the N-particle sector, as sorted bitstrings."""
    states = np.array(sorted(sum(1 << j for j in sites) for sites in combinations(range(L), N)), dtype=np.int64)
    return states, {int(s): i for i, s in enumerate(states)}


def occ(j, s):
    """Is site j occupied in the bitstring state s?

    Example: s = 13, bin(s) = '0b1101', so occ(0, s) = 1 but occ(1, s) = 0.
    """
    return (s >> j) & 1


def c(j, s):
    """Remove a particle at site j: if the jth bit is 1 it is set to 0, else nothing.

    For that `else' the caller needs an external guardrail -- see h_fermion_ring.
    """
    return s & ~(1 << j)


def cdag(j, s):
    """Add a particle at site j: if the jth bit is 0 it is set to 1, else nothing.

    For that `else' the caller needs an external guardrail -- see h_fermion_ring.
    """
    return s | (1 << j)


def count_below(j, s):
    """Number of particles on sites with index strictly below j (Jordan-Wigner string).

    (1 << j) is 2^j; (1 << j) - 1 is 2^j - 1 = 2^0 + ... + 2^(j-1), i.e. a mask of
    ones on every site below j.  For j = 3 that is 1000 - 1 = 111.  The & then keeps
    only those sites of s, and .count("1") counts the particles among them.
    """
    return bin(s & ((1 << j) - 1)).count("1")


def h_fermion(L, N, t=1.0, dtype=np.complex128):
    """Hopping matrix for spinless fermions on an L-site open chain, N-particle sector (CSR scipy matrix).
    Pure kinetic energy: -t sum_j c^dag_{j+1} c_j + h.c.), with Jordan-Wigner signs.
    """

    states, idxs = basis_states(L, N)
    dim = len(states)

    vals = []
    rows = []
    cols = []

    for s in states:
        a = idxs[s]
        for j in range(L - 1):  # OBC, so no hopping from L-1 to 0
            # k = (j + 1) % L
            k = j + 1
            for src, dst, amp in [(j, k, -t), (k, j, -t)]:
                if (not occ(src, s)) or occ(dst, s):
                    continue

                sign = -1 if count_below(src, s) % 2 else 1
                s1 = c(src, s)
                sign *= -1 if count_below(dst, s1) % 2 else 1
                s2 = cdag(dst, s1)

                b = idxs[s2]

                vals.append(amp * sign)
                rows.append(b)
                cols.append(a)

    H = sp.coo_matrix((vals, (rows, cols)), shape=(dim, dim), dtype=np.complex128).tocsr()
    H.sum_duplicates()
    return H.astype(dtype), states


def hubbard_operator(L, Nup, Ndn, t=1.0, U=0.0):
    """
    Matrix-free Fermi-Hubbard H in the (Nup, Ndn) sector.
    Returns (LinearOperator, double-occupancy diagonal, (D_up, D_dn)).
    """

    Hup, up_states = h_fermion(L, Nup, t)
    Hdn, dn_states = h_fermion(L, Ndn, t)

    Du = len(up_states)
    Dd = len(dn_states)

    # docc[a, b] = number of doubly occupied sites in |s_up[a]> (x) |s_dn[b]>
    docc = np.array([[bin(int(a) & int(b)).count("1") for b in dn_states] for a in up_states], dtype=np.float64).ravel()

    def matvec(x):
        P = x.reshape(Du, Dd)
        return (Hup @ P + (Hdn @ P.T).T).ravel() + (U * docc) * x

    return spla.LinearOperator((Du * Dd, Du * Dd), matvec=matvec, dtype=np.complex128), docc, (Du, Dd)


def get_E(L: int, Nup: int, Ndn: int, U: float, t: float):
    H, _, _ = hubbard_operator(L, Nup, Ndn, U=U, t=t)
    return spla.eigsh(H, k=1, which="SA", tol=0, return_eigenvectors=False)[0]
