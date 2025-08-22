import numpy as np

# ---------- tiny helper ----------
def _encode_with_lut(seqs, lut, dim):
    """
    seqs: list[str] of equal length
    lut: dict {'A': [...dim], 'C': [...], 'G': [...], 'T': [...]}
    returns: (batch, L, dim) float32; non-ACGT -> zeros
    """
    B = len(seqs)
    L = len(seqs[0])
    assert all(len(s) == L for s in seqs), "All sequences must be same length"

    out = []
    zero = [0.0] * dim
    for s in seqs:
        vecs = np.array([lut.get(b.upper(), zero) for b in s], dtype=np.float32)
        out.append(vecs)
    return np.stack(out, axis=0)

# ==================== 1) One-hot ====================
def dna_to_one_hot_real(seqs):
    """
    Standard ACGT one-hot.
    Output: (batch, L, 4)
    """
    lut = {
        'A': [1.0, 0.0, 0.0, 0.0],
        'C': [0.0, 1.0, 0.0, 0.0],
        'G': [0.0, 0.0, 1.0, 0.0],
        'T': [0.0, 0.0, 0.0, 1.0],
    }
    return _encode_with_lut(seqs, lut, dim=4)

# ==================== 2) Simplex-monomer ====================
def dna_to_simplex_monomer(seqs):
    """
    Classic 3D simplex (regular tetrahedron), ~±0.577.
    Output: (batch, L, 3)
    """
    lut = {
        'A': [ 0.577, -0.577, -0.577],
        'C': [-0.577,  0.577, -0.577],
        'G': [-0.577, -0.577,  0.577],
        'T': [ 0.577,  0.577,  0.577],
    }
    return _encode_with_lut(seqs, lut, dim=3)

# ==================== 3) Simplex-monomer (rotated) ====================
def dna_to_simplex_dimer(seqs):
    """
    Rotated 3D simplex (your LUT_1 set).
    Output: (batch, L, 3)
    """
    lut = {
        'A': [ 0.788675, -0.001155, -0.614948],
        'C': [-0.788675,  0.197254, -0.458016],
        'G': [-0.211325, -0.785674,  0.431891],
        'T': [ 0.211325,  0.588862,  0.641073],
    }
    return _encode_with_lut(seqs, lut, dim=3)

# ==================== 4) Scalar (1D) ====================
def dna_to_scalar(seqs):
    """
    Scalar per-base: A=0.25, C=0.50, G=0.75, T=1.00.
    Output: (batch, L, 1)
    """
    lut = {
        'A': [0.25],
        'C': [0.50],
        'G': [0.75],
        'T': [1.00],
    }
    return _encode_with_lut(seqs, lut, dim=1)

# ==================== 5) Scalar 2X (2D) ====================
def dna_to_simplex_monomer_scalar(seqs):
    """
    Two scalar channels (your 2X set):
      A -> [0.25, 1.00]
      C -> [0.50, 0.75]
      G -> [0.75, 0.50]
      T -> [1.00, 0.25]
    Output: (batch, L, 2)
    """
    lut = {
        'A': [0.25, 1.00],
        'C': [0.50, 0.75],
        'G': [0.75, 0.50],
        'T': [1.00, 0.25],
    }
    return _encode_with_lut(seqs, lut, dim=2)

# ==================== ROUTER ====================

def encode_sequence(seqs, method="one_hot"):
    if method == "one_hot":
        return dna_to_one_hot_real(seqs)
    elif method == "simplex_monomer":
        return dna_to_simplex_monomer(seqs)
    elif method == "simplex_dimer":
        return dna_to_simplex_dimer(seqs)
    elif method == "scalar":
        return dna_to_scalar(seqs)
    elif method == "simplex_monomer_scalar":
        return dna_to_simplex_monomer_scalar(seqs)
    else:
        raise ValueError(f"Unknown encoding method: {method}")


# ==================== DISPATCH COMPATIBILITY WRAPPER ====================

# Global encoding method (can be changed from outside this module)
ENCODING_METHOD = "one_hot"

# Preserve the original name, route to encode_sequence()
def dna_to_one_hot(seqs):
    return encode_sequence(seqs, method=ENCODING_METHOD)
