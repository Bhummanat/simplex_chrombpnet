import numpy as np

# ==================== ONE-HOT ENCODING ====================

def dna_to_one_hot_real(seqs):
    """
    Converts a list of DNA ("ACGT") sequences to one-hot encodings.
    Each sequence becomes a matrix of shape (L, 4), ordered by ACGT.
    Unknown bases become [0, 0, 0, 0].
    Output shape: (batch, L, 4)
    """
    seq_len = len(seqs[0])
    assert np.all(np.array([len(s) for s in seqs]) == seq_len)

    seq_concat = "".join(seqs).upper() + "ACGT"
    one_hot_map = np.identity(5)[:, :-1].astype(np.int8)

    base_vals = np.frombuffer(bytearray(seq_concat, "utf8"), dtype=np.int8)
    base_vals[~np.isin(base_vals, np.array([65, 67, 71, 84]))] = 85  # non-ACGT → 'N'

    _, base_inds = np.unique(base_vals, return_inverse=True)
    return one_hot_map[base_inds[:-4]].reshape((len(seqs), seq_len, 4))


# ==================== SIMPLEX MONOMER ENCODING ====================

def create_mono_lut():
    return np.array([
        [1, -1, -1],    # A
        [-1, 1, -1],    # C
        [-1, -1, 1],    # G
        [1, 1, 1],      # T
    ], dtype=np.float32)

def base_to_index_array(seq_concat):
    base_vals = np.frombuffer(bytearray(seq_concat, "utf8"), dtype=np.int8)
    base_vals[~np.isin(base_vals, np.array([65, 67, 71, 84]))] = 85  # N = 85
    base_map = {65: 0, 67: 1, 71: 2, 84: 3, 85: 4}
    return np.vectorize(base_map.get)(base_vals)

def dna_to_simplex_monomer(seqs):
    mono_lut = create_mono_lut()
    seq_len = len(seqs[0])
    assert all(len(s) == seq_len for s in seqs)

    seq_concat = "".join(seqs).upper()
    idxs = base_to_index_array(seq_concat)
    idxs = idxs.reshape(len(seqs), seq_len)

    output = []
    for seq_i in idxs:
        mono_vecs = np.zeros((seq_len, 3), dtype=np.float32)
        valid_mask = seq_i < 4
        mono_vecs[valid_mask] = mono_lut[seq_i[valid_mask]]
        output.append(mono_vecs)

    return np.array(output, dtype=np.float32)


# ==================== SIMPLEX DIMER ENCODING ====================

def create_mono_di_luts():
    mono_lut = create_mono_lut()
    di_lut = np.array([
        [+1, -1, -1, -1, +1, +1, -1, +1, +1],  # AA
        [-1, +1, -1, +1, -1, +1, +1, -1, +1],  # AC
        [-1, -1, +1, +1, +1, -1, +1, +1, -1],  # AG
        [+1, +1, +1, -1, -1, -1, -1, -1, -1],  # AT
        [-1, +1, +1, +1, -1, -1, -1, +1, +1],  # CA
        [+1, -1, +1, -1, +1, -1, +1, -1, +1],  # CC
        [+1, +1, -1, -1, -1, +1, +1, +1, -1],  # CG
        [-1, -1, -1, +1, +1, +1, -1, -1, -1],  # CT
        [-1, +1, +1, -1, +1, +1, +1, -1, -1],  # GA
        [+1, -1, +1, +1, -1, +1, -1, +1, -1],  # GC
        [+1, +1, -1, +1, +1, -1, -1, -1, +1],  # GG
        [-1, -1, -1, -1, -1, -1, +1, +1, +1],  # GT
        [+1, -1, -1, +1, -1, -1, +1, -1, -1],  # TA
        [-1, +1, -1, -1, +1, -1, -1, +1, -1],  # TC
        [-1, -1, +1, -1, -1, +1, -1, -1, +1],  # TG
        [+1, +1, +1, +1, +1, +1, +1, +1, +1],  # TT
    ], dtype=np.float32)

    # For scrambling test
    #np.random.seed(42)
    #np.random.shuffle(di_lut)

    return mono_lut, di_lut

def dna_to_simplex_dimer(seqs, k=2):
    assert k == 2, "Only k=2 is supported for simplex dimer encoding"
    mono_lut, di_lut = create_mono_di_luts()
    seq_len = len(seqs[0])
    assert all(len(s) == seq_len for s in seqs)

    seq_concat = "".join(seqs).upper()
    idxs = base_to_index_array(seq_concat)
    idxs = idxs.reshape(len(seqs), seq_len)

    output = []
    for seq_i in idxs:
        kmers = []
        for i in range(seq_len - k + 1):
            mono_indices = seq_i[i:i+k]
            di_indices = 4 * mono_indices[:-1] + mono_indices[1:]

            mono_vecs = np.zeros((k, 3), dtype=np.float32)
            valid_mono = mono_indices < 4
            mono_vecs[valid_mono] = mono_lut[mono_indices[valid_mono]]

            di_vecs = np.zeros((k - 1, 9), dtype=np.float32)
            for j in range(k - 1):
                if mono_indices[j] < 4 and mono_indices[j + 1] < 4:
                    di_vecs[j] = di_lut[di_indices[j]]

            vec = np.concatenate([mono_vecs.flatten(), di_vecs.flatten()])
            kmers.append(vec)
        output.append(kmers)

    output_array = np.array(output, dtype=np.float32)
    expected_dim = 3 * k + 9 * (k - 1)
    assert output_array.shape[-1] == expected_dim, f"Expected dim {expected_dim}, got {output_array.shape[-1]}"
    return output_array


# ==================== ROUTER ====================

def encode_sequence(seqs, method="one_hot"):
    if method == "one_hot":
        return dna_to_one_hot_real(seqs)
    elif method == "simplex_monomer":
        return dna_to_simplex_monomer(seqs)
    elif method == "simplex_dimer":
        return dna_to_simplex_dimer(seqs)
    else:
        raise ValueError(f"Unknown encoding method: {method}")


# ==================== DISPATCH COMPATIBILITY WRAPPER ====================

# Global encoding method (can be changed from outside this module)
ENCODING_METHOD = "one_hot"

# Preserve the original name, route to encode_sequence()
def dna_to_one_hot(seqs):
    return encode_sequence(seqs, method=ENCODING_METHOD)
