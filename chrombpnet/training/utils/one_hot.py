# """
# Written by Alex Tseng
# https://gist.github.com/amtseng/010dd522daaabc92b014f075a34a0a0a
# """

# import numpy as np

# def dna_to_one_hot(seqs):
#     """
#     Converts a list of DNA ("ACGT") sequences to one-hot encodings, where the
#     position of 1s is ordered alphabetically by "ACGT". `seqs` must be a list
#     of N strings, where every string is the same length L. Returns an N x L x 4
#     NumPy array of one-hot encodings, in the same order as the input sequences.
#     All bases will be converted to upper-case prior to performing the encoding.
#     Any bases that are not "ACGT" will be given an encoding of all 0s.
#     """
#     seq_len = len(seqs[0])
#     assert np.all(np.array([len(s) for s in seqs]) == seq_len)

#     # Join all sequences together into one long string, all uppercase
#     seq_concat = "".join(seqs).upper() + "ACGT"
#     # Add one example of each base, so np.unique doesn't miss indices later

#     one_hot_map = np.identity(5)[:, :-1].astype(np.int8)

#     # Convert string into array of ASCII character codes;
#     base_vals = np.frombuffer(bytearray(seq_concat, "utf8"), dtype=np.int8)

#     # Anything that's not an A, C, G, or T gets assigned a higher code
#     base_vals[~np.isin(base_vals, np.array([65, 67, 71, 84]))] = 85

#     # Convert the codes into indices in [0, 4], in ascending order by code
#     _, base_inds = np.unique(base_vals, return_inverse=True)

#     # Get the one-hot encoding for those indices, and reshape back to separate
#     return one_hot_map[base_inds[:-4]].reshape((len(seqs), seq_len, 4))


# def one_hot_to_dna(one_hot):
#     """
#     Converts a one-hot encoding into a list of DNA ("ACGT") sequences, where the
#     position of 1s is ordered alphabetically by "ACGT". `one_hot` must be an
#     N x L x 4 array of one-hot encodings. Returns a list of N "ACGT" strings,
#     each of length L, in the same order as the input array. The returned
#     sequences will only consist of letters "A", "C", "G", "T", or "N" (all
#     upper-case). Any encodings that are all 0s will be translated to "N".
#     """
#     bases = np.array(["A", "C", "G", "T", "N"])
#     # Create N x L array of all 5s
#     one_hot_inds = np.tile(one_hot.shape[2], one_hot.shape[:2])

#     # Get indices of where the 1s are
#     batch_inds, seq_inds, base_inds = np.where(one_hot)

#     # In each of the locations in the N x L array, fill in the location of the 1
#     one_hot_inds[batch_inds, seq_inds] = base_inds

#     # Fetch the corresponding base for each position using indexing
#     seq_array = bases[one_hot_inds]
#     return ["".join(seq) for seq in seq_array]


### Below is the modified Simplex-encoded version of `dna_to_one_hot`

import numpy as np

def create_luts():
    # Mononucleotide LUT Index: A=0, C=1, G=2, T=3
    mono_lut = np.array([
        [1, -1, -1],    # A
        [-1, 1, -1],    # C
        [-1, -1, 1],    # G
        [1, 1, 1],      # T
    ], dtype=np.float32)

    return mono_lut

# Convert DNA to integer indices (ASCII)
# Same method as chrombpnet (numpy handle integer to vector better)
def base_to_index_array(seq_concat):
    # A=65, C=67, G=71, T=84 → map to 0–3
    base_vals = np.frombuffer(bytearray(seq_concat, "utf8"), dtype=np.int8)
    base_vals[~np.isin(base_vals, np.array([65, 67, 71, 84]))] = 85  # Replace non-ACGT with 'N'; N = 85
    # Map ACGT to 0–3, N to 4
    base_map = {65: 0, 67: 1, 71: 2, 84: 3, 85: 4}
    idx_arr = np.vectorize(base_map.get)(base_vals)   # ASCII to index
    return idx_arr

# Main Simplex LUT Encoder
def dna_to_one_hot(seqs, k=10):  # Renamed from dna_to_simplex_lut
    """
    Converts DNA sequences to simplex-encoded vectors using mononucleotide LUT.
    Returned shape: (batch, sequence_length, 3)
    """
    mono_lut = create_luts()
    # Ensuring all input sequences are of same length -- needed for reshaping and batching
    seq_len = len(seqs[0])
    assert np.all(np.array([len(s) for s in seqs]) == seq_len)
    
    # Step 1: Flatten and convert to index
    seq_concat = "".join(seqs).upper()   # Join all sequences into one long string and uppercase them
    idxs = base_to_index_array(seq_concat)   # Get 1D array of base indices (0-4)

    # Step 2: Reshape flat array into batches (batch, seq_len)
    total_bases = len(seqs) * seq_len
    idxs = idxs[:total_bases].reshape(len(seqs), seq_len)

    output = []
    # Loop over each sequence in the batch
    for seq_i in idxs:
        mono_vecs = np.zeros((seq_len, 3), dtype=np.float32)
        valid_mask = seq_i < 4
        mono_vecs[valid_mask] = mono_lut[seq_i[valid_mask]]
        output.append(mono_vecs)

    # Converts list of lists to a NumPy array of shape
    return np.array(output, dtype=np.float32)
