from tensorflow.keras.utils import Sequence
import pandas as pd
import numpy as np
import math
import pyfaidx
from chrombpnet.training.utils import one_hot  # this imports dna_to_one_hot & ENCODING_METHOD

class SNPGenerator(Sequence):
    def __init__(self,
                 snp_regions,
                 inputlen,
                 genome_fasta,
                 batch_size=50,
                 debug_mode_on=False,
                 encoding_method="one_hot"  # <- NEW ARG
                 ):

        self.snp_regions = snp_regions
        self.num_snps = self.snp_regions.shape[0]
        self.inputlen = inputlen
        self.batch_size = batch_size
        self.genome = pyfaidx.Fasta(genome_fasta)
        self.debug_mode_on = debug_mode_on
        self.encoding_method = encoding_method

    def __getitem__(self, idx):
        ref_seqs = []
        alt_seqs = []
        rsids = []

        cur_entries = self.snp_regions.iloc[
            idx * self.batch_size: min([self.num_snps, (idx + 1) * self.batch_size])
        ]
        flank_size = self.inputlen // 2

        for index, entry in cur_entries.iterrows():
            cur_chrom = str(entry["CHR"])
            cur_pos = int(entry["POS0"])
            ref_snp = str(entry["REF"])
            alt_snp = str(entry["ALT"])
            meta = str(entry["META_DATA"])

            rsid = f"{cur_chrom}_{cur_pos}_{ref_snp}_{alt_snp}_{meta}"

            left_flank = str(self.genome[cur_chrom][max(0, cur_pos - flank_size):cur_pos])
            right_flank = str(self.genome[cur_chrom][cur_pos + 1:cur_pos + flank_size])

            cur_ref_seq = left_flank + ref_snp + right_flank
            cur_alt_seq = left_flank + alt_snp + right_flank

            if self.debug_mode_on:
                print(f"CHR_POS_REF_ALT_META: {rsid}")
                print(f"Reference/alternate right flank: {right_flank}")
                print(f"Reference/alternate left flank: {left_flank}")

            if len(cur_ref_seq) != self.inputlen or len(cur_alt_seq) != self.inputlen:
                print("❗ Sequence length mismatch. Skipping SNP:", rsid)
                continue

            ref_seqs.append(cur_ref_seq)
            alt_seqs.append(cur_alt_seq)
            rsids.append(rsid)

        return rsids, one_hot.encode_sequence(ref_seqs, method=self.encoding_method), one_hot.encode_sequence(alt_seqs, method=self.encoding_method)

    def __len__(self):
        return math.ceil(self.num_snps / self.batch_size)
