import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    if len(seqs) == 0:
        return np.array([])
    
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    
    N = len(seqs)
    out = np.full((N, max_len), pad_value)
    
    for i, seq in enumerate(seqs):
        length = min(len(seq), max_len)
        out[i, :length] = seq[:length]
    
    return out