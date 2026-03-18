import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    
    pos = np.arange(seq_len)[:, None] #[[0][1][2]]
    i = np.arange(d_model)[None, :] #[[0 1 2 3]]
    
    angle_rates = 1 / (base ** ((2 * (i // 2)) / d_model)) #[[0 0 2/4 2/4]]=>[[]]
    angles = pos * angle_rates
    
    pe = np.zeros((seq_len, d_model))
    
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    
    return pe