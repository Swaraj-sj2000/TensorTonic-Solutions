import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a=np.array(a)
    b=np.array(b)
    mag_a=np.linalg.norm(a)
    mag_b=np.linalg.norm(b)
    return 0 if mag_a==0 or mag_b==0 else a@b/(mag_a*mag_b) 
