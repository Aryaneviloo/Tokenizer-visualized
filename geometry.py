import numpy as np
import math

#Simulates a trained embedding matrix. Takes a list of token IDs 
  #  and returns a dense matrix where each token gets a unique, static 
   # vector of shape (d_model,) filled with continuous values between -1 and 1.

def get_semantic(token_ids, d_model=16):
    token_embeddings=[]
    for tid in token_ids:
        rng = np.random.default_rng(tid)
        token_embeddings.append(rng.uniform(-1, 1, d_model))
    return np.array(token_embeddings)

def sinusodial_encoding(seq_len, d_model=16):
    pos_embeddings = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            div_term = 10000 ** (i/d_model)
            pos_embeddings[pos, i] = math.sin(pos/div_term)
            if i + 1 <d_model:
                pos_embeddings[pos, i+1] = math.cos(pos/div_term)
    return pos_embeddings

def compute_additive(token_embeddings, pos_embeddings):
    return token_embeddings+pos_embeddings


def simulate_rope_rotation(base_vector, positions_to_simulate, theta_base=0.4):
    x_base = base_vector[0]
    y_base = base_vector[1]
    rotated_points = []
    for pos in positions_to_simulate:
        angle = pos * theta_base
        x_rot = x_base * math.cos(angle) - y_base * math.sin(angle)
        y_rot = x_base * math.sin(angle) + y_base * math.cos(angle)
        rotated_points.append({
            "position": pos,
            "x": x_rot,
            "y": y_rot
        })
        
    return rotated_points

