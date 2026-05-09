import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    g = np.array(g,dtype = float)
    if max_norm <= 0:
        return g
    # v = g.flatten()
    # total = 0
    # for i in v:
    #   total += i**2
    # norm = np.sqrt(total)
    # if len(g.shape) == 1:
    #     for i in range(len(g)):
    #         if norm <= max_norm:
    #           g[i] = g[i]
    #         else:
    #           g[i] = (max_norm/norm)*g[i]
    # else:
    #     H = g.shape[-2:][0]
    #     W = g.shape[-2:][1]
    #     for h in range(H):
    #       for w in range(W):
    #         if norm <= max_norm:
    #           g[...,h,w] = g[...,h,w]
    #         else:
    #           g[...,h,w] = (max_norm/norm)*g[...,h,w]

    norm = np.linalg.norm(g)
    if norm > max_norm:
        g *= (max_norm/norm)
    return g