import torch
from einops import rearrange, repeat

import numpy as np

def ctloss(mapping, y, m=1):
    n = y.size(0)
    eq_matrix = y.unsqueeze(1) == y.unsqueeze(0)
    mapping_y = repeat(mapping, 'n c -> n b c', b=n)
    mapping_x = repeat(mapping, 'n c -> b n c', b=n)

    np_eq_matrix = np.array(eq_matrix.detach())
    np_mapping_y = np.array(mapping_y.detach())
    np_mapping_X = np.array(mapping_x.detach())

    D = torch.norm(mapping_x - mapping_y, dim=-1, p=2)

    np_D = np.array(D.detach())

    Ls = torch.sum(D[eq_matrix] ** 2)
    Ld = torch.sum(torch.clamp(m - D[~eq_matrix], min=0) ** 2)
    L = (Ls + Ld) / (n * (n-1))
    return L


if __name__ == '__main__':
    # X = torch.rand(16, 2, requires_grad=True)
    # y = torch.randint(high=2, size=(16,))
    # loss = ctloss(X, y)
    # loss.backward()
    a = torch.tensor([-2, -1, 0, 1, 2, 100])
    print(torch.clamp(a, min=0))
