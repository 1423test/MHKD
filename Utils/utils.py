import numpy as np

from scipy.sparse import coo_matrix,diags,eye
import torch

import torch.nn.functional as F
from torch import nn



def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = diags(r_inv)
    mx = r_mat_inv.dot(mx)
    adj = torch.FloatTensor(mx.todense())
    return adj

def matrix(h):
    matrix = np.zeros((len(h), len(h)))
    for i in range(len(h)):
        seq_a = set(h[i])
        seq_a.discard(0)
        for j in range(i+1, len(h)):
            seq_b = set(h[j])
            seq_b.discard(0)
            overlap = seq_a.intersection(seq_b)
            ab_set = seq_a | seq_b
            if len(overlap) == 0 or len(ab_set) == 0:
                matrix[i][j] = 0
            else:
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
            matrix[j][i] = matrix[i][j]
    matrix = matrix + eye(matrix.shape[0])
    adj = coo_matrix(matrix)
    adj = normalize_adj(adj)
    return adj


def loss(h_s,h_t,a,b,norm):

    if norm:
        h_s = F.normalize(h_s)
        h_t = F.normalize(h_t)

    kl = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    p_t = nn.LogSoftmax(dim=1)(h_t)
    p_s = F.softmax(h_s, dim=1)
    L_n = kl(p_t, p_s)

    matrix_exp_t = torch.mm(h_t, h_s.t()).exp_()
    matrix_exp_s = torch.mm(h_s, h_t.t()).exp_()
    L_g = -torch.log(matrix_exp_t.diag() / torch.sum(matrix_exp_t, dim=1)).mean() \
             + -torch.log(matrix_exp_s.diag() / torch.sum(matrix_exp_s, dim=1)).mean()

    loss = L_n * a + L_g * b


    return loss
