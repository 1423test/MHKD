from Utils.utils import*
import dgl
from collections import defaultdict
import dgl
import os.path as osp
from itertools import product

import numpy as np
import scipy.sparse as sp
import torch

device = torch.device("cuda:0")

def meta_dblp():

    raw_dir = './data/dblp/raw/'

    node_types = ['author', 'paper', 'term', 'conference']
    x = sp.load_npz(osp.join(raw_dir, f'features_0.npz'))
    author_features = torch.from_numpy(x.todense()).to(torch.float) #4057

    x = sp.load_npz(osp.join(raw_dir, f'features_1.npz'))
    paper_features = torch.from_numpy(x.todense()).to(torch.float) #14328

    x = np.load(osp.join(raw_dir, f'features_2.npy'))
    term_features = torch.from_numpy(x).to(torch.float) #7723

    y = np.load(osp.join(raw_dir, 'labels.npy'))
    labels = torch.from_numpy(y).to(torch.long) #4075

    node_type_idx = np.load(osp.join(raw_dir, 'node_types.npy'))
    node_type_idx = torch.from_numpy(node_type_idx).to(torch.long)

    s = {}
    N_a = author_features.shape[0]
    N_p = paper_features.shape[0]
    N_t = term_features.shape[0]
    N_c = int((node_type_idx == 3).sum())
    s['author'] = (0, N_a)
    s['paper'] = (N_a, N_a + N_p)
    s['term'] = (N_a + N_p, N_a + N_p + N_t)
    s['conference'] = (N_a + N_p + N_t, N_a + N_p + N_t + N_c)

    A = sp.load_npz(osp.join(raw_dir, 'adjM.npz'))
    for src, dst in product(node_types, node_types):
        A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]].tocoo()
        if A_sub.nnz > 0:
            row = torch.from_numpy(A_sub.row).to(torch.long)
            col = torch.from_numpy(A_sub.col).to(torch.long)
            s[src, dst] = (row, col)

    hg = dgl.heterograph(
        {
            ('paper', 'pa', 'author'): s['paper', 'author'],
            ('author', 'ap', 'paper'): s['author', 'paper'],
            ('paper', 'pt', 'term'): s['paper', 'term'],
            ('term', 'tp', 'paper'): s['term', 'paper'],
            ('paper', 'pc', 'conference'): s['paper', 'conference'],
            ('conference', 'cp', 'paper'): s['conference', 'paper'],
        })

    return hg


def hyper_dblp():

    g = meta_dblp()

    meta0 = g['author', 'ap', 'paper'].edges()
    meta1 = g['paper', 'pt', 'term'].edges()
    meta2 = g['paper', 'pc', 'conference'].edges()
    target_node0 = meta0[0].tolist()
    ass_node0 = meta0[1].tolist()
    target_node1 = meta1[0].tolist()
    ass_node1 = meta1[1].tolist()
    target_node2 = meta2[0].tolist()
    ass_node2 = meta2[1].tolist()

    hypergraph = defaultdict(list)
    p_t = defaultdict(list)
    p_c = defaultdict(list)

    # 1-author , author-paper+author, paper+author-paper+author+term
    n = g.number_of_nodes('author')
    n1 = n + g.number_of_nodes('paper')
    n2 = n1 + g.number_of_nodes('term')

    for i, m in enumerate(target_node0):
        hypergraph[m].append(n + ass_node0[i])
    for i, m in enumerate(target_node1):
        p_t[m + n].append(n1 + ass_node1[i])
    for i, m in enumerate(target_node2):
        p_c[m + n].append(n2 + ass_node2[i])

    for key, val in hypergraph.items():
        for v in val:
            if v in p_t.keys():
                val.extend(p_t.get(v))

            if v in p_c.keys():
                val.extend(p_c.get(v))

    device = torch.device("cuda:1")

    h = list(hypergraph.values())
    adj = matrix(h)

    return adj
