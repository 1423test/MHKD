from torch.nn.parameter import Parameter
import scipy.sparse as sp
import math
import numpy
import torch
from teacher.HeCo.code.utils import load_data, set_params, evaluate
from Utils import hypergraph, finetune
import warnings
import datetime
import pickle as pkl
import os
import random
from Utils.utils import *
from Utils.stu_params import set_stu_params
from collections import defaultdict


warnings.filterwarnings('ignore')
args = set_params()
stu_args = set_stu_params()


warnings.filterwarnings('ignore')
args = set_params()

## random seed ##
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        try:
            input = input.float()
        except:
            pass
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(torch.nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)

    def forward(self, x, adj, use_relu=False):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
            x = self.gc2(x, adj)
        return x


adj = hypergraph.hyper_dblp()
adj = adj.to(device)

ei_index, features, mps, pos, label, idx_train, idx_val, idx_test = \
    load_data(stu_args.dataset, stu_args.ratio, stu_args.type_num)

num_class = label.shape[-1]

student = GCN(nfeat=features[0].shape[1], nhid=stu_args.hidden_dim).to(device)
opt = torch.optim.Adam(student.parameters(), lr=stu_args.lr, weight_decay=stu_args.lr)

h_t =   torch.from_numpy(np.load("/home/yhkj/D/MHGCN/HeCo/code/embeds/"+stu_args.dataset+"/"+str(args.turn)+".pkl", "wb", allow_pickle=True)).to(device)


label = label.to(device)
features = features[0].to(device)
idx_train = [i.cuda() for i in idx_train]
idx_val = [i.cuda() for i in idx_val]
idx_test = [i.cuda() for i in idx_test]


cnt_wait = 0
best = 1e9
best_t = 0
starttime = datetime.datetime.now()

for epoch in range(3000):
    student.train()
    opt.zero_grad()

    h_s = student(features, adj)
    loss_kd = loss(h_t,h_s,stu_args.Ln1,stu_args.Lg1,stu_args.norm1)

    print("loss_kd {:.4f} ".format(loss_kd))

    if loss_kd < best:
        best = loss_kd
        best_t = epoch
        cnt_wait = 0
        torch.save(student.state_dict(), 'hyper'+ stu_args.dataset + '.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == stu_args.patience:
        print('Early stopping!')
        break

    loss_kd.backward()
    opt.step()

print('Loading {}th epoch'.format(best_t))
student.load_state_dict(torch.load('hyper'+ stu_args.dataset + '.pkl'))
student.eval()
os.remove('hyper'+ stu_args.dataset + '.pkl')
embeds = student(features, adj)
for i in range(len(idx_train)):
    finetune.finetune(embeds.detach(), stu_args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label,
                      num_class, device, stu_args.dataset,
                      args.eva_lr, args.eva_wd,
                      stu_args.Ln2, stu_args.Lg2, stu_args.norm2, device)
endtime = datetime.datetime.now()
endtime = datetime.datetime.now()
time = (endtime - starttime).seconds
print("Total time: ", time, "s")

print('*'*30)
for i in range(len(idx_train)):
    evaluate(h_t.detach(), args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label,
                      num_class, device, args.dataset,
                      args.eva_lr, args.eva_wd)
