import argparse
import sys

argv = sys.argv
dataset = 'dblp'


def acm_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', default=True, action="store_true")
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--patience', type=int, default=10)

    # The parameters of loss
    parser.add_argument('--Ln1', type=float, default=0.8)
    parser.add_argument('--Lg1', type=float, default=0.2)
    parser.add_argument('--norm1', type=float, default=True)


    parser.add_argument('--Ln2', type=float, default=0.7)
    parser.add_argument('--Lg2', type=float, default=0)
    parser.add_argument('--norm2', type=float, default=False)

    # The parameters of learning process
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--wd', type=float, default=0.0001)


    # model-specific parameter
    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60]  # the number of every node type

    return args


def dblp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', default=True, action="store_true")
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--patience', type=int, default=10)

    # The parameters of loss
    parser.add_argument('--Ln1', type=float, default=1)
    parser.add_argument('--Lg1', type=float, default=1)
    parser.add_argument('--norm1', type=float, default=True)

    parser.add_argument('--Ln2', type=float, default=0.3)
    parser.add_argument('--Lg2', type=float, default=0.05)
    parser.add_argument('--norm2', type=float, default=False)

    # The parameters of learning process
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--wd', type=float, default=0.0001)


    args, _ = parser.parse_known_args()
    args.type_num = [4057, 14328, 7723, 20]  # the number of every node type
    args.nei_num = 1  # the number of neighbors' types
    return args


def aminer_params():
    parser = argparse.ArgumentParser()
     parser.add_argument('--save_emb', default=True, action="store_true")
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--patience', type=int, default=10)

    # The parameters of loss
    parser.add_argument('--Ln1', type=float, default=1)
    parser.add_argument('--Lg1', type=float, default=1)
    parser.add_argument('--norm1', type=float, default=True)

    parser.add_argument('--Ln2', type=float, default=0.6)
    parser.add_argument('--Lg2', type=float, default=0)
    parser.add_argument('--norm2', type=float, default=False)

    # The parameters of learning process
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--wd', type=float, default=0.0001)

    args, _ = parser.parse_known_args()
    args.type_num = [6564, 13329, 35890]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args


def freebase_params():
    parser = argparse.ArgumentParser()
      parser.add_argument('--save_emb', default=True, action="store_true")
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--patience', type=int, default=10)

    # The parameters of loss
    parser.add_argument('--Ln1', type=float, default=0.7)
    parser.add_argument('--Lg1', type=float, default=0.05)
    parser.add_argument('--norm1', type=float, default=True)

    parser.add_argument('--Ln2', type=float, default=0.6)
    parser.add_argument('--Lg2', type=float, default=0)
    parser.add_argument('--norm2', type=float, default=False)

    # The parameters of learning process
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--wd', type=float, default=0.0001)

    args, _ = parser.parse_known_args()
    args.type_num = [3492, 2502, 33401, 4459]  # the number of every node type
    args.nei_num = 3  # the number of neighbors' types
    return args


def set_stu_params():
    if dataset == "acm":
        args = acm_params()
    elif dataset == "dblp":
        args = dblp_params()
    elif dataset == "aminer":
        args = aminer_params()
    elif dataset == "freebase":
        args = freebase_params()
    return args
