# coding=utf-8
from __future__ import division
from __future__ import print_function
from utils import *
import torch.nn as nn
from torch_sparse import SparseTensor
import torch_geometric.utils.undirected as undirected

def generate_random_walk(adj, config, p, q):
    long_walks_per_node = config.long_walks_per_node
    long_walk_len = config.long_walk_len
    walk_len = config.walk_len
    batch = torch.arange(config.n)
    batch = batch.repeat(long_walks_per_node)

    rowptr, col, _ = adj.csr()

    rw = torch.ops.torch_cluster.random_walk(rowptr, col, batch, long_walk_len, p, q)  

    if not isinstance(rw, torch.Tensor):
        rw = rw[0]
    walks = []
    num_walks_per_rw = 1 + long_walk_len + 1 - walk_len
    for j in range(num_walks_per_rw):
        walks.append(rw[:, j:j + walk_len])
    out_walks = torch.cat(walks, dim=0)
    out_walks = out_walks[:, torch.arange(start=walk_len - 1, end=-1, step=-1)]
    return out_walks

class PathAgg_att_sample_layer(nn.Module): 
    def __init__(self, in_dim, out_dim, config, strategy="DFS"):
        super(PathAgg_att_sample_layer, self).__init__()
        self.dropout = config.dropout
        self.rnn = nn.GRU(input_size=in_dim, hidden_size=out_dim, batch_first=True)
        self.a = nn.Parameter(torch.rand(out_dim, config.head_num, dtype=torch.float32) * 2 - 1)
        self.head_num = config.head_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.strategy = strategy

    def forward(self, x, adj, config):
        if self.strategy == "DFS":
            path_list = generate_random_walk(adj, config, p=10, q=0.1)

        else:  # BFS
            path_list = generate_random_walk(adj, config, p=0.1, q=10)

        path_list = path_list.long()

        M = SparseTensor(row=path_list[:, -1], col=torch.arange(len(path_list)), value=None, sparse_sizes=(config.n, len(path_list)))

        M = M.to(x.device)

        path_list = path_list.to(x.device)

        path_features = F.embedding(path_list, x)

        path_features = path_features.to(torch.float32)

        _, emb = self.rnn(path_features)  
        path_emb = emb.squeeze(0)

        # path_att_un = (path_emb @ self.a).exp()
        path_att_un = torch.exp(F.leaky_relu(path_emb @ self.a, negative_slope=0.2))  # path_num x head_num, path attention before softmax

        path_att_un_sum_by_node = M @ path_att_un  # node_num x head_num, path attention sum on nodes 
        path_att_un_low = M.t() @ (1 / path_att_un_sum_by_node)  # path_num x head_num, path attention
        path_att_n = path_att_un * path_att_un_low  # path_num x head_num, path attention on node after softmax 

        path_att_n = path_att_n.reshape(-1, 1).repeat(1, self.out_dim).reshape(config.walks_per_node * config.n, -1)
        path_emb = path_emb.repeat(1, self.head_num)

        path_att_emb = path_att_n * path_emb  # path_num x hid_num, weighted path hidden embedding

        node_att_emb = M @ path_att_emb  # node_num x hid_num, weighted node hidden embedding

        del path_list
        return node_att_emb

class PathAgg_att_sample2(nn.Module):
    def __init__(self, config):
        super(PathAgg_att_sample2, self).__init__()
        self.dropout = config.dropout
        self.layer1 = PathAgg_att_sample_layer(in_dim=config.fdim, out_dim=config.nhid1, config=config, strategy="DFS")
        self.layer2 = PathAgg_att_sample_layer(in_dim=config.fdim, out_dim=config.nhid1, config=config, strategy="BFS")


    def forward(self, data, config):
        edge_index = data.edge_index

        edge_index_filtered = load_graph_edgelist_no_loop(edge_index)

        edge_index_with_loops = add_self_loop_for_all_nodes(edge_index_filtered, data.num_nodes)

        adj = SparseTensor(row=edge_index_with_loops[0], col=edge_index_with_loops[1],sparse_sizes=(data.num_nodes, data.num_nodes))

        features, labels = load_feature_and_label(data)

        x1 = self.layer1(features, adj, config)
        x2 = self.layer2(features, adj, config)

        x3 = torch.cat([x1, x2], dim=1)
        # x3 = x1 
        # x3 = x2 
        x3 = F.relu(x3)
        return x3

    def get_param_groups(self):
        param_groups = [
            {'params': self.layer1.rnn.parameters(), 'lr': 1e-2, 'weight_decay': 5e-4},
            {'params': self.layer2.rnn.parameters(), 'lr': 1e-2, 'weight_decay': 0},
            {'params': self.layer1.a, 'lr': 1e-2, 'weight_decay': 5e-4},
            {'params': self.layer2.a, 'lr': 1e-2, 'weight_decay': 5e-4}
        ]
        return param_groups


def add_self_loop_for_all_nodes(edge_list, node_num):  
    loops = torch.tensor([[i for i in range(node_num)], [i for i in range(node_num)]], dtype=torch.int).t()
    edge_list = edge_list.t()
    edge_list = torch.cat([edge_list, loops], dim=0)
    return edge_list



def add_self_loop_for_isolated_nodes(edge_list, node_num): 
    print("calculating isolated nodes.")
    l = torch.zeros(node_num)
    l[edge_list[:, 0]] = 1
    l[edge_list[:, 1]] = 1
    idx = torch.arange(0, node_num)[l==0]
    if idx.size(0) == 0:
        print("no isolated nodes.")
    else:
        print("found {} isolated nodes.".format(idx.size(0)))
    idx = torch.stack([idx, idx], dim = 1)
    edge_list = torch.cat([edge_list, idx], dim = 0)
    return edge_list


def load_graph_edgelist_no_loop(edge_index):
    sedges = edge_index.cpu().numpy().transpose()

    filtered_edges = []

    for src, dst in sedges:
        if src != dst:
            filtered_edges.append((src, dst))

    filtered_tensor = torch.tensor(filtered_edges, dtype=torch.int64).t().contiguous()
    undirected_edges = undirected.to_undirected(filtered_tensor)

    return undirected_edges



def load_idx_from_masks(data, ratio=0.8, seed=None):
    idx_train = torch.where(data.train_mask)[0]
    idx_val = torch.where(data.val_mask)[0]
    idx_test = torch.where(data.test_mask)[0]

    if seed is not None:
        np.random.seed(seed)

    np.random.shuffle(idx_train.numpy())
    train_size = int(len(idx_train) * ratio)
    idx_train1 = idx_train[:train_size]

    np.random.shuffle(idx_val.numpy())
    val_size = int(len(idx_val) * ratio)
    idx_val1 = idx_val[:val_size]

    np.random.shuffle(idx_test.numpy())
    val_size = int(len(idx_test) * ratio)
    idx_test1 = idx_val[:val_size]

    return idx_train1, idx_val1, idx_test1

def load_feature_and_label(data): 
    features = data.x
    label = data.y
    return features, label









