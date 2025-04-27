from __future__ import division
from __future__ import print_function
from torch_geometric.nn import GCNConv, MessagePassing
from torch.nn import Linear
from torch_scatter import scatter_add
from torch_geometric.nn import GATConv
from utils import *
import torch.nn as nn
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv


class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(args.num_features, args.n_hidden)
        self.conv2 = GCNConv(args.n_hidden, args.n_hidden)
        self.dropout = args.dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training,p = self.dropout)
        x = self.conv2(x, edge_index)
        return x

# class GCNEncoder(torch.nn.Module):
#     def __init__(self, args):
#         super(GCNEncoder, self).__init__()
#         self.conv1 = GCNConv(args.num_features,args.n_hidden)
#         self.conv2 = GCNConv(args.n_hidden,args.n_hidden)
#         self.dropout = args.dropout
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = x.float()
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x,p=self.dropout, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x



class GAT(torch.nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        self.conv1 = GATConv(args.num_features, args.n_hidden, heads=args.num_heads, dropout=args.dropout)
        self.conv2 = GATConv(args.n_hidden * args.num_heads, args.n_hidden, heads=1, concat=False, dropout=args.dropout)
        self.dropout = args.dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float() 
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GCN_GAT(torch.nn.Module):
    def __init__(self, args):
        super(GCN_GAT, self).__init__()
        self.gcn_conv1 = GCNConv(args.num_features, args.n_hidden)
        self.gcn_conv2 = GCNConv(args.n_hidden, 16)

        self.gat_conv1 = GATConv(args.num_features, args.n_hidden, heads=args.num_heads, dropout=args.dropout)
        self.gat_conv2 = GATConv(args.n_hidden * args.num_heads, 16, heads=1, concat=False,dropout=args.dropout)

        self.dropout = args.dropout

    def forward(self, data):
        x_gcn = data.x.float() 
        x_gcn = F.relu(self.gcn_conv1(x_gcn, data.edge_index))
        x_gcn = F.dropout(x_gcn, training=self.training, p=self.dropout)
        x_gcn = self.gcn_conv2(x_gcn, data.edge_index)

        x_gat = data.x.float()  
        x_gat = F.relu(self.gat_conv1(x_gat, data.edge_index))
        x_gat = F.dropout(x_gat, p=self.dropout, training=self.training)
        x_gat = self.gat_conv2(x_gat, data.edge_index)

        x = torch.cat((x_gcn, x_gat), dim=1)  
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, args):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(args.num_features, args.n_hidden)
        self.conv2 = SAGEConv(args.n_hidden, args.n_hidden)
        self.dropout = args.dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x




