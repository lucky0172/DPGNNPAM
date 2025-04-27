import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

def load_data1(node_file_path, edge_file_path, imb_ratio=1.0):
    node_df = pd.read_csv(node_file_path, header=0)
    node_features = torch.tensor(node_df.iloc[:, 2:-1].values.astype(float))  
    node_labels = torch.tensor(node_df.iloc[:, -1].values).long()  

    edge_df = pd.read_csv(edge_file_path, delimiter=' ', header=0)
    edge_index = torch.tensor(edge_df.iloc[:, :2].values, dtype=torch.long).t().contiguous()  
    edge_weights = torch.tensor(edge_df.iloc[:, 2].values.astype(float))

    data = Data(x=node_features, edge_index=edge_index, y=node_labels)
    data.edge_weight = edge_weights 

    generate_masks1(data)

    # homophily = (node_labels[edge_index[0]] == node_labels[edge_index[1]]).sum() / len(edge_index[0])
    # print('homophily', homophily)

    valid_labels = node_labels[(node_labels == 1) | (node_labels == 0)]
    valid_class_counts = torch.bincount(valid_labels)
    class_counts = valid_class_counts.tolist()

    total_count = sum(class_counts)

    class_sample_num = total_count
    num_classes = 2

    ratio = [class_count / total_count for class_count in class_counts]
    ratio = torch.tensor(ratio, dtype=torch.float32)

    c_train_num = [int(ratio[i] * class_sample_num) for i in range(len(ratio))]

    classes = torch.tensor([i for i in range(len(class_counts))])

    return data, class_sample_num, num_classes, data.num_features, torch.tensor(c_train_num), classes

def generate_masks1(data, train_ratio=0.8, test_ratio=0.2):
  
    valid_nodes = torch.where(data.y != 2)[0]
    valid_labels = data.y[valid_nodes].numpy()
    train_nodes, test_nodes = train_test_split(valid_nodes.numpy(), train_size=train_ratio, test_size=test_ratio,stratify=valid_labels)

    num_nodes = data.x.size(0)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_nodes] = True
    test_mask[test_nodes] = True

    data.train_mask = train_mask
    data.test_mask = test_mask

def shuffle_masks1(data):
    train_indices = torch.where(data.train_mask)[0]
    test_indices = torch.where(data.test_mask)[0]

    shuffled_train_indices = train_indices[torch.randperm(train_indices.size(0))]
    shuffled_test_indices = test_indices[torch.randperm(test_indices.size(0))]

    new_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    new_test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    new_train_mask[shuffled_train_indices] = True
    new_test_mask[shuffled_test_indices] = True

    data.train_mask = new_train_mask
    data.test_mask = new_test_mask

    return data


