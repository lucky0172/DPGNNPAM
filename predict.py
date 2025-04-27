import argparse
import torch
import pandas as pd
from model import *
from dataset1 import load_data1
from learn import *
from random_model import *
from torch_geometric.nn import GAE

def predict_labels(args):
    data, class_sample_num, args.num_classes, args.num_features, args.c_train_num, args.classes = load_data1(args.node_file_path, args.edge_file_path)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.encoder == 'GCN':
        encoder = GCN(args)

    if args.encoder == 'GAT':
        encoder = GAT(args)

    if args.encoder == 'GAE':
        encoder = GAE(args)

    if args.encoder == 'Random':
        encoder = PathAgg_att_sample2(config)

    dist_encoder = dist_embed(args)
    proto = prototype()

    if args.encoder == 'GCN':
        encoder.load_state_dict(torch.load('encoder_gcn.pkl'),strict=False)
        dist_encoder.load_state_dict(torch.load('dist_encoder_gcn.pkl'),strict=False)
    elif args.encoder == 'GAT':
        encoder.load_state_dict(torch.load('encoder_gat.pkl'),strict=False)
        encoder.load_state_dict(torch.load('dist_encoder_gat.pkl'),strict=False)
    elif args.encoder == 'GAE':
        encoder.load_state_dict(torch.load('encoder_gae.pkl'),strict=False)
        dist_encoder.load_state_dict(torch.load('dist_encoder_gae.pkl'),strict=False)
    else:
        encoder.load_state_dict(torch.load('encoder_random.pkl'),strict=False)
        dist_encoder.load_state_dict(torch.load('dist_encoder_random.pkl'),strict=False)



    pred_probabilities = predict(encoder, dist_encoder, proto, data, args,config)

    node_indices = torch.arange(data.num_nodes).cpu().numpy()

    pred_probabilities = pred_probabilities

    output_data = pd.DataFrame({
        'Node_Index': node_indices,
        'Class_0_Probability': pred_probabilities[:, 0],
        'Class_1_Probability': pred_probabilities[:, 1]  
    })

    if args.encoder == 'GCN':
        output_data.to_csv('gcn_predict.csv', index=False)
    elif args.encoder == 'GAT':
        output_data.to_csv('gat_predict.csv', index=False)
    elif args.encoder == 'GAE':
        output_data.to_csv('gae_predict.csv', index=False)
    else:
        output_data.to_csv('random_predict.csv', index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_file_path', type=str, default="./data/gene_end1.csv")
    parser.add_argument('--edge_file_path', type=str, default="./data/ppi_end1.txt")
    parser.add_argument('--shuffle', type=str, default='eval_total')  
    parser.add_argument('--encoder', type=str, default='Random')  
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--episodic_samp', type=float, default=1.0) 
    parser.add_argument('--runs', type=int, default=5)  
    parser.add_argument('--imb_ratio', type=float, default=10)  
    parser.add_argument('--label_prop', type=str, default='yes')  
    parser.add_argument('--eta', type=float, default=3.0)  
    parser.add_argument('--ssl', type=str, default='yes') 

    parser.add_argument('--dropout', type=float, default=0.3)  
    parser.add_argument('--lamb1', type=float, default=10)
    parser.add_argument('--lamb2', type=float, default=10)

    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--n_hidden', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()

    predict_labels(args)
