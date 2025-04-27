from model import *
from random_model import *
from torch_geometric.nn import GAE
from dataset1 import *
from evaluation import *
from prototype import *
from config import *


def main(args):
    data, class_sample_num, args.num_classes, args.num_features, args.c_train_num, args.classes = load_data1(args.node_file_path, args.edge_file_path)

    if args.ssl2 == 'yes' or args.ssl3 == 'yes':  
        deg_inv_sqrt = deg(data.edge_index, data.x)

    pbar = tqdm(total=args.runs)

    F1 = np.zeros((args.runs, args.num_classes), dtype=float)
    F1_weight = np.zeros(args.runs, dtype=float)
    Accuracy = np.zeros(args.runs, dtype=float)
    Precision = np.zeros(args.runs, dtype=float)
    Recall = np.zeros(args.runs, dtype=float)
    avg_roc_auc = np.zeros(args.runs, dtype=float)
    avg_pr_auc = np.zeros(args.runs, dtype=float)

    log_file_path = "Results.txt"
    log_file = open(log_file_path, "w")


    for count in range(args.runs):
        random.seed(args.seed + count)
        np.random.seed(args.seed + count)
        torch.manual_seed(args.seed + count)
        torch.cuda.manual_seed(args.seed + count)

        data = shuffle_masks1(data)

        data.y_aug, data.train_mask = data.y, data.train_mask

        proto = prototype()

        args.deg_inv_sqrt = deg_inv_sqrt

        if args.encoder == 'GCN':
            encoder = GCN(args)
        elif args.encoder == 'GAT':
            encoder = GAT(args)
        elif args.encoder == 'Graphsage':
            encoder = GraphSAGE(args)
        elif args.encoder == 'GCN_GAT':
            encoder = GCN_GAT(args)
        else:
            config_file = "./config/config.ini"
            config = Config(config_file)
            encoder = PathAgg_att_sample2(config)

        dist_encoder = dist_embed(args)

        model_param_group = []
        if args.encoder in ['GCN', 'GAT', 'Graphsage']:
            model_param_group.extend([
                {'params': encoder.conv1.parameters(), 'lr': 1e-2, 'weight_decay': 5e-4},
                {'params': encoder.conv2.parameters(), 'lr': 1e-2, 'weight_decay': 0},
            ])
        elif args.encoder == 'GCN_GAT':
            model_param_group.extend([
                {'params': encoder.gcn_conv1.parameters(), 'lr': 1e-2, 'weight-decay': 5e-4},
                {'params': encoder.gcn_conv2.parameters(), 'lr': 1e-2, 'weight-decay': 5e-4},
                {'params': encoder.gat_conv1.parameters(), 'lr': 1e-2, 'weight-decay': 5e-4},
                {'params': encoder.gat_conv2.parameters(), 'lr': 1e-2, 'weight-decay': 5e-4}
            ])
        else:
            model_param_group = encoder.get_param_groups()

        model_param_group.append({'params': dist_encoder.lin.parameters(), 'lr': 1e-2, 'weight_decay': 0})

        optimizer = torch.optim.Adam(model_param_group)
        criterion = torch.nn.NLLLoss()

        best_tmp_f1_mean = 0
        tmp_f1_history = []

        for epoch in range(args.epochs):
            train(encoder, dist_encoder, proto, data, optimizer, criterion, args, config)

            f1, f1w, accs, precisions, recalls, preds, ys, pred_probs,confusion_matrices = test(encoder, dist_encoder, proto, data, args, config)

            tmp_test_f1 = f1[0]
            tmp_test_f1w = f1w[0]
            tmp_test_acc = accs[0]
            tmp_test_precision = precisions[0]
            tmp_test_recall = recalls[0]
            tmp_test_roc_auc, tmp_test_pr_auc = plot_roc_and_pr(ys, pred_probs)


            if np.mean(tmp_test_f1) > best_tmp_f1_mean:
                best_tmp_f1_mean = np.mean(tmp_test_f1)
                test_f1 = tmp_test_f1
                test_acc = tmp_test_acc
                test_f1w = tmp_test_f1w
                test_precision = tmp_test_precision
                test_recall = tmp_test_recall
                test_roc_auc = tmp_test_roc_auc
                test_pr_auc = tmp_test_pr_auc

                save_model(encoder, dist_encoder, args)

            tmp_f1_history.append(np.mean(tmp_test_f1))
            if args.early_stopping > 0 and epoch > args.epochs // 2:
                tmp = torch.tensor(tmp_f1_history[-(args.early_stopping + 1): -1])
                if np.mean(tmp_test_f1) < tmp.mean().item():
                    break

        F1[count] = test_f1
        Accuracy[count] = test_acc
        F1_weight[count] = test_f1w
        Precision[count] = test_precision
        Recall[count] = test_recall
        avg_roc_auc[count] = test_roc_auc
        avg_pr_auc[count] = test_pr_auc

        log_message = (
            f'Run {count + 1} Results: '
            f'F1-macro: {test_f1:.4f}, '
            f'F1-weight: {test_f1w:.4f}, '
            f'Accuracy: {test_acc:.4f}, '
            f'Precision: {test_precision:.4f}, '
            f'Recall: {test_recall:.4f}, '
            f'avg_roc_auc: {test_roc_auc:.4f}, '
            f'avg_pr_auc: {test_pr_auc:.4f}'

        )
        tqdm.write(log_message)  
        log_file.write(log_message)  

        pbar.update(1)  

    final_results_file_path = "Final_Results.txt"

    with open(final_results_file_path, "w") as results_file:
        final_results = (
                "\n final result：\n"
                "F1-macro: {:.4f}\n".format(np.mean(F1)) +
                "F1-weight: {:.4f}\n".format(np.mean(F1_weight)) +
                "Accuracy: {:.4f}\n".format(np.mean(Accuracy)) +
                "Precision: {:.4f}\n".format(np.mean(Precision)) +
                "Recall: {:.4f}\n".format(np.mean(Recall)) +
                "roc_auc: {:.4f}\n".format(np.mean(avg_roc_auc)) +
                "pr_auc: {:.4f}\n".format(np.mean(avg_pr_auc))
        )
        tqdm.write(final_results)  
        results_file.write(final_results)  

    pbar.close()  



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_file_path', type=str, default="./data/gene_end1.csv")
    parser.add_argument('--edge_file_path', type=str, default="./data/ppi_end1.txt")
    parser.add_argument('--shuffle', type=str, default='eval_total') 
    parser.add_argument('--encoder', type=str, default='Random') 
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--episodic_samp', type=float, default=1.0) 
    parser.add_argument('--runs', type=int, default=5) 
    # parser.add_argument('--label_prop', type=str, default='yes') 
    # parser.add_argument('--eta', type=float, default=1.0)  

    parser.add_argument('--ssl2', type=str, default='yes')  
    parser.add_argument('--ssl3', type=str, default='no')  

    parser.add_argument('--dropout', type=float, default=0.5)  
    parser.add_argument('--lamb1', type=float, default=10)
    parser.add_argument('--lamb2', type=float, default=10)

    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--n_hidden', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()
    main(args)

