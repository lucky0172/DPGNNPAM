from learn import *
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc



def save_model(encoder, dist_encoder, args):
    if args.encoder == 'GCN':
        torch.save(encoder.state_dict(), 'encoder_gcn.pkl')
        torch.save(dist_encoder.state_dict(), 'dist_encoder_gcn.pkl')
    elif args.encoder == 'GAT':
        torch.save(encoder.state_dict(), 'encoder_gat.pkl')
        torch.save(dist_encoder.state_dict(), 'dist_encoder_gat.pkl')
    elif args.encoder == 'GAE':
        torch.save(encoder.state_dict(), 'encoder_gae.pkl')
        torch.save(dist_encoder.state_dict(), 'dist_encoder_gae.pkl')
    elif args.encoder == 'Graphsage':
        torch.save(encoder.state_dict(), 'encoder_graphsage.pkl')
        torch.save(dist_encoder.state_dict(), 'dist_encoder_graphsage.pkl')
    elif args.encoder == 'GCN_GAT':
        torch.save(encoder.state_dict(), 'encoder_gcn_gat.pkl')
        torch.save(dist_encoder.state_dict(), 'dist_encoder_gcn_gat.pkl')
    else:
        torch.save(encoder.state_dict(), 'encoder_random_ssl2.pkl')
        torch.save(dist_encoder.state_dict(), 'dist_encoder_random_ssl2.pkl')



def plot_roc_and_pr(y_true, y_pred_probs):
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    pr_auc = auc(recall, precision)

    return roc_auc, pr_auc
