from utils import *
import torch
import copy
from sklearn.metrics import *
from collections import Counter

def train(encoder, dist_encoder, prototype, data, optimizer, criterion, args,config):
    encoder.train()

    support, query = episodic_generator(data, args.episodic_samp, args.classes, data.x.size(0))  

    if (args.encoder == 'GCN' or args.encoder == 'GAT' or args.encoder == 'Graphsage' or args.encoder == 'GCN_GAT'):
        embedding = encoder(data)
    else:
        embedding = encoder(data,config)

    support_embed = [embedding[support[i]] for i in range(len(args.classes))]
    query_embed = [embedding[query[i]] for i in range(len(args.classes))]

    query_embed = torch.stack(query_embed, dim=0)  
    proto_embed = [prototype(support_embed[i]) for i in range(len(args.classes))]
    proto_embed = torch.stack(proto_embed,dim = 0)

    query_dist_embed = dist_encoder(query_embed, proto_embed, args.classes)
    proto_dist_embed = dist_encoder(proto_embed, proto_embed, args.classes)

    logits = torch.log_softmax(torch.mm(query_dist_embed, proto_dist_embed), dim=1)

    loss1 = criterion(logits, args.classes)  
    loss2 = 0
    loss3 = 0

    if(args.ssl2 == 'yes'):
        class_sim = cos_sim_pair(proto_embed)
        loss2 = (torch.sum(class_sim) - torch.trace(class_sim)) / ((class_sim.size(0)**2 - class_sim.size(0)) / 2)

    if(args.ssl3 == 'yes'): 
        dist_embed = dist_encoder(embedding, proto_embed, args.classes)
        loss3 = torch.mean((dist_embed[data.edge_index[0]] * args.deg_inv_sqrt[data.edge_index[0]].view(-1, 1) -
                            dist_embed[data.edge_index[1]] * args.deg_inv_sqrt[data.edge_index[1]].view(-1, 1)) ** 2)


    loss = loss1 + args.lamb1 * loss2 + args.lamb2 * loss3  

    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()


def test(encoder, dist_encoder, prototype, data, args, config):
    encoder.eval()

    with torch.no_grad():
        if args.encoder in ['GCN', 'GAT', 'Graphsage', 'GCN_GAT']:
            embedding = encoder(data)
        elif args.encoder == 'GAE':
            embedding = encoder.encode(data)
        else:
            embedding = encoder(data, config)

        support, query = episodic_generator(data, 1, args.classes, data.x.size(0))  # 生成样本
        support_embed = [embedding[support[i]] for i in range(len(args.classes))]

        proto_embed = [prototype(support_embed[i]) for i in range(len(args.classes))]
        proto_embed = torch.stack(proto_embed, dim=0) 

        f1, f1w, acc, precisions, recalls, confusion_matrices = [], [], [], [], [], []
        preds, ys, pred_probs = [], [], []
        for _, mask in data('test_mask'):
            y = data.y[mask]  
            query_embed = embedding[mask]  

            query_dist_embed = dist_encoder(query_embed, proto_embed, args.classes)
            proto_dist_embed = dist_encoder(proto_embed, proto_embed, args.classes)
            logits = torch.log_softmax(torch.mm(query_dist_embed, proto_dist_embed), dim=1)

            probs = torch.softmax(logits, dim=1)

            threshold = 0.6

            pred = (probs[:, 1] > threshold).long() 

            acc.append(pred.eq(y).sum().item() / mask.sum().item())
            f1.append(f1_score(y.tolist(), pred.tolist(), average='binary', pos_label=1, zero_division=0))
            f1w.append(f1_score(y.tolist(), pred.tolist(), average='weighted', zero_division=0))
            precisions.append(precision_score(y.tolist(), pred.tolist(), average='binary', pos_label=1, zero_division=0))
            recalls.append(recall_score(y.tolist(), pred.tolist(), average='binary', pos_label=1, zero_division=0))

            cm = confusion_matrix(y.tolist(), pred.tolist(), labels=np.arange(0, len(args.classes)))

            preds.append(pred.cpu().numpy())
            ys.append(y.cpu().numpy())
            pred_probs.append(probs[:, 1].cpu().numpy()) 

    preds = np.concatenate(preds) if preds else np.array([])
    ys = np.concatenate(ys) if ys else np.array([])
    pred_probs = np.concatenate(pred_probs) if pred_probs else np.array([])

    return f1, f1w, acc, precisions, recalls, preds, ys, pred_probs, cm



def predict(encoder, dist_encoder, prototype, data, args,config):
    encoder.eval()
    dist_encoder.eval()

    with torch.no_grad():
        if args.encoder in ['GCN', 'GAT', 'Graphsage', 'GCN_GAT']:
            embedding = encoder(data)
        elif args.encoder == 'GAE':
            embedding = encoder.encode(data)
        else:
            embedding = encoder(data, config)
        labeled_nodes = data.train_mask.nonzero(as_tuple=True)[0]
        proto_embed = [prototype(embedding[(data.y == i) & data.train_mask]) for i in range(len(args.classes))]
        proto_embed = torch.stack(proto_embed, dim=0)

        query_dist_embed = dist_encoder(embedding, proto_embed, args.classes)
        proto_dist_embed = dist_encoder(proto_embed, proto_embed, args.classes)

        logits = torch.mm(query_dist_embed, proto_dist_embed.t())
        probabilities = torch.softmax(logits, dim=1)

    return probabilities


