import torch
from torch.nn import CrossEntropyLoss

from detecter.model import AST_GRU
from detecter.model import Similarity
from detecter.dataset.OJClone import DataLoader
from detecter.parser.code2tree import Tree

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

batch_size = 20
loader = DataLoader("dataset/OJClone/test.jsonl", batch_size)
v_loader = DataLoader("dataset/OJClone/valid.jsonl", batch_size)

# model = AST_GAT(300).cuda()
model = AST_GRU(300).cuda()
similarity = Similarity(model.EMBEDDING_SIZE).cuda()
# loss_func = BCEWithLogitsLoss()
loss_func = CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), 0.001)

# k = torch.zeros(1).cuda()
k = 0.5

for it in range(30):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for labels, tree in loader:
        try:
            optimizer.zero_grad()

            tree = tree.cuda()

            B = len(labels)
            hidden = model(tree.x, tree.edge_index) # B, 768
            hidden = hidden.index_select(0, tree.root)

            l_indexs = [i for i in range(B) for j in range(B)]
            r_indexs = [j for i in range(B) for j in range(B)]

            outputs = similarity.forward(hidden[l_indexs], hidden[r_indexs])
            lresults = [labels[i] == labels[j] for i in range(B) for j in range(B)]
            results = torch.tensor(lresults, dtype=torch.bool).cuda()

            t_outputs = outputs[:, 1] > outputs[:, 0]

            tp += torch.count_nonzero(torch.logical_and(t_outputs, results)).item()
            fp += torch.count_nonzero(torch.logical_and(~t_outputs, ~results)).item()
            fn += torch.count_nonzero(torch.logical_and(t_outputs, ~results)).item()
            tn += torch.count_nonzero(torch.logical_and(~t_outputs, results)).item()

            print(it, "rate", (tp + fn) / (tp + fp + fn + tn))
            precision = tp / (tp + fp)
            recal = tp / (tp + tn)
            print(it, "f1  ", 2 / (1 / precision + 1 / recal))

            loss = loss_func.forward(outputs, results.long())
            loss.backward()
            
            optimizer.step()
        except Exception as err:
            print("ERROR")
            raise err


    tp = 0
    fp = 0
    fn = 0
    tn = 0

    with torch.no_grad():
        for labels, tree in v_loader:
            tree: Tree = tree.cuda()
            # tree: Tree = tree.cuda()
            B = len(labels)
            hidden = model(tree.x, tree.edge_index)
            hidden = hidden.index_select(0, tree.root)

            l_indexs = [i for i in range(B) for j in range(B)]
            r_indexs = [j for i in range(B) for j in range(B)]

            outputs = similarity.forward(hidden[l_indexs], hidden[r_indexs])
            results = [labels[i] == labels[j] for i in range(B) for j in range(B)]
            results = torch.tensor(results, dtype=torch.bool).cuda()

            t_outputs = outputs[:, 1] > outputs[:, 0]

            tp += torch.count_nonzero(torch.logical_and(t_outputs, results)).item()
            fp += torch.count_nonzero(torch.logical_and(~t_outputs, ~results)).item()
            fn += torch.count_nonzero(torch.logical_and(t_outputs, ~results)).item()
            tn += torch.count_nonzero(torch.logical_and(~t_outputs, results)).item()
        
    print("=======================================")
    print(it, "rate", (tp + fn) / (tp + fp + fn + tn))
    precision = tp / (tp + fp)
    recal = tp / (tp + tn)
    print(it, "f1  ", 2 / (1 / precision + 1 / recal))
    print("=======================================")
