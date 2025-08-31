
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
CUDA_LAUNCH_BLOCKING=1
# from utils import load_data,accuracy,load_graph,reload_graph,mask_test_edges,get_roc_score
from utils import load_data,accuracy,load_graph,mask_test_edges,get_roc_score,knn_cosine_similarity,knn2_cosine_similarity
from self_attention import Co_attention,GC_Contrastive_Learning,Co_attention1
import torch
import torch.optim as optim
import torch.nn.functional as F

import time
import argparse
import numpy as np

from config import Config
from sklearn.metrics import f1_score
from noise import noisify_p
import Kmeans_torch
from sklearn.cluster import KMeans
from embedding_visualization import visualization
import random
# from visdom import Visdom
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')

parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')

parser.add_argument('--seed', type=int, default=42, help='Random seed.')

parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')

parser.add_argument('--lr', type=float, default= 0.0002,
                    help='Initial learning rate.')

# default = 5e-4
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')

parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')

parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument("-d", "--dataset", help="dataset", type=str, default="acm")
parser.add_argument("-l", "--labelrate", help="labeled data for train per class", type = int, default=75)

# args = parser.parse_args()
args, unknown = parser.parse_known_args()

config_file = "./config/" + str(args.labelrate) + str(args.dataset) + ".ini"
# config_file = "./config/" + str(args.dataset) + ".ini"
config = Config(config_file)

args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()

# [batch,n,feature]
# features = torch.unsqueeze(features,dim=1)

# sadj, fadj = load_graph(args.labelrate, config)
# feature_edges = np.genfromtxt("./data/coraml/test20.txt", dtype=np.int32)
# print(feature_edges)
features, labels, idx_train, idx_test = load_data(config)
sadj, fadj,adj_1 = load_graph(args.labelrate, config)

# adj_train, train_edges, train_edges_false, val_edges, val_edges_false, \
#     test_edges, test_edges_false = mask_test_edges(adj_1, test_frac=0.1, val_frac=0.05)

# Model and optimizer
# model = Co_attention(num_attention_heads = 1,
#                       input_size = features.shape[-1],
#                       hidden_size1 = 300,
#                       hidden_size2 = 150,
#                       hidden_dropout_prob = 0,
#                       class_size = labels.max().item() + 1)
# viz = Visdom(port=6006)

# i = 42
# max_i=0
# max_iacc=0
# step = 0
# while i<1024:
model1 = GC_Contrastive_Learning(input_size = features.shape[-1], hidden_size = config.nhid1)

model2 = Co_attention1(num_attention_heads = config.num_attention_heads,input_size = config.nhid1,
                       hidden_size = config.nhid2,hidden_dropout_prob = config.dropout,class_size = labels.max().item() + 1)

# optimizer = optim.Adam(model.parameters(),
#                        lr=args.lr, weight_decay=args.weight_decay)

optimizer1 = optim.Adam(model1.parameters(),
                       lr=config.lr, weight_decay=config.weight_decay)

optimizer2 = optim.Adam([
                {'params': model2.parameters()},
                {'params': model1.parameters(), 'lr': 1e-4}
            ],lr=config.lr, weight_decay=config.weight_decay)

if args.cuda:
    # model.cuda()
    model1.cuda()
    model2.cuda()
    features = features.cuda()
    # adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    sadj = sadj.cuda()
    fadj = fadj.cuda()

size = features.shape[0]
tmp = [i for i in range(size)]
index = [0 for i in range(size)]
# file_path2 = '/data/ludan/PycharmProject/model/Contrastive_Learning_Coattention_shared_weight/data/chameleon/11'
# index = []  
#
# with open(file_path2, 'r') as file2:
#     for line in file2:
#         fields = line.strip().split()
#         index.extend(fields)
# print(index)
reload_feature = features
global_step = 0

def semi_shuffle(labels,train_idx):
    # random.seed(seed)
    # labels = torch.tensor([0, 1, 2, 0, 2, 2, 1, 2, 1, 0, 2, 0, 1])
    # print(id(labels))
    op_labels = labels.data.cpu().numpy().tolist()
    # print(id(op_labels))
    # aa = [1,2,3]
    # index_list = copy.copy(list(op_labels))
    # index_list.append(66)

    op_train_idx = train_idx.data.cpu().numpy().tolist()

    rest_op_train_idx = np.delete([i for i in range(len(op_labels))],op_train_idx)

    class_num = max(op_labels) + 1
    class_count = class_num  

    class_dict = [{} for i in range(class_num)]

    for i, v in enumerate(op_labels):
        class_dict[v][i] = v

    flag = 0
    dict_list = [i for i in range(class_num)]

    # labels_num = int(len(op_labels) * rate) 

    # labels_remain = len(op_labels) - labels_num + 1

    for i in op_train_idx:
        if class_count == 1:
            for j in class_dict:
                if len(j) != 0:
                    flag = 1
                    break
        if flag == 1:
            break

        dict_index = random.sample(dict_list, 1)[0]

        while dict_index == op_labels[i]:
            dict_index = random.sample(dict_list, 1)[0]

        tmp = random.sample(class_dict[dict_index].keys(), 1)[0]
        op_labels[i] = tmp
        del (class_dict[dict_index][tmp])
        if (len(class_dict[dict_index]) == 0):
            class_count -= 1
            dict_list.remove(dict_index)

    for i in rest_op_train_idx:
        dict_index = random.sample(dict_list, 1)[0]

        tmp = random.sample(class_dict[dict_index].keys(), 1)[0]
        op_labels[i] = tmp

        del (class_dict[dict_index][tmp])
        if (len(class_dict[dict_index]) == 0):
            class_count -= 1
            dict_list.remove(dict_index)
    return op_labels

def train(epoch):
    global reload_feature
    global fadj
    global global_step
    t = time.time()
    if (epoch % 5 == 0 and epoch != 0):
        fadj1 = knn2_cosine_similarity(reload_feature, config, fadj, weight=1.0)
        fadj2 = knn_cosine_similarity(reload_feature, config, fadj, weight=1.0)
        m = 0.3
        fadj1 = fadj1 * m
        fadj2 = fadj2 * (1-m)
        fadj = fadj1 + fadj2
        fadj = fadj.cuda()
    # if (epoch % 40 == 0 and epoch != 0):
    #     fadj = reload_graph(reload_feature, config,fadj)
    #     fadj = fadj.cuda()815

    model1.train() 
    optimizer1.zero_grad()
    feature_g, feature_f = model1(features, sadj, fadj)
    index = semi_shuffle(labels,idx_train)
    feature_f_ne = feature_f[index]

    # print(feature_f_ne)
    loss1 = F.triplet_margin_loss(feature_g, feature_f, feature_f_ne, margin=0.8,reduction='mean')
    loss1.backward()
    optimizer1.step()

    global_step += 1
    # viz.line([loss1.item()], [global_step], win='loss1', update='append')

    model2.train()
    optimizer2.zero_grad()

    feature_g, feature_f = model1(features, sadj, fadj)
    reload_feature = feature_f
    # feature_g.detach_()
    # feature_f.detach_()
    output,x = model2(feature_g,feature_f)

    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer2.step()
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t))
    # return acc_train

def test():
    # model.eval()
    # output = model(features,sadj=sadj,fadj=fadj)
    # output = output.view(output.shape[0], -1)
    model1.eval()
    model2.eval()
    feature_g, feature_f = model1(features, sadj, fadj)
    output,x = model2(feature_g, feature_f)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    centroids = Kmeans_torch.train(data = output, num_clustres=3,max_iterations=250)
    output2 = torch.cat((output,centroids),0)

    cluster_label = torch.tensor([7,7,7]).cuda()
    labels2 = torch.cat((labels,cluster_label))

    kmean = KMeans(n_clusters=3, n_init=10)
    output_copy = output.cpu().detach().numpy()
    pre_z = kmean.fit_predict(output_copy)
    visualization(output2,labels2)
    centroids = torch.tensor(kmean.cluster_centers_).cuda()
    output3 = torch.cat((output,centroids),0)
    # visualization(output3, labels2)
    # print("Test set results:",
    #       "loss= {:.4f}".format(loss_test.item()),
    #       "accuracy= {:.4f}".format(acc_test.item()))

# Train model
t_total = time.time()
max_acc = 0
loss = 100
max_epoch = 0
f1_max = 0
sc_roc_max = 0
sc_ap_max = 0

for epoch in range(config.epochs):
    train(epoch)
    model1.eval()
    model2.eval()
    feature_g, feature_f = model1(features, sadj, fadj)
    output,x = model2(feature_g, feature_f)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    # sc_roc, sc_ap = get_roc_score(test_edges, test_edges_false, x.cpu().detach().numpy(), adj_1)
    # print('AUC:{:.4f}'.format(sc_roc), 'AP:{:.4f}'.format(sc_ap))

    # if sc_roc > sc_roc_max:
    #     sc_roc_max = sc_roc
    # if sc_ap > sc_ap_max:
    #     sc_ap_max = sc_ap

    label_max = []
    for idx in idx_test:
        label_max.append(torch.argmax(output[idx]).item())
    labelcpu = labels[idx_test].data.cpu()
    macro_f1 = f1_score(labelcpu, label_max, average='macro')



    # print("acc_test:",acc_test)
    # print("max_acc:",max_acc)
    # print("acc_test:",type(acc_test))
    # print("max_acc:",type(max_acc))
    if acc_test >= max_acc:
        max_acc = acc_test
        if acc_test == max_acc and macro_f1 > f1_max:
            f1_max = macro_f1
        loss = loss_test
        max_epoch = epoch

print("Test set results:",
        "loss= {:.4f}".format(loss.item()),
        "acc= {:.4f}".format(max_acc.item()),
         # "max_epoch = {}".format(max_epoch),
        "f1: {:.4f}".format(f1_max),
         # "sc_ap_max:{:.4f}".format(sc_ap_max),
        # "sc_roc_max:{:.4f}".format(sc_roc_max)
      )
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()


