import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics.pairwise import cosine_similarity as cos
import scipy
import networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import copy

import random
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# random.seed("mark")

def encode_onehot(labels):
  
    classes = set(labels)
   
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

 return adj, features, labels, idx_train, idx_val, idx_test

def normalize(mx):
    """Row-normalize sparse matrix"""
  
    rowsum = np.array(mx.sum(1))

    r_inv = np.power(rowsum, -1).flatten()

    r_inv[np.isinf(r_inv)] = 0.

    r_mat_inv = sp.diags(r_inv)

    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):

    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def load_data(config):
    # f = np.loadtxt(config.feature_path, dtype=float, delimiter=',')
    f = np.loadtxt(config.feature_path, dtype=float)
    l = np.loadtxt(config.label_path, dtype = int)
    test = np.loadtxt(config.test_path, dtype = int)
    train = np.loadtxt(config.train_path, dtype = int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))
    #---
    # features = normalize_features(features)
    # features = torch.tensor(features)

    idx_test = test.tolist()
    idx_train = train.tolist()

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    label = torch.LongTensor(np.array(l))

    return features, label, idx_train, idx_test

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_graph(dataset, config):
# def load_graph(config):
    featuregraph_path = config.featuregraph_path + str(config.k) + '.txt'
    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))

    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = normalize(sadj+sp.eye(sadj.shape[0])) 

    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    adj_1 = sp.csr_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    # for i in range(2708):
    #     for j in range(2708):
    #         if adj_1.A[i][j] !=  adj_1.A[j][i]:
    #             print("false")

    # nfadj = torch.eye(config.n)
    # print(adj_1.shape)

    return nsadj, nfadj,adj_1

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def knn2_cosine_similarity(features, config,fadj_test,weight=1.0):
    edges = []
    features = features.data.cpu()
    dist = cos(features) 
    inds = []
    for i in range(dist.shape[0]):  
        ind = np.argpartition(dist[i, :], config.k + 1)[:config.k + 1]
        ind = ind[ind != i] 
        inds.append(ind)

    for i, v in enumerate(inds):  
        for vv in v:  
            if vv <= i:  
                pass
            else:  
                edges.append([i, vv, weight])

    edges_feature = np.array(list(edges), dtype=np.int32)
    fadj = sp.coo_matrix((edges_feature[:, 2], (edges_feature[:, 0], edges_feature[:, 1])),
                          shape=(config.n, config.n), dtype=np.float32)
    fadj = fadj + fadj.T - sp.diags(fadj.diagonal()) 
    nfadj = normalize(fadj + sp.eye(fadj.shape[0])) 
    fadj1 = sparse_mx_to_torch_sparse_tensor(nfadj) 
    return fadj1 


def knn_cosine_similarity(features,config,fadj_test,weight=1.0):
    edges = []
    features = features.data.cpu()
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):  
        ind = np.argpartition(dist[i, :], -(config.k + 1))[-(config.k + 1):]
        inds.append(ind)

    for i, v in enumerate(inds):  
        for vv in v:  
            if vv <= i:    
                pass
            else:   
                # f.write('{} {}\n'.format(i, vv))
                edges.append([i,vv,weight])  
    edges_feature = np.array(list(edges), dtype=np.int32)
    fadj = sp.coo_matrix((edges_feature[:, 2], (edges_feature[:, 0], edges_feature[:, 1])),
                         shape=(config.n, config.n),dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)  
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))  
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)  
    # count = 0
    # for i in range(nfadj.shape[0]):
    #     if(not nfadj.to_dense()[i].cuda().equal(fadj_test.to_dense()[i])):
    #         count +=1
    # print(count)
    return nfadj   
# def reload_graph(features,config,fadj_test):
#
#     edges = []
#     features = features.data.cpu()
#     dist = cos(features)
#     inds = []
#     for i in range(dist.shape[0]):
#         ind = np.argpartition(dist[i, :], -(config.k + 1))[-(config.k + 1):]
#         inds.append(ind)
#
#     for i, v in enumerate(inds):
#         for vv in v:
#             if vv <= i:
#                 pass
#             else:
#                 # f.write('{} {}\n'.format(i, vv))
#                 edges.append([i,vv])
#
#     edges_feature = np.array(list(edges), dtype=np.int32)
#
#     fadj = sp.coo_matrix((np.ones(edges_feature.shape[0]), (edges_feature[:, 0], edges_feature[:, 1])), shape=(config.n, config.n),
#                          dtype=np.float32)
#     fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
#     nfadj = normalize(fadj + sp.eye(fadj.shape[0]))
#     nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
#     # count = 0
#     # for i in range(nfadj.shape[0]):
#     #     if(not nfadj.to_dense()[i].cuda().equal(fadj_test.to_dense()[i])):
#     #         count +=1
#     # print(count)
#     return nfadj

# def semi_shuffle(labels,rate):
#     # labels = torch.tensor([0, 1, 2, 0, 2, 2, 1, 2, 1, 0, 2, 0, 1])
#     # print(id(labels))
#     op_labels = labels.data.cpu().numpy().tolist()
#     # print(id(op_labels))
#     # aa = [1,2,3]
#     # index_list = copy.copy(list(op_labels)) 
#     # index_list.append(66)
#     class_num = max(op_labels) + 1
#     class_count = class_num 
#
#     
#     class_dict = [{} for i in range(class_num)]
#
#     for i, v in enumerate(op_labels):
#         class_dict[v][i] = v
#
#     flag = 0
#     dict_list = [i for i in range(class_num)]
#
#     labels_num = int(len(op_labels) * rate)  
#
#     # labels_remain = len(op_labels) - labels_num + 1
#
#     for i in range(int(labels_num)):
#         if class_count == 1:
#             for j in class_dict:
#                 if len(j) != 0:
#                     flag = 1
#                     break
#         if flag == 1:
#             break
#
#         dict_index = random.sample(dict_list, 1)[0]
#
#         while dict_index == op_labels[i]:
#             dict_index = random.sample(dict_list, 1)[0]
#
#         tmp = random.sample(class_dict[dict_index].keys(), 1)[0]
#         op_labels[i] = tmp

#         del (class_dict[dict_index][tmp])
#         if (len(class_dict[dict_index]) == 0):
#             class_count -= 1
#             dict_list.remove(dict_index)
#
#     for i in range(labels_num, len(op_labels)):
#   
#         dict_index = random.sample(dict_list, 1)[0]
#
#         tmp = random.sample(class_dict[dict_index].keys(), 1)[0]
#         op_labels[i] = tmp
#
#  
#         del (class_dict[dict_index][tmp])
#         if (len(class_dict[dict_index]) == 0):
#             class_count -= 1
#             dict_list.remove(dict_index)
#
#     return op_labels

def semi_shuffle(labels,train_idx,seed):
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


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not scipy.sparse.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def mask_test_edges(adj, test_frac=.1, val_frac=.05, prevent_disconnect=True, verbose=False):
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    "from https://github.com/tkipf/gae"

    if verbose == True:
        print('preprocessing...')

   
    adj = adj - scipy.sparse.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)

    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0
    g = nx.from_scipy_sparse_matrix(adj)

    orig_num_cc = nx.number_connected_components(g)

    adj_triu = scipy.sparse.triu(adj)  # upper triangular portion of adj matrix

    adj_tuple = sparse_to_tuple(adj_triu)  # (coords, values, shape), edges only 1 way
    edges = adj_tuple[0]  # all edges, listed only once (not 2 ways)
    # edges_all = sparse_to_tuple(adj)[0] # ALL edges (includes both ways)
    num_test = int(np.floor(edges.shape[0] * test_frac))  # controls how large the test set should be
    num_val = int(np.floor(edges.shape[0] * val_frac))  # controls how alrge the validation set should be

    # Store edges in list of ordered tuples (node1, node2) where node1 < node2
    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
    all_edge_tuples = set(edge_tuples)
    train_edges = set(edge_tuples)  # initialize train_edges to have all edges
    test_edges = set()
    val_edges = set()

    if verbose == True:
        print('generating test/val sets...')

    # Iterate over shuffled edges, add to train/val sets
    np.random.shuffle(edge_tuples)
    for edge in edge_tuples:
        # print edge
        node1 = edge[0]
        node2 = edge[1]

        # If removing edge would disconnect a connected component, backtrack and move on
        g.remove_edge(node1, node2)
        if prevent_disconnect == True:
            if nx.number_connected_components(g) > orig_num_cc:
                g.add_edge(node1, node2)
                continue

        # Fill test_edges first
        if len(test_edges) < num_test:
            test_edges.add(edge)
            train_edges.remove(edge)

        # Then, fill val_edges
        elif len(val_edges) < num_val:
            val_edges.add(edge)
            train_edges.remove(edge)

        # Both edge lists full --> break loop
        elif len(test_edges) == num_test and len(val_edges) == num_val:
            break

    if (len(val_edges) < num_val or len(test_edges) < num_test):
        print("WARNING: not enough removable edges to perform full train-test split!")
        print("Num. (test, val) edges requested: (", num_test, ", ", num_val, ")")
        print("Num. (test, val) edges returned: (", len(test_edges), ", ", len(val_edges), ")")

    if prevent_disconnect == True:
        assert nx.number_connected_components(g) == orig_num_cc

    if verbose == True:
        print('creating false test edges...')

    test_edges_false = set()
    while len(test_edges_false) < num_test:
        idx_i = np.random.randint(0, adj.shape[0]) 
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge not an actual edge, and not a repeat
        if false_edge in all_edge_tuples:
            continue
        if false_edge in test_edges_false:
            continue
        test_edges_false.add(false_edge)

    if verbose == True:
        print('creating false val edges...')

    val_edges_false = set()
    while len(val_edges_false) < num_val:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
                false_edge in test_edges_false or \
                false_edge in val_edges_false:
            continue
        val_edges_false.add(false_edge)

    if verbose == True:
        print('creating false train edges...')

    train_edges_false = set()
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false,
        # not in val_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
                false_edge in test_edges_false or \
                false_edge in val_edges_false or \
                false_edge in train_edges_false:
            continue

        train_edges_false.add(false_edge)

    if verbose == True:
        print('final checks for disjointness...')

    # assert: false_edges are actually false (not in all_edge_tuples)
    assert test_edges_false.isdisjoint(all_edge_tuples)
    assert val_edges_false.isdisjoint(all_edge_tuples)
    assert train_edges_false.isdisjoint(all_edge_tuples)

    # assert: test, val, train false edges disjoint
    assert test_edges_false.isdisjoint(val_edges_false)
    assert test_edges_false.isdisjoint(train_edges_false)
    assert val_edges_false.isdisjoint(train_edges_false)

    # assert: test, val, train positive edges disjoint
    assert val_edges.isdisjoint(train_edges)
    assert test_edges.isdisjoint(train_edges)
    assert val_edges.isdisjoint(test_edges)

    if verbose == True:
        print('creating adj_train...')

    # Re-build adj matrix using remaining graph
    adj_train = nx.adjacency_matrix(g)

    # Convert edge-lists to numpy arrays
    train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
    train_edges_false = np.array([list(edge_tuple) for edge_tuple in train_edges_false])
    val_edges = np.array([list(edge_tuple) for edge_tuple in val_edges])
    val_edges_false = np.array([list(edge_tuple) for edge_tuple in val_edges_false])
    test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])
    test_edges_false = np.array([list(edge_tuple) for edge_tuple in test_edges_false])

    if verbose == True:
        print('Done with train-test split!')
        print('')

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, \
           val_edges, val_edges_false, test_edges, test_edges_false

def get_roc_score(edges_pos, edges_neg, embeddings, adj_sparse):
    "from https://github.com/tkipf/gae"

    score_matrix = np.dot(embeddings, embeddings.T)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Store positive edge predictions, actual values
    preds_pos = []
    pos = []
    for edge in edges_pos:
        preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]]))  # predicted score
        pos.append(adj_sparse[edge[0], edge[1]])  # actual value (1 for positive)

    # Store negative edge predictions, actual values
    preds_neg = []
    neg = []
    for edge in edges_neg:
        preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]]))  # predicted score
        neg.append(adj_sparse[edge[0], edge[1]])  # actual value (0 for negative)

    # Calculate scores
    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])

    # print(preds_all, labels_all )

    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score