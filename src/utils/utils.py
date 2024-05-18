from community import community_louvain
import csv
import networkx as nx
import numpy as np
import pandas as pd
import pyod.models.iforest
import sklearn.preprocessing as skpp
import scipy.io as sio
import scipy.linalg as linalg
import scipy.sparse as sp
import scipy.stats as st
import torch

# local imports
import utils.graph_to_egonet_features as g2f

def setup(graph_name='libra', requested_signals=['f_amount_in', 'f_amount_out', 'f_nr_trans_in', 'f_nr_trans_out'], weights_normalization=None, signals_normalization=None):
    """Setup function to prepare the working graph
    
    Parameters
    ----------
    graph_name : graph identifier, default='libra'
        Available graphs:
            * 'v9'
            * 'v10'
            * 'v12'
            * 'v13'
                 
    requested_signals: list of signals/features identifiers, default=['f_amount_in', 'f_amount_in', 'f_nr_trans_in', 'f_nr_trans_out']
        Available signals/features:
            * 'f_degree_in'
            * 'f_degree_out'
            * 'f_amount_in'
            * 'f_amount_out'
            * 'f_nr_trans_in'
            * 'f_nr_trans_out'
            * 'f_ego_nr_nodes'
            * 'f_ego_nr_edges',
            * 'f_egored_degree_in'
            * 'f_egored_degree_out'
            * 'f_egored_amount_in'
            * 'f_egored_amount_out'
            * 'f_egored_nr_trans_in'
            * 'f_egored_nr_trans_out
            * 'f_egored_nr_nodes'
            * 'f_egored_nr_edges'
            * other columns indicated by summable attributes as described in graph_to_egonet_features.py

    normalization: normalization type identifier, default=None
        Available normalization types:
            * 'l1'
            * 'l2'
            * 'max'
            * 'log10'

    Returns
    -------
    G               : DiGraph
    adj_sym         : symmterized adjacency matrix
    adj_skew_sym    : skew-symmetrized adjacency matrix
    signals         : graph node signals (nb_nodes x nb_signals)
    labels          : flag specifying whether node at corresponding index is anomalous or not
    weighted_labels : anomaly weight for node at corresponding index
    """

    is_libra = False

    if graph_name == 'libra':
        is_libra = True
    if is_libra:
        base_name = './dataset/Libra_bank_3months_graph'
    else:
        base_name = './dataset/synth_graph_'

    # Extract egonet features from graph
    if is_libra == False:
        egonet_file_name = base_name + graph_name + '_ego_features.csv'
        train_graph_file = base_name + graph_name + '.csv'
    else:
        egonet_file_name = './dataset/libra_egonet_features.csv'
        train_graph_file = base_name + '.csv'
    
    train_df_graph = pd.read_csv(train_graph_file)
    
    try:                                                           # read from precomputed file if available
        train_node_features_ego = pd.read_csv(egonet_file_name)
    except FileNotFoundError:                                      # if file not exists, create it (possibly time consuming operation)
        FULL_egonets = True                                        # build full egonet, as for undirected graphs
        IN_egonets = False                                         # normal value is False, as out egonets are implicit in Networkx
                                                                   # it is used only if FULL_egonets is False
        if is_libra:
            summable_attr = ["nr_alerts", "nr_reports"]
        else:
            summable_attr = ["nr_alerts"]

        train_node_features_ego = g2f.graph_to_egonet_features(train_df_graph, FULL_egonets=FULL_egonets, IN_egonets=IN_egonets, \
                            summable_attributes=summable_attr, verbose=False)
    
        # save feature file as csv
        train_node_features_ego.to_csv(egonet_file_name, index=False)

    G = nx.from_pandas_edgelist(df=train_df_graph, source='id_source', target='id_destination',
                            edge_attr=True, create_using=nx.DiGraph)

    Aa = train_df_graph.loc[train_df_graph['nr_alerts'] > 0]
    anomalous_nodes = set()
    for node in Aa['id_source'].values:
        anomalous_nodes.add(node)
    for node in Aa['id_destination'].values:
        anomalous_nodes.add(node)

    labels = torch.zeros(G.number_of_nodes())
    for anomalous_node in anomalous_nodes:
        labels[anomalous_node] = 1

    weighted_labels = np.zeros(nx.number_of_nodes(G))
    for node in range(nx.number_of_nodes(G)):
        weighted_labels[node] += train_node_features_ego["nr_alerts"][node]

    w = nx.get_edge_attributes(G, 'cum_amount')
    if weights_normalization != None:
        if weights_normalization == 'l1' or weights_normalization == 'l2' or weights_normalization == 'max':
            w_values = torch.tensor(list(w.values())).reshape(-1, 1)
            w_values = torch.tensor(skpp.normalize(w_values, weights_normalization, axis=0).flatten())
            w = {k: v for k, v in zip(w, w_values)}
        elif weights_normalization == 'log10':
            w_values = torch.tensor(list(w.values())).reshape(-1, 1)
            w_values = torch.log10(1 + w_values).flatten()
            w = {k: v for k, v in zip(w, w_values)}
        elif weights_normalization == 'log2':
            w_values = torch.tensor(list(w.values())).reshape(-1, 1)
            w_values = torch.log2(1 + w_values).flatten()
            w = {k: v for k, v in zip(w, w_values)}
        elif weights_normalization == 'log':
            w_values = torch.tensor(list(w.values())).reshape(-1, 1)
            w_values = torch.log(1 + w_values).flatten()
            w = {k: v for k, v in zip(w, w_values)}

    nx.set_edge_attributes(G, w, 'weight')
    
    dict_signals = {}
    for requested in requested_signals:
        dict_signals[requested] = train_node_features_ego[requested]
    signals = pd.DataFrame(dict_signals)
    signals = torch.tensor(signals.values)
    signals = signals.to(torch.float32)
    if signals_normalization != None:
        if signals_normalization == 'l1' or signals_normalization == 'l2' or signals_normalization == 'max':
            signals = torch.tensor(skpp.normalize(signals, signals_normalization, axis=0))
        elif signals_normalization == 'log10':
            signals = torch.log10(1 + signals)
        elif signals_normalization == 'log2':
            signals = torch.log2(1 + signals)
        elif signals_normalization == 'log':
            signals = torch.log(1 + signals)
    adj = nx.adjacency_matrix(G, weight='weight') 
    adj_sym = 0.5*(adj + adj.transpose())
    adj_skew_sym = 0.5*(adj - adj.transpose())

    return G, adj, adj_sym, adj_skew_sym, signals, labels, weighted_labels


def compute_scaled_normalized_laplacian(A, renormalize=True, lambda_max=None):
    if renormalize == True:
        A = A + sp.identity(A.shape[0])
    
    # Symmetrically normalized adjacency: D^(-1/2) * A * D^(-1/2)
    
    # abs() needed for the skew-symmetric adjacency
    # no effect on the symmetric adjacency since entries are positive
    A = abs(A)

    d = A.sum(axis=1)
    d = np.power(d, -0.5)
    d[np.isinf(d)] = 0.
    D = sp.diags(np.squeeze(np.asarray(d)), format='csc')

    A_n = D.dot(A).dot(D)
    L = sp.identity(A.shape[0]) - A_n

    # L = (2 / lambda_max) * L - I

    if sp.issparse(L):
        id = sp.identity(L.shape[0], format='csc')
        # get the maximum eigenvalue
        if lambda_max == None:
            eigval_max = max(sp.linalg.eigsh(A=L, k=1, which='LM', return_eigenvectors=False))
        else:
            eigval_max = lambda_max
    else:
        id = np.identity(L.shape[0])
        if lambda_max == None:
            eigval_max = max(linalg.eigvals(a=L).real)
        else:
            eigval_max = lambda_max
    
    # If the L is symmetric or random walk normalized Laplacian,
    # then the maximum eigenvalue has to be smaller than or equal to 2.
    if eigval_max >= 2:
        L = L - id
    else:
        L = 2 * L / eigval_max - id
    return L


def cnv_sparse_mat_to_coo_tensor(sp_mat):
    # convert a compressed sparse row (csr) or compressed sparse column (csc) matrix to a hybrid sparse coo tensor
    sp_coo_mat = sp_mat.tocoo()
    i = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)))
    v = torch.from_numpy(sp_coo_mat.data)
    s = torch.Size(sp_coo_mat.shape)

    if sp_mat.dtype == np.complex64 or sp_mat.dtype == np.complex128:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.complex64, requires_grad=False)
    elif sp_mat.dtype == np.float32 or sp_mat.dtype == np.float64:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.float32, requires_grad=False)
    else:
        raise TypeError(f'ERROR: The dtype of {sp_mat} is {sp_mat.dtype}, not been applied in implemented models.')


def evaluate(labels, weighted_labels, error):
    TPR_milestones_perc = [0.1, 0.2, 0.5, 1]  # percentage in all nodes where to compute TPR values
    AUC_milestones_perc = [1, 100]            # same, for AUC computation
    
    nbNodes = len(labels)

    sorted_indices = torch.argsort(torch.sum(error, dim=1), descending=True)
    sorted_labels = labels[sorted_indices]
    sorted_weighted_labels = weighted_labels[sorted_indices]

    anomalous_nodes = np.cumsum(np.array(sorted_labels))
    tpr = np.cumsum(np.array(sorted_weighted_labels)) / sum(sorted_weighted_labels)

    p_perc = np.array(TPR_milestones_perc)/100*nbNodes
    p_perc = p_perc.astype('int')
    
    nodes = []
    weights = []
    TPR = []
    TPR_AUC = []
    print("True positives detected in first", *TPR_milestones_perc, "% anomalies")
    for i in range(len(p_perc)):
        print("{} nodes, with weight {:.4f} and TPR {:.4f}".format(anomalous_nodes[np.array(p_perc[i]).astype('int')], tpr[np.array(p_perc[i]).astype('int')]*sum(sorted_weighted_labels), tpr[np.array(p_perc[i]).astype('int')]), end=' ') 
        print("")
        nodes.append(str(anomalous_nodes[np.array(p_perc[i]).astype('int')]))
        weights.append(str(tpr[np.array(p_perc[i]).astype('int')]*sum(sorted_weighted_labels)))
        TPR.append(str(tpr[np.array(p_perc[i]).astype('int')]))

    a_perc = np.array(AUC_milestones_perc)/100*nbNodes
    a_perc = a_perc.astype('int')
    print("TPR AUC in first", *AUC_milestones_perc, "% anomalies")
    for i in range(len(a_perc)):
        print("{:.4f}".format(np.average(tpr[:a_perc[i]])), end=' ')
        TPR_AUC.append(str(np.average(tpr[:a_perc[i]])))
    
    data = [nodes[0], weights[0], TPR[0], nodes[1], weights[1], TPR[1], nodes[2], weights[2], TPR[2], nodes[3], weights[3], TPR[3], TPR_AUC[0], TPR_AUC[1]]
    print("")
    print("")

    # import matplotlib.pyplot as plt
    # plt.plot(tpr)
    # plt.title(graph_name + ' Alerts: True positive rate')
    # plt.grid()
    # plt.show()
    return data

def write_csv_header(filename, isscore=False):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        if isscore == True:
            field = ["Graph", "Density", "Weights norm.", \
             "0.1 AN", "0.1 W", "0.1 TPR", "0.2 AN", "0.2 W", "0.2 TPR", "0.5 AN", "0.5 W", "0.5 TPR", "1 AN", "1 W", "1 TPR", "TPR AUC 1", "TPR AUC 100"]   
        else:
            field = ["Graph", "Density", "Features norm.", "Weights norm.", "Sym. renorm.", "K", "Num. layers E/D", "Hidden dim.", "Lr", "Epochs", "Loss", "Time", \
             "0.1 AN", "0.1 W", "0.1 TPR", "0.2 AN", "0.2 W", "0.2 TPR", "0.5 AN", "0.5 W", "0.5 TPR", "1 AN", "1 W", "1 TPR", "TPR AUC 1", "TPR AUC 100"]  
        writer.writerow(field)
