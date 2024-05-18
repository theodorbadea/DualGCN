import torch
import networkx as nx
import time
import csv
# local imports
import utils
from DualGCN import DualGCN
import utils.utils as utils


# PARAMETERS 
graph_name = "libra"
hidden_dimensions = 2
number_layers = 1
K = 1
learning_rate = 0.01
signals_normalization = None
weights_normalization = None
requested_signals=["f_amount_in", "f_amount_out", "f_nr_trans_in", "f_nr_trans_out"]
rounds = 10
renormalize = True
####################################################################################

csvFileName = graph_name + '_DualGCN.csv'
utils.write_csv_header(csvFileName)

G, adj, adj_sym, adj_skew_sym, signals, labels, weighted_labels = utils.setup(graph_name=graph_name, \
                                                                requested_signals=requested_signals, \
                                                                weights_normalization=weights_normalization, \
                                                                signals_normalization=signals_normalization)
signals = torch.tensor(signals, dtype=torch.float32)
L_sym = utils.compute_scaled_normalized_laplacian(adj_sym, renormalize=renormalize, lambda_max=2.0)
L_sym = utils.cnv_sparse_mat_to_coo_tensor(L_sym)

L_skew_sym = utils.compute_scaled_normalized_laplacian(adj_skew_sym, renormalize=renormalize, lambda_max=2.0)
L_skew_sym = utils.cnv_sparse_mat_to_coo_tensor(L_skew_sym)

size_in = signals.shape[1]
size_out = signals.shape[1]

for round in range(1, rounds+1):
    csvData = [graph_name, str(nx.number_of_edges(G) / nx.number_of_nodes(G)), signals_normalization if signals_normalization != None else 'No', \
       weights_normalization if weights_normalization != None else 'No', 'Yes' if renormalize == True else 'No', \
       str(K), str(number_layers), str(hidden_dimensions), str(learning_rate)]
        
    model = DualGCN(size_in=size_in, size_out=size_out, hidden_dim=hidden_dimensions, nb_layers=number_layers, K=K, enable_bias=False, droprate=None)                              

    # Loss function
    criterion = torch.nn.L1Loss(reduction="none")
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 1000
    startTime = time.time()
    losses = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output1, output2, output = model(signals, signals, L_sym, L_skew_sym)
        loss = criterion(output, signals)
        loss.sum().backward()
        optimizer.step()
        l = loss.sum().item()
        if (len(losses) < 10):
            losses.append(l)
        else:
            avg = sum(losses) / 10
            if abs(l - avg) < 0.01*avg:
                csvData += [epoch+1]
                break
            else:
                losses.pop(0)
                losses.append(l)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {l}')
        if(epoch + 1 == num_epochs):
            csvData += [epoch+1]

    stopTime = time.time()
    
    csvData += [str(losses[9])]
    csvData += [str(stopTime - startTime)]

    G_eval = G
    L_sym_eval = L_sym
    L_skew_sym_eval = L_skew_sym
    signals_eval = signals
    labels_eval = labels
    weighted_labels_eval = weighted_labels

    model.eval()
    with torch.no_grad():
        outEval1, outEval2, outEval = model(signals_eval, signals_eval, L_sym_eval, L_skew_sym_eval)
        err = criterion(outEval, signals_eval)
        data = utils.evaluate(labels_eval, weighted_labels_eval, err)
        csvData += data

    with open(csvFileName, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csvData)