import torch
import esm
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json, pickle
from collections import OrderedDict
import os
from tqdm import tqdm

def protein_graph_construct(proteins, save_dir):
    # Load ESM-1b model
    # torch.set_grad_enabled(False)
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    target_graph = {}

    count = 0
    key_list=[]
    for key in proteins:
        key_list.append(key)


    for k_i in tqdm(range(len(key_list))):
        key=key_list[k_i]
        # if len(proteins[key]) < 1500:
        #     continue
        data = []
        pro_id = key
        if os.path.exists(save_dir + pro_id + '.npy'):
            continue
        seq = proteins[key]
        if len(seq) <= 1000:
            data.append((pro_id, seq))
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            contact_map = results["contacts"][0].numpy()
            target_graph[pro_id] = contact_map
        else:
            contact_prob_map = np.zeros((len(seq), len(seq)))  # global contact map prediction
            interval = 500
            i = math.ceil(len(seq) / interval)
            
            for s in range(i):
                start = s * interval  # sub seq predict start
                end = min((s + 2) * interval, len(seq))  # sub seq predict end
                sub_seq_len = end - start

                # prediction
                temp_seq = seq[start:end]
                temp_data = []
                temp_data.append((pro_id, temp_seq))
                batch_labels, batch_strs, batch_tokens = batch_converter(temp_data)
                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
                # insert into the global contact map
                row, col = np.where(contact_prob_map[start:end, start:end] != 0)
                row = row + start
                col = col + start
                contact_prob_map[start:end, start:end] = contact_prob_map[start:end, start:end] + results["contacts"][
                    0].numpy()
                contact_prob_map[row, col] = contact_prob_map[row, col] / 2.0
                if end == len(seq):
                    break
            target_graph[pro_id] = contact_prob_map

        np.save(save_dir + pro_id + '.npy', target_graph[pro_id])
        count += 1


# df1 = pd.read_csv('dataset/Train_set/ProLaTherm_train.csv')
# sequence = df1['sequence']
# label = df1['label']
# X = sequence.values
# y = label.values
# def create_custom_id1(index):
#     return f"fulldata_{index}"
# def convert_to_dict1(data):
#     protein_dict = {}
#     for i, sequence in enumerate(data):
#         unique_id = create_custom_id1(i)
#         protein_dict[unique_id] = sequence
#     return protein_dict
# protein_array1 = X
# proteins_train = convert_to_dict1(protein_array1)
# save_dir='/'
#protein_graph_construct(proteins_train, save_dir)