import traceback
import pandas as pd
import numpy as np
import networkx as nx
import sys
import os
import pandas as pd
import random
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch


df1 = pd.read_csv('dataset/Train_set/ProLaTherm_train.csv')
sequence = df1['sequence']
label = df1['label']

df2 = pd.read_csv('dataset/Test_set/ProLaTherm_test1.csv')
sequence2 = df2['sequence']
label2 = df2['label']

df3 = pd.read_csv('dataset/Test_set/ProLaTherm_test2.csv')
sequence3 = df3['sequence']
label3 = df3['label']

X = sequence.values
y = label.values
X2 = sequence2.values
y2 = label2.values
X3 = sequence3.values
y3 = label3.values

def create_custom_id1(index):
    return f"fulldata_{index}"

def create_custom_id2(index):
    return f"seq_test1_{index}"

def create_custom_id3(index):
    return f"test_set2_{index}"

def convert_to_dict1(data):
    protein_dict = {}
    for i, sequence in enumerate(data):
        unique_id = create_custom_id1(i)
        protein_dict[unique_id] = sequence
    return protein_dict

def convert_to_dict2(data):
    protein_dict = {}
    for i, sequence in enumerate(data):
        unique_id = create_custom_id2(i)
        protein_dict[unique_id] = sequence
    return protein_dict

def convert_to_dict3(data):
    protein_dict = {}
    for i, sequence in enumerate(data):
        unique_id = create_custom_id3(i)
        protein_dict[unique_id] = sequence
    return protein_dict

protein_array1 = X
proteins_train = convert_to_dict1(protein_array1)

protein_array2 = X2
proteins_test = convert_to_dict2(protein_array2)

protein_array3 = X3
proteins_test2 = convert_to_dict3(protein_array3)

def dic_normalize(dic):  
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 
                 'S', 'T', 'V', 'W', 'Y','X']
pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 
                    'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 
                    'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 
                    'Y': 163.18}
res_weight_table['X'] = np.average([res_weight_table[k] for k in res_weight_table.keys()])

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 
                 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 
                 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
res_pka_table['X'] = np.average([res_pka_table[k] for k in res_pka_table.keys()])

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 
                 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 
                 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
res_pkb_table['X'] = np.average([res_pkb_table[k] for k in res_pkb_table.keys()])

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 
                 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
res_pkx_table['X'] = np.average([res_pkx_table[k] for k in res_pkx_table.keys()])

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 
                'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 
                'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}
res_pl_table['X'] = np.average([res_pl_table[k] for k in res_pl_table.keys()])

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 
                             'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 
                             'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}
res_hydrophobic_ph2_table['X'] = np.average([res_hydrophobic_ph2_table[k] for k in res_hydrophobic_ph2_table.keys()])

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 
                             'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 
                             'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}
res_hydrophobic_ph7_table['X'] = np.average([res_hydrophobic_ph7_table[k] for k in res_hydrophobic_ph7_table.keys()])

# nomarlize the residue feature
res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


# one ont encoding
def one_hot_encoding(x, allowable_set):
    if x not in allowable_set: 
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


# one ont encoding with unknown symbol
def one_hot_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def seq_feature(seq): 
    residue_feature = []
    for residue in seq: 
        # replace some rare residue with 'X'
        if residue not in pro_res_table: 
            residue = 'X'
        res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 
                         1 if residue in pro_res_aromatic_table else 0,
                         1 if residue in pro_res_polar_neutral_table else 0,
                         1 if residue in pro_res_acidic_charged_table else 0,
                         1 if residue in pro_res_basic_charged_table else 0]
        res_property2 = [res_weight_table[residue], res_pka_table[residue], 
                         res_pkb_table[residue],res_pkx_table[residue],
                         res_pl_table[residue], res_hydrophobic_ph2_table[residue], 
                         res_hydrophobic_ph7_table[residue]]
        residue_feature.append(res_property1 + res_property2) 
        

    pro_hot = np.zeros((len(seq), len(pro_res_table))) 
    pro_property = np.zeros((len(seq), 12)) 
    for i in range(len(seq)):
        pro_hot[i,] = one_hot_encoding_unk(seq[i], pro_res_table)
        pro_property[i,] = residue_feature[i]
        
    seq_feature = np.concatenate((pro_hot, pro_property), axis=1)
    return seq_feature


def sequence_to_graph(target_key, target_sequence, distance_dir,distance_dir_pssm):
    target_edge_index = []
    target_edge_distance = []
    target_size = len(target_sequence)
    contact_map_file = os.path.join(distance_dir, target_key + '.npy')
    pssm_data_file = os.path.join(distance_dir_pssm,target_key+'.npy')
    distance_map = np.load(contact_map_file)
    pssm_data = np.load(pssm_data_file)   
    normalized_pssm = 1 / (1 + np.exp(-pssm_data))
    # the neighbor residue should have a edge
    # add self loop
    for i in range(target_size):
        distance_map[i, i] = 1
        if i + 1 < target_size:
            distance_map[i, i + 1] = 1
    # print(distance_map)
    index_row, index_col = np.where(distance_map >= 0.5)  # for threshold
    # print(len(index_row))
    # print(len(index_col))
    # print(len(index_row_))
    # print(len(index_col_))
    # print(distance_map.shape)
    # print((len(index_row) * 1.0) / (distance_map.shape[0] * distance_map.shape[1]))
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])  
        target_edge_distance.append(distance_map[i, j])  
    #target_feature = seq_feature(target_sequence)
#     print(seq_feature(target_sequence).shape)
#     print(pssm_data.shape)
    target_feature = np.concatenate((seq_feature(target_sequence), normalized_pssm), axis=1)

    return target_size, target_feature, target_edge_index, target_edge_distance


# initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, root='/', y=None, transform=None,
                 pre_transform=None, target_key=None, target_graph=None):
        super(DTADataset, self).__init__(root, transform, pre_transform)

        self.target = target_key
        self.y = y
        self.target_graph = target_graph
        self.process(target_key, y, target_graph)

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        # return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']
        pass

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        #         if not os.path.exists(self.processed_dir):
        #             os.makedirs(self.processed_dir)
        pass

    def process(self, target_key, y, target_graph):
        data_list_pro = []
        data_len = len(target_key)  # 序列个数
        print('loading tensors ...')
        for i in tqdm(range(data_len)):
            tar_key = target_key[i]
            labels = y[i]
            # print(labels,type(labels))

            target_size, target_features, target_edge_index, target_edge_weight = target_graph[tar_key]

            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData_pro = DATA.Data(x=torch.Tensor(target_features),
                                    edge_index=torch.LongTensor(target_edge_index).transpose(1, 0),
                                    edge_weight=torch.FloatTensor(target_edge_weight),
                                    y=torch.FloatTensor([labels]))
            GCNData_pro.__setitem__('target_size', torch.LongTensor([target_size]))
            # print(GCNData_pro.x.size(), GCNData_pro.edge_index.size(), GCNData_pro.y.size())
            # print(GCNData_pro.edge_index)
            data_list_pro.append(GCNData_pro)

        if self.pre_filter is not None:
            data_list_pro = [data for data in data_list_pro if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list_pro = [self.pre_transform(data) for data in data_list_pro]
        self.data_pro = data_list_pro

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # return GNNData_pro
        return self.data_pro[idx]


def collate(data_list):
    batchA = Batch.from_data_list([data for data in data_list])
    return batchA


 def create_train_dataset():
    
    # load protein feature and predicted distance map
    pro_distance_dir = 'contact_map/Train'
    pssm_dir = 'pssm/Train'
    
    prots = []  # sequences
    prot_keys = []  # protein id (or name)
    labels = y

    # seqs
    for t in proteins_train.keys():
        prots.append(proteins_train[t])
        prot_keys.append(t)
    
    train_prot_keys = prot_keys 
    train_Y = labels 

    # create target graph
    # print('target_key', len(target_key), len(set(target_key)))
    target_graph = {}
    for i in tqdm(range(len(prot_keys))):
        key = prot_keys[i] 
        g_t = sequence_to_graph(key, proteins_train[key], pro_distance_dir,pssm_dir)
        target_graph[key] = g_t

    train_prot_keys, train_Y = np.asarray(train_prot_keys), np.asarray(train_Y)
    train_dataset = DTADataset(root='/', target_key=train_prot_keys, y=train_Y, target_graph=target_graph)

    return train_dataset 



def create_test_dataset():
    
    # load protein feature and predicted distance map
    pro_distance_dir = 'contact_map/Test 1'
    pssm_dir = 'pssm/Test 1'
    
    prots = []  # sequences
    prot_keys = []  # protein id (or name)
    labels = y2

    # seqs
    for t in proteins_test.keys():
        prots.append(proteins_test[t])
        prot_keys.append(t)
    
    train_prot_keys = prot_keys
    train_Y = labels

    # create target graph
    # print('target_key', len(target_key), len(set(target_key)))
    target_graph = {}
    for i in tqdm(range(len(prot_keys))):
        key = prot_keys[i]
        g_t = sequence_to_graph(key, proteins_test[key], pro_distance_dir,pssm_dir)
        target_graph[key] = g_t

    train_prot_keys, train_Y = np.asarray(train_prot_keys), np.asarray(train_Y)
    test1_dataset = DTADataset(root='/', target_key=train_prot_keys, y=train_Y, 
                               target_graph=target_graph)

    return test1_dataset



def create_test_dataset2():
    
    # load protein feature and predicted distance map
    pro_distance_dir = 'contact_map/Test 2'
    pssm_dir = 'pssm/Test 2'
    
    prots = []  # sequences
    prot_keys = []  # protein id (or name)
    labels = y3

    # seqs
    for t in proteins_test2.keys():
        prots.append(proteins_test2[t])
        prot_keys.append(t)
    
    train_prot_keys = prot_keys
    train_Y = labels

    # create target graph
    # print('target_key', len(target_key), len(set(target_key)))
    target_graph = {}
    for i in tqdm(range(len(prot_keys))):
        key = prot_keys[i]
        g_t = sequence_to_graph(key, proteins_test2[key], pro_distance_dir,pssm_dir)
        target_graph[key] = g_t

        
    train_prot_keys, train_Y = np.asarray(train_prot_keys), np.asarray(train_Y)
    test2_dataset = DTADataset(root='/', target_key=train_prot_keys, y=train_Y, 
                               target_graph=target_graph)

    return test2_dataset
