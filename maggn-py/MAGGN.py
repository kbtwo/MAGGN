from torch_geometric.nn import GCNConv, GCN2Conv,GraphConv, GATConv,GATv2Conv, global_max_pool as gmp, global_add_pool as gap, \
    global_mean_pool as gep, global_sort_pool
from torch_geometric.utils import dropout_adj
import torch.nn as nn
import torch
from torch_geometric.nn import EdgeConv,RGATConv,GINConv
from GRU import GRU_gate

class Model(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=53, output_dim=128, 
        hidden_channels=64,dropout=0.5):
        super(Model, self).__init__()
        
        self.gru_gate1 = GRU_gate(num_features_pro*4)
        self.transform1 = nn.Linear(num_features_pro, num_features_pro*4)
        
        self.n_output = n_output
        self.pro_conv1 = GATConv(num_features_pro, num_features_pro, heads=4, dropout=dropout)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 256)
        self.pro_fc_g2 = torch.nn.Linear(256, output_dim) 
        
        self.pro_conv2 = GINConv(nn=nn.Sequential(
            nn.Linear(num_features_pro, num_features_pro*2), 
            nn.ReLU(), 
            nn.Linear(num_features_pro*2, num_features_pro*2)
        ))
        self.pro_fc_g3 = torch.nn.Linear(num_features_pro * 2, 256)
        self.pro_fc_g4 = torch.nn.Linear(256, output_dim)
        

        self.pro_conv3 = EdgeConv(nn=nn.Sequential(
            nn.Linear(num_features_pro*2, num_features_pro*2),  
            nn.ReLU(),
            nn.Linear(num_features_pro*2, num_features_pro*2)  
        ))
        self.pro_fc_g5 = torch.nn.Linear(num_features_pro*2, 256)
        self.pro_fc_g6 = torch.nn.Linear(256, output_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(output_dim*3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, self.n_output)
        self.sigmoid = nn.Sigmoid()


    def forward(self, data_pro):

        target_x = data_pro.x 
        target_edge_index = data_pro.edge_index 
        target_weight = data_pro.edge_weight 
        target_batch = data_pro.batch
        
        xt = self.pro_conv1(target_x, target_edge_index)
        xt = self.relu(xt)
        xt = self.dropout(xt)
        xt = gep(xt, target_batch) 
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.dropout(xt)
        xt = self.pro_fc_g2(xt)
        xt = self.relu(xt)
        xt = self.dropout(xt) 
        
        xt2 = self.pro_conv2(target_x, target_edge_index)
        xt2 = self.relu(xt2)
        xt2 = self.dropout(xt2) 
        xt2 = gep(xt2, target_batch)
        xt2 = self.relu(self.pro_fc_g3(xt2))
        xt2 = self.dropout(xt2)
        xt2 = self.pro_fc_g4(xt2)
        xt2 = self.relu(xt2)
        xt2 = self.dropout(xt2) 
        
        xt3 = self.pro_conv3(target_x, target_edge_index)
        xt3 = self.relu(xt3)
        xt3 = self.dropout(xt3)
        xt3 = gep(xt3, target_batch) 
        xt3 = self.relu(self.pro_fc_g5(xt3))
        xt3 = self.dropout(xt3)
        xt3 = self.pro_fc_g6(xt3)
        xt3 = self.relu(xt3)
        xt3 = self.dropout(xt3) 
        
        xt4 = torch.cat((xt,xt2,xt3), dim=1)
        
        xt4 = self.fc1(xt4)
        xt4 = self.relu(xt4)
        xt4 = self.dropout(xt4)
        xt4 = self.fc2(xt4)
        xt4 = self.relu(xt4)
        xt4 = self.dropout(xt4)
        out = self.sigmoid(self.out(xt4))

        return out