import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Data_process import *
from GRU import GRU_gate
from MAGGN import Model
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score, matthews_corrcoef


TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256

train_data = create_train_dataset()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                               collate_fn=collate)

test_data = create_test_dataset()
test_loader = torch.utils.data.DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=True,
                                               collate_fn=collate)

test_data2 = create_test_dataset2()
test_loader2 = torch.utils.data.DataLoader(test_data2, batch_size=TEST_BATCH_SIZE, shuffle=True,
                                               collate_fn=collate)

if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print("The current device is a GPU, and the GPU device number is:{}".format(device))
else:
    print("The current device is a CPU")


def train(model, device, train_loader, optimizer, epoch, loss_fn, TRAIN_BATCH_SIZE=512):
    model.train()
    LOG_INTERVAL = 10
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * TRAIN_BATCH_SIZE,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))  
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    #print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()



model = Model()
model.to(device)

LR = 0.001
NUM_EPOCHS = 1000
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
model_file_name =  'MAGGN.model'

for epoch in range(NUM_EPOCHS):
    train(model, device, train_loader, optimizer, epoch + 1, loss_fn, TRAIN_BATCH_SIZE)

torch.save(model.state_dict(), model_file_name)
print('Training ends')
print('----------------------------------------------------------')
print('----------------------------------------------------------')
print('----------------------------------------------------------')

# # test1
print('Test set 1')
print('all training done. Testing...')
model_p = Model()
model_p.to(device)
model_p.load_state_dict(torch.load(model_file_name))
test_T, test_P = predicting(model_p, device, test_loader)

test_accuracy = accuracy_score(test_T, np.where(test_P >= 0.5, 1, 0))
test_mcc = matthews_corrcoef(test_T, np.where(test_P >= 0.5, 1, 0))
test_auc = roc_auc_score(test_T, test_P)
test_recall = recall_score(test_T, np.where(test_P >= 0.5, 1, 0))
test_precision = precision_score(test_T, np.where(test_P >= 0.5, 1, 0))
test_f1_score=f1_score(test_T, np.where(test_P >= 0.5, 1, 0))
result_str = 'test result:' + '\n' +'test_accuracy:' + str(test_accuracy) + '\n' +'test_auc:' + str(test_auc) + '\n' + 'test_recall:' + str(
    test_recall) + '\n' + 'test_precision:' + str(test_precision) + '\n'+ 'test_f1_core:' + str(test_f1_score) + '\n' +'test_mcc:' + str(test_mcc) + '\n'
 
print(result_str)

print('----------------------------------------------------------')
print('----------------------------------------------------------')
print('----------------------------------------------------------')

# # test2
print('Test set 2')
print('all training done. Testing...')
model_p = Model()
model_p.to(device)
model_p.load_state_dict(torch.load(model_file_name))
test_T, test_P = predicting(model_p, device, test_loader2)

test_accuracy = accuracy_score(test_T, np.where(test_P >= 0.5, 1, 0))
test_mcc = matthews_corrcoef(test_T, np.where(test_P >= 0.5, 1, 0))
test_auc = roc_auc_score(test_T, test_P)
test_recall = recall_score(test_T, np.where(test_P >= 0.5, 1, 0))
test_precision = precision_score(test_T, np.where(test_P >= 0.5, 1, 0))
test_f1_score=f1_score(test_T, np.where(test_P >= 0.5, 1, 0))
result_str = 'test result:' + '\n' +'test_accuracy:' + str(test_accuracy) + '\n' +'test_auc:' + str(test_auc) + '\n' + 'test_recall:' + str(
    test_recall) + '\n' + 'test_precision:' + str(test_precision) + '\n'+ 'test_f1_core:' + str(test_f1_score) + '\n' +'test_mcc:' + str(test_mcc) + '\n'
 
print(result_str)