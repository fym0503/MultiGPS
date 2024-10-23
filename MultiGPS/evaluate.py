from os import replace
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tqdm import tqdm
import torch.nn as nn
from sklearn.model_selection import train_test_split
from .benchmark_task import clustering, classification, reconstruction, reconstruction_RMSE
import pdb
import scanpy as sc
import numpy as np

def default_checkout(epoch, var, models, args, tasks, train_loss, val_loss):
    device = args.device
    array = models['rep'].mu.detach().cpu().numpy()
    index = np.argsort(array)[::-1]
    var_index = [var[i] for i in index]
    
    saving_name = args.tasks.replace(",","-")+ '-seed' + str(args.seed) + '-' + args.task_name 
    with open(args.log_dir + saving_name + '.txt','a') as file:
        print(f'epoch={epoch}',file=file)
        print(np.array(index),file=file)
        for t_index in range(len(tasks[1:])):
            t = tasks[t_index+1]
            print(f'Task {t} loss in train:{train_loss[t_index]}',file=file)
            if args.val:
                print(f'Task {t} loss in val:{val_loss[t_index]}',file=file)
    with open(args.saving_dir + saving_name + '-epoch:' + str(epoch)+ '.txt', 'w') as file:         
        print(var_index, file=file)
    
    if args.save:
        for model_key in models.keys():
            torch.save(args.saving_dir + saving_name + '-epoch' + str(epoch)+ '.pt')
            