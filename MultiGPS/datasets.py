import torch
import anndata as ad
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from MultiGPS.stgutils import FastTensorDataLoader
import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix
import random

def get_dataset(args):
    device = torch.device(args.device)
    
    datas = []
    tasks = args.tasks.split(",")
    new_tasks = []
    adata = ad.read_h5ad(args.adata_file)
    adata.var_names_make_unique()
    var = adata.var.index
    
    if args.layer == 'raw':
        raw_x = adata.X
    else:
        raw_x = adata.layers[args.layer].copy()
    if isinstance(raw_x, csr_matrix):
        raw_x = raw_x.toarray()
        
    standard_scaler = StandardScaler()
    raw_x = standard_scaler.fit_transform(raw_x)  
    
    if args.hvg:
        if args.layer == 'raw':
            sc.pp.highly_variable_genes(adata, layer=args.layer, n_top_genes=10000)
        
        hvg_indices = [i for i in range(raw_x.shape[1]) if adata.var['highly_variable'].values[i]]
        x = raw_x[:,hvg_indices]
        var = [var[i] for i in hvg_indices]
    else:
        x = raw_x
            
    datas.append(x)
    new_tasks.append('input')

    if 'recon' in tasks:
        datas.append(x)
        new_tasks.append('recon')
    
    if 'cls' in tasks:
        cls_data = adata.obs['cell_type'].astype('category').cat.codes.values
        datas.append(cls_data)
        new_tasks.append('cls')
        
    if "coordination" in tasks or "standard_coordination" in tasks:
        coo = adata.obsm['spatial']
        datas.append(coo)
        new_tasks.append('coo')
    
    return var, new_tasks, datas

def split_dataset(tasks, datas, args, ratio=0.2):
    device = torch.device(args.device)
    if args.val:
        dataset_size = len(datas[0])
        val_size = int(dataset_size * ratio)
        train_size = dataset_size - val_size

        val_indices = np.random.choice(np.arange(dataset_size), size=int(val_size), replace=False)
        train_indices = np.setdiff1d(np.arange(dataset_size), val_indices)
        
        train_data_list = [data[list(train_indices)] for data in datas]
        val_data_list = [data[list(val_indices)] for data in datas]
        
        train_dataloader = FastTensorDataLoader(*[torch.from_numpy(data).float().to(device) if np.issubdtype(data.dtype, np.floating)
                else torch.from_numpy(data).long().to(device) for data in train_data_list],
                tensor_names=tasks, batch_size=args.batch_size, shuffle=True)
        
        val_dataloader = FastTensorDataLoader(*[torch.from_numpy(data).float().to(device) if np.issubdtype(data.dtype, np.floating)
                else torch.from_numpy(data).long().to(device) for data in val_data_list],
                tensor_names=tasks, batch_size=args.batch_size, shuffle=True)

    else:
        train_dataloader = FastTensorDataLoader(*[torch.from_numpy(data).float().to(device) if np.issubdtype(data.dtype, np.floating)
                else torch.from_numpy(data).long().to(device) for data in datas],
                tensor_names=tasks, batch_size=args.batch_size, shuffle=True)
        val_dataloader = None
    
    return train_dataloader, val_dataloader