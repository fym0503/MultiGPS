import scanpy as sc
import numpy as np
import anndata
from scib_metrics.benchmark import Benchmarker

import sys
from .benchmark_model import SimpleDataset, MLP_rebuild, MLP_cls, RMSE_loss, calculate_accuracy, explained_variance
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def classification(X, label, repeat_times, device, epoch = 10, batch = 1024, lr = 0.001):
    train_result = []
    test_result = []
    
    for j in range(repeat_times):
        train_x, val_x, train_y, val_y = train_test_split(X, label, test_size = 0.2, random_state = j)
        input_dim = train_x.shape[1]
        y_dim = len(np.unique(label))
        train_dataset = SimpleDataset(train_x, train_y)
        val_dataset = SimpleDataset(val_x, val_y)

        MLP = MLP_cls(input_dim, y_dim,[128,128], device = device).to(device)
        MLP.fit(train_dataset, val_dataset, bar=True, max_nepochs = epoch, mbsize = batch, lr = lr)

        train_loader = DataLoader(train_dataset, batch_size = batch, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size = batch, shuffle=True)

        train_criterion = calculate_accuracy(MLP, train_loader, device)
        test_criterion = calculate_accuracy(MLP, val_loader, device)

        train_result.append(train_criterion)
        test_result.append(test_criterion)
    return train_result, test_result

def reconstruction(X, label, repeat_times, device, epoch = 100, batch = 1024, lr = 0.001):
    train_result = []
    test_result = []
    
    for j in range(repeat_times):
        train_x, val_x, train_y, val_y = train_test_split(X, label, test_size = 0.2, random_state = j)
        input_dim = train_x.shape[1]
        y_dim = train_y.shape[1]
        train_dataset = SimpleDataset(train_x, train_y)
        val_dataset = SimpleDataset(val_x, val_y)
        MLP = MLP_rebuild(input_dim, y_dim,[128, 128], device = device).to(device)
        MLP.fit(train_dataset, val_dataset, bar=True, max_nepochs = epoch, mbsize = batch, lr = lr)

        train_loader = DataLoader(train_dataset, batch_size = batch, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size = batch, shuffle=True)
        train_criterion = explained_variance(MLP, train_loader, device)
        test_criterion = explained_variance(MLP, val_loader, device)
        train_result.append(train_criterion)
        test_result.append(test_criterion)
        
    return train_result, test_result

def reconstruction_RMSE(X, label, repeat_times, device, epoch = 20, batch = 1024, lr = 0.001):
    train_result = []
    test_result = []
    
    for j in range(repeat_times):
        train_x, val_x, train_y, val_y = train_test_split(X, label, test_size=0.2,random_state=j)
        input_dim = train_x.shape[1]
        y_dim = train_y.shape[1]
        train_dataset = SimpleDataset(train_x, train_y)
        val_dataset = SimpleDataset(val_x, val_y)
        MLP = MLP_rebuild(input_dim, y_dim,[128, 128],device = device).to(device)
        MLP.fit(train_dataset, val_dataset, bar=True, max_nepochs = epoch, mbsize = batch, lr = lr)

        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True)
        train_criterion = RMSE_loss(MLP, train_loader, device)
        test_criterion = RMSE_loss(MLP, val_loader, device)
        train_result.append(train_criterion)
        test_result.append(test_criterion)
        
    return train_result, test_result

def clustering(X, label, device):
    adata = anndata.AnnData(X)
    adata.obs['label'] = label
    #using this because the benchmark function requires a batch key and the key can't be all the same
    adata.obs['null'] = np.random.randint(10,size=[adata.shape[0],1],dtype=np.int8)
    sc.tl.pca(adata)
    bm = Benchmarker(
        adata,
        label_key="label",
        batch_key="null",
        embedding_obsm_keys=["X_pca"],
        n_jobs=6,
    )
    bm.benchmark()
    df = bm.get_results(min_max_scale=False)
    return df
