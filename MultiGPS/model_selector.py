from ast import arg
from .stgmodel import STGCommonModel, MLPclassify, MLPModel
import torch.nn as nn
import numpy as np
import torch
from torch.optim import Adam, SGD, RMSprop
import pdb

def get_model(tasks, datas, args):
    device = torch.device(args.device)
    model = {}
    input_data = datas[0]
    model['rep'] = STGCommonModel(input_data.shape[1], 128, [128], device=device, lam=args.lam, sigma=args.sigma)
    for (task, data) in zip(tasks[1:],datas[1:]):
        if task == 'cls':
            model[task] = MLPclassify(128, len(np.unique(data)), hidden_dims=[128, 128], activation=args.activation).to(device)        
        elif task == 'recon':
            if args.hurdle:
                model[task] = MLPModel(128, data.shape[1]*2, hidden_dims=[128, 128], activation=args.activation).to(device)
            else:
                model[task] = MLPModel(128, data.shape[1], hidden_dims=[128, 128], activation=args.activation).to(device)
        elif task == 'coo':
            model[task] = MLPModel(128, data.shape[1], hidden_dims=[128, 128], activation=args.activation).to(device)
    return model

def get_optimizer(model_params, args):
    if 'RMSprop' == args.optimizer:
        optimizer = RMSprop(model_params, lr=args.learning_rate)
    elif 'Adam' == args.optimizer:
        optimizer = Adam(model_params, lr=args.learning_rate)
    elif 'SGD' == args.optimizer:
        optimizer = SGD(model_params, lr=args.learning_rate, momentum=0.9)
    return optimizer