import pdb
import numpy as np
import torch

import torch
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn as nn
import os
import MultiGPS.losses as losses
from MultiGPS.datasets import get_dataset, split_dataset
import MultiGPS.evaluate as checkout
import MultiGPS.model_selector as model_selector
from MultiGPS.min_norm_solvers import MinNormSolver, gradient_normalizers
import argparse
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import torch
import random

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def MultiGPS(args):
    var, tasks, datas = get_dataset(args)
    os.makedirs(args.log_dir,exist_ok=True)
    os.makedirs(args.saving_dir,exist_ok=True)
    train_dataloader, val_dataloader = split_dataset(tasks, datas, args)
   
    loss_fn = losses.get_loss(args)
    model = model_selector.get_model(tasks, datas, args)
    
    model_params = []
    for m in model:
        model_params += model[m].parameters()
    optimizer = model_selector.get_optimizer(model_params, args)

    recording_epoch = args.record
      
    # Start Training
    n_iter = 0
    for epoch in tqdm(range(args.epoch)):
        print('Epoch {} Started'.format(epoch))
        train_losses = []
        val_losses = []
        
        epoch_losses = {}
        for t in tasks[1:]:
            epoch_losses[t]=[]
            
        for m in model:
            model[m].train()
    
        for feed_dict in train_dataloader:
            n_iter += 1
            input_data = feed_dict['input']

            labels = {}
            for t in tasks[1:]:
                labels[t] = feed_dict[t]
            
            # Scaling the loss functions based on the algorithm choice
            loss_data = {}
            grads = {}
            scale = {}
            mask = None
            masks = {}
    
            optimizer.zero_grad()
            # First compute representations (z)
            input_volatile = Variable(input_data.data, volatile=True)
            # pdb.set_trace()
            rep, mask = model['rep'](input_volatile, mask)
            rep = rep.float()
            
            # As an approximate solution we only need gradients for input
            if isinstance(rep, list):
                rep = rep[0]
                rep_variable = [Variable(rep.data.clone(), requires_grad=True)]
                list_rep = True
            else:
                rep_variable = Variable(rep.data.clone(), requires_grad=True)
                list_rep = False

            # Compute gradients of each loss function wrt z
            for t in tasks[1:]:
                optimizer.zero_grad()
                out_t, masks[t] = model[t](rep_variable, None)
                reg = torch.mean(model['rep'].reg((model['rep'].mu + 0.5)/model['rep'].sigma)) 
                
                loss = loss_fn[t](out_t, labels[t]) + model['rep'].lam * reg
                loss_data[t] = loss.item()
                
                loss.backward()
                grads[t] = []
                if list_rep:
                    grads[t].append(Variable(rep_variable[0].grad.data.clone(), requires_grad=False))
                    rep_variable[0].grad.data.zero_()
                else:
                    grads[t].append(Variable(rep_variable.grad.data.clone(), requires_grad=False))
                    rep_variable.grad.data.zero_()
            
            gn = gradient_normalizers(grads, loss_data, 'loss+')
            for t in tasks[1:]:
                for gr_i in range(len(grads[t])):
                    grads[t][gr_i] = grads[t][gr_i] / gn[t]

            # Frank-Wolfe iteration to compute scales.
            if len(tasks)>2:
                sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks[1:]])
                for i, t in enumerate(tasks[1:]):
                    scale[t] = float(sol[i])
            else:
                for i, t in enumerate(tasks[1:]):
                    scale[t] = 1
            
            # Scaled back-propagation
            optimizer.zero_grad()
            rep, _ = model['rep'](input_data, mask)
            for i, t in enumerate(tasks[1:]):
                out_t, _ = model[t](rep, masks[t])
                reg = torch.mean(model['rep'].reg((model['rep'].mu + 0.5)/model['rep'].sigma)) 
                loss_t = loss_fn[t](out_t, labels[t]) + model['rep'].lam * reg
                
                epoch_losses[t].append(loss_fn[t](out_t, labels[t]).detach().cpu().numpy())
                loss_data[t] = loss_t.item()
                if i > 0:
                    loss = loss + scale[t]*loss_t
                else:
                    loss = scale[t]*loss_t
            
            loss.backward()
            optimizer.step()
        
        epoch_val_losses = {}
        for t in tasks[1:]:
            epoch_val_losses[t] = []
        
        if args.val:
            with torch.no_grad():
                for feed_dict in val_dataloader:
                    n_iter += 1
                    # First member is always input
                    input_data = feed_dict['input']

                    labels = {}
                    for t in tasks[1:]:
                        labels[t] = feed_dict[t]
                    rep, mask = model['rep'](input_volatile, _)
                    rep = rep.float()
                    
                    rep, _ = model['rep'](input_data, _)
                    for i, t in enumerate(tasks[1:]):
                        out_t, _ = model[t](rep, None)
                        reg = torch.mean(model['rep'].reg((model['rep'].mu + 0.5)/model['rep'].sigma)) 
                        loss_t = loss_fn[t](out_t, labels[t])
                        
                        epoch_val_losses[t].append(loss_fn[t](out_t, labels[t]).detach().cpu().numpy())
                        loss_data[t] = loss_t.item()
                        if i > 0:
                            loss = loss + scale[t]*loss_t
                        else:
                            loss = scale[t]*loss_t
        
        for t in tasks[1:]:
            print(f'Task {t} Loss in train: {np.mean(epoch_losses[t])}')
            train_losses.append(np.mean(epoch_losses[t]))
            if args.val:
                print(f'Task {t} Loss in val: {np.mean(epoch_val_losses[t])}')
                val_losses.append(np.mean(epoch_val_losses[t]))
            
        
        if epoch % recording_epoch == recording_epoch-1:
            checkout.default_checkout(epoch, var, model, args, tasks, train_losses, val_losses)


if __name__ == '__main__':
    print("MultiGPS start!")
    parser = argparse.ArgumentParser(description='MultiGPS')
    parser.add_argument('--adata_file',type=str,required=True,help='the Imaging-Profiling h5ad Dataset')
    parser.add_argument('--saving_dir',type=str,required=True,help='Outputing File Directory')
    parser.add_argument('--log_dir',type=str,required=True,help='Outputing Log File Directory')
    parser.add_argument('--tasks',type=str,required=True,help='Tasks (including recon, cell type...)')

    parser.add_argument('--task_name',type=str,required=True,help='Experiment name')
    parser.add_argument('--hvg',action='store_true',help='Only choosing among HVGs')

    parser.add_argument('--layer',type=str, default='raw',help="Using adata.layer['X'] or adata.X itself")
    parser.add_argument('--learning_rate',type=float,default=0.001)
    parser.add_argument('--batch_size',type=int,default=32)
    parser.add_argument('--panel_size',type=int,default=32)
    parser.add_argument('--lam',type=float,default=0.5,help='lam of STG model')
    parser.add_argument('--sigma',type=float,default=0.5,help='simga of STG model')
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--epoch',type=int,default=5000)
    parser.add_argument('--record',type=int,default=200, help='Record every k epoches')
    parser.add_argument('--optimizer',type=str,default='Adam')
    parser.add_argument('--device',type=str,default='cpu')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--val',action='store_true', help='Whether using validation split')
    
    parser.add_argument('--hurdle',action='store_true',help='Use Hurdle Loss in Reconstruction Task')
    parser.add_argument('--save',action='store_true',help='Whether Saving models')
    parser.add_argument('--evaluate_epoch',type=int,default=10, help='Evaluate every k epoches')
    
    args = parser.parse_args()
    set_random_seed(args.seed)
    
    MultiGPS(args)
    