import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np

__all__ = [
    'LinearLayer', 'MLPLayer', 'FeatureSelector',
]


def get_batcnnorm(bn, nr_features=None, nr_dims=1):
    if isinstance(bn, nn.Module):
        return bn

    assert 1 <= nr_dims <= 3

    if bn in (True, 'async'):
        clz_name = 'BatchNorm{}d'.format(nr_dims)
        return getattr(nn, clz_name)(nr_features)
    else:
        raise ValueError('Unknown type of batch normalization: {}.'.format(bn))


def get_dropout(dropout, nr_dims=1):
    if isinstance(dropout, nn.Module):
        return dropout

    if dropout is True:
        dropout = 0.5
    if nr_dims == 1:
        return nn.Dropout(dropout, True)
    else:
        clz_name = 'Dropout{}d'.format(nr_dims)
        return getattr(nn, clz_name)(dropout)


def get_activation(act):
    if isinstance(act, nn.Module):
        return act

    assert type(act) is str, 'Unknown type of activation: {}.'.format(act)
    act_lower = act.lower()
    if act_lower == 'relu':
        return nn.ReLU(True)
    elif act_lower == 'selu':
        return nn.SELU(True)
    elif act_lower == 'sigmoid':
        return nn.Sigmoid()
    elif act_lower == 'tanh':
        return nn.Tanh()
    else:
        try:
            return getattr(nn, act)
        except AttributeError:
            raise ValueError('Unknown activation function: {}.'.format(act))


class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma, device):
        super(FeatureSelector, self).__init__()
        self.mu = torch.nn.Parameter(0.01*torch.randn(input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size()) 
        self.sigma = sigma
        self.device = device
    
    def forward(self, prev_x):
        z = self.mu + self.sigma*self.noise.normal_()*self.training 
        stochastic_gate = self.hard_sigmoid(z).to(self.device)
        new_x = prev_x * stochastic_gate
        return new_x
    
    def hard_sigmoid(self, x):
        return torch.clamp(x+0.5, 0.0, 1.0)

    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / math.sqrt(2))) 

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self


class GatingLayer(nn.Module):
    '''To implement L1-based gating layer (so that we can compare L1 with L0(STG) in a fair way)
    '''
    def __init__(self, input_dim, device):
        super(GatingLayer, self).__init__()
        self.mu = torch.nn.Parameter(0.01*torch.randn(input_dim, ), requires_grad=True)
        self.device = device
    
    def forward(self, prev_x):
        new_x = prev_x * self.mu 
        return new_x
    
    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return torch.sum(torch.abs(x))


class LinearLayer(nn.Sequential):
    def __init__(self, in_features, out_features, batch_norm=None, dropout=None, bias=None, activation=None):
        if bias is None:
            bias = (batch_norm is None)

        modules = [nn.Linear(in_features, out_features, bias=bias)]
        if batch_norm is not None and batch_norm is not False:
            modules.append(get_batcnnorm(batch_norm, out_features, 1))
        if dropout is not None and dropout is not False:
            modules.append(get_dropout(dropout, 1))
        if activation is not None and activation is not False:
            modules.append(get_activation(activation))
        super().__init__(*modules)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, batch_norm=None, dropout=None, activation='relu', flatten=True):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = []
        elif type(hidden_dims) is int:
            hidden_dims = [hidden_dims]

        dims = [input_dim]
        dims.extend(hidden_dims)
        dims.append(output_dim)
        modules = []

        nr_hiddens = len(hidden_dims)
        for i in range(nr_hiddens):
            layer = LinearLayer(dims[i], dims[i+1], batch_norm=batch_norm, dropout=dropout, activation=activation)
            modules.append(layer)
            modules.append(torch.nn.Dropout(0.2))
        layer = nn.Linear(dims[-2], dims[-1], bias=True)
        modules.append(layer)
        self.mlp = nn.Sequential(*modules)
        self.flatten = flatten

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, input, mask):
        if self.flatten:
            input = input.view(input.size(0), -1)
        return self.mlp(input), mask


class L1GateRegressionModel(MLPModel):
    def __init__(self, input_dim, output_dim, hidden_dims, device, batch_norm=None, dropout=None, activation='relu',
                 sigma=1.0, lam=0.1):
        super().__init__(input_dim, output_dim, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.GateingLayer = GatingLayer(input_dim, device)
        self.reg = self.GateingLayer.regularizer
        self.mu = self.GateingLayer.mu
        self.loss = nn.MSELoss()
        self.lam = lam

    def forward(self, feed_dict):
        x = self.GateingLayer(self._get_input(feed_dict))
        pred = super().forward(x)
        if self.training:
            loss = self.loss(pred, self._get_label(feed_dict))
            reg = torch.mean(self.reg(self.mu))
            total_loss = loss + self.lam * reg
            return total_loss, dict(), dict()
        else:
            return self._compose_output(pred)
            
            
class STGCommonModel(MLPModel):
    def __init__(self, input_dim, output_dim, hidden_dims, device, batch_norm=None, dropout=None, activation='relu',
                 sigma=1.0, lam=0.1):
        super().__init__(input_dim, output_dim, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        super().to(device)
        self.FeatureSelector = FeatureSelector(input_dim, sigma, device)
        self.softmax = nn.Softmax()
        self.loss = nn.MSELoss()
        self.reg = self.FeatureSelector.regularizer
        self.lam = lam 
        self.mu = self.FeatureSelector.mu
        self.sigma = self.FeatureSelector.sigma
        
    def forward(self, x, masks):
        output = self.FeatureSelector(x)
        output, masks = super().forward(output.float(), masks)
        return output, masks
        
    def get_gates(self, mode):
        if mode == 'raw':
            return self.mu.detach().cpu().numpy()
        elif mode == 'prob':
            return np.minimum(1.0, np.maximum(0.0, self.mu.detach().cpu().numpy() + 0.5)) 
        else:
            raise NotImplementedError()
        
    
class MLPclassify(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, batch_norm=None, dropout=None, activation='relu', flatten=True):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = []
        elif type(hidden_dims) is int:
            hidden_dims = [hidden_dims]
        
        dims = [input_dim]
        dims.extend(hidden_dims)
        dims.append(output_dim)
        modules = []

        nr_hiddens = len(hidden_dims)
        for i in range(nr_hiddens):
            layer = LinearLayer(dims[i], dims[i+1], batch_norm=batch_norm, dropout=dropout, activation=activation)
            modules.append(layer)
        layer = nn.Linear(dims[-2], dims[-1], bias=True)
        modules.append(layer)
        self.mlp = nn.Sequential(*modules)
        self.flatten = flatten
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, input, masks):
        if self.flatten:
            input = input.view(input.size(0), -1)
        output =  self.mlp(input)
        return F.log_softmax(output, dim=1), masks

