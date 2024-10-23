import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import explained_variance_score, precision_score, recall_score, f1_score

        
class SimpleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def calculate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_pred = []
    all_label = []
    
    with torch.no_grad():
        for k, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            images = images.float()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            all_label.append(labels)
            all_pred.append(predicted)
            correct += (predicted == labels).sum().item()
    all_pred = torch.concat(all_pred).cpu().numpy()
    all_label = torch.concat(all_label).cpu().numpy()
    
    accuracy = 100 * correct / total
    return accuracy

def RMSE_loss(model, data_loader, device):
    lossf = torch.nn.MSELoss()
    model.eval()
    losses=[]

    with torch.no_grad():
        for k, (data, label) in enumerate(data_loader):
            data = data.float()
            label = label.float()
            data, label = data.to(device), label.to(device)
            outputs = model(data)
            losses.append(lossf(outputs, label).detach().cpu().numpy())
            
    return np.mean(losses)

def explained_variance(model, data_loader, device):
    lossf = torch.nn.MSELoss()
    model.eval()
    outputs = []
    labels = []

    with torch.no_grad():
        for k, (data, label) in enumerate(data_loader):
            data = data.float()
            label = label.float()
            data, label = data.to(device), label.to(device)
            output = model(data)
            outputs.append(output.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())

    return explained_variance_score(np.concatenate(labels), np.concatenate(outputs))


class MLP_cls(nn.Module):
    '''
    Multilayer perceptron (MLP) model for classification
    '''

    def __init__(self,
                 input_size,
                 output_size,
                 hidden,
                 activation=nn.ReLU(),
                 device=torch.device('cpu')
                 ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        fc_layers = [nn.Linear(d_in, d_out) for d_in, d_out in
                     zip([input_size] + hidden, hidden + [output_size])]

        self.device = device
        self.fc = nn.ModuleList(fc_layers).to(device)

        self.activation = activation.to(device)

    def forward(self, x):
        x = x.to(self.device)
        for fc in self.fc[:-1]:
            x = fc(x)
            x = self.activation(x)

        x = self.fc[-1](x)
        return x

    def fit(self, train_dataset, val_dataset, mbsize = 16, max_nepochs = 10,
            lr = 1e-3, bar = False, early_stop = False):
        optimizer = optim.Adam(self.parameters(), lr=lr)

        has_init = hasattr(train_dataset, 'init_worker')
        if has_init:
            train_init = train_dataset.init_worker
            val_init = val_dataset.init_worker
        else:
            train_init = None
            val_init = None
        train_loader = DataLoader(
            train_dataset, batch_size=mbsize, shuffle=True, drop_last=True,
            worker_init_fn=train_init)
        val_loader = DataLoader(
            val_dataset, batch_size=mbsize, worker_init_fn=val_init)

        if bar:
            tqdm_bar = tqdm(
                total=max_nepochs, desc='Training epochs', leave=True)

        for epoch in range(max_nepochs):
            for k,(x, y) in enumerate(train_loader):
                x = x.float()
                y = y.type(torch.LongTensor)
                
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.forward(x)
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(pred, y)

                loss.backward()
                optimizer.step()
                self.zero_grad()

            # Update bar.
            if bar:
                tqdm_bar.update(1)

    def predict(self, x):
        with torch.no_grad():
            x = x.to(self.device)
            logits = self.forward(x)
            predicted_labels = torch.argmax(logits, dim=1)
        return predicted_labels

class MLP_rebuild(nn.Module):
    '''
    Multilayer perceptron (MLP) model.
    '''

    def __init__(self,
                 input_size,
                 output_size,
                 hidden,
                 activation=nn.ReLU(),
                 device=torch.device('cpu')
                 ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        fc_layers = [nn.Linear(d_in, d_out) for d_in, d_out in
                     zip([input_size] + hidden, hidden + [output_size])]

        self.device = device
        self.fc = nn.ModuleList(fc_layers).to(device)

        self.activation = activation.to(device)

    def forward(self, x):
        x = x.to(self.device)
        for fc in self.fc[:-1]:
            x = fc(x)
            x = self.activation(x)

        x = self.fc[-1](x)
        return x

    def fit(self, train_dataset, val_dataset, mbsize=16, max_nepochs=10,
            lr=1e-3, bar=False, show_loss=False):

        optimizer = optim.Adam(self.parameters(), lr=lr)
        loss = []
        has_init = hasattr(train_dataset, 'init_worker')
        if has_init:
            train_init = train_dataset.init_worker
            val_init = val_dataset.init_worker
        else:
            train_init = None
            val_init = None
        train_loader = DataLoader(
            train_dataset, batch_size=mbsize, shuffle=True, drop_last=True,
            worker_init_fn=train_init)
        val_loader = DataLoader(
            val_dataset, batch_size=mbsize, worker_init_fn=val_init)

        if bar:
            tqdm_bar = tqdm(
                total=max_nepochs, desc='Training epochs', leave=True)
            
        for epoch in range(max_nepochs):
            loss_epoch=[]
            for k,(x, y) in enumerate(train_loader):
                x = x.float()
                y = y.float()
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.forward(x)

                loss_fn = nn.MSELoss()
                loss = loss_fn(pred, y)
                loss_epoch.append(loss.detach().cpu().numpy())

                loss.backward()
                optimizer.step()
                self.zero_grad()

            if bar:
                tqdm_bar.update(1)
            if show_loss:
                print(f"Epoch {epoch} Loss: {np.mean(loss_epoch)}")

    def predict(self, x):
        with torch.no_grad():
            x = x.to(self.device)
            pred = self.forward(x)
            
        return pred
