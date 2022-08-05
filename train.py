import torch
from torch_geometric.loader import DataLoader
from myqm9_v2 import QM9
from torch import nn
from gat import NN
import numpy as np
import os
from tqdm import tqdm
from calculate_r2 import *

save_dir = 'gat_test'
class GNN:
    def __init__(self):
        self.model = NN()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-16)

    def load(self, name, device='cpu'):
        self.model.load_state_dict(torch.load(name, map_location=torch.device(device)))
    
    def convert(self, y):
        o = torch.tensor([0 for _ in range(6)]).float()
        o[y[0]] = 1
        return o

    def sample(self, dataset, batch_size):
        n = torch.randint(0, len(dataset), (batch_size,))
        batch = []
        for i in n:
            batch.append(dataset[i])
        return batch

    def learn(self, dataset, batch_size, epoch = 0, target = 0, device='cpu'):
        self.model.train()
        train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        train_loader = tqdm(train_loader, total=int(len(dataset)/batch_size)+1)
        losses = []
        accuracy = []
        for data in train_loader:
            data = data.to(device)
            self.optimizer.zero_grad()
            y = data.y[: , target]
            y = y.reshape((len(y),1))
            out = self.model(data)
            loss = self.loss_fn(out, y)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            acc = getr2(y.reshape(len(y)), out.reshape(len(out)))
            accuracy.append(acc)
            train_loader.set_description('loss: {:.5f}, r2: {:.5f}, epochs: {}'.format(loss.item(), acc, epoch))
        return losses, accuracy

    def save(self, name='molecule_model'):
        torch.save(self.model.state_dict(), name)

    def decay_lr(self, factor=10):
        for g in self.optimizer.param_groups:
            g['lr'] /= factor

def train(load_old=False, target=0, device='cpu', num_epochs=10):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    path = save_dir + '/' + 'nn_{}'.format(target)
    loss_dir = save_dir + '/' + 'loss_{}.npy'.format(target)
    acc_dir = save_dir + '/' + 'acc_{}.npy'.format(target)
    tacc_dir = save_dir + '/' + 'tacc_{}.npy'.format(target)

    print('loading data ...')
    dataset = QM9('qm9')
    print('data loaded')

    # split = 0.05

    # test_set = dataset[len(dataset) - int(len(dataset)*split):]
    # # dataset = dataset[:len(dataset) - int(len(dataset)*split)]

    point = 10000

    test_set = dataset[len(dataset) - point:]
    dataset = dataset[:len(dataset) - point]

    model = GNN()
    model.model.to(device)

    if load_old:
        model.load(path)
        losses = np.load(loss_dir, allow_pickle=True)
        acc = np.load(acc_dir, allow_pickle=True)
        tacc = np.load(tacc_dir, allow_pickle=True)
    else:
        losses = []
        acc = []
        tacc = []
    for e in range(num_epochs):

        l, a = model.learn(dataset=dataset, batch_size=96, target=target, epoch=e, device=device)
        losses += l
        acc += a
        model.save(path)
        np.save(loss_dir, losses)
        np.save(acc_dir, acc)

        val_acc = calculate(test_set, model.model, target, device=device)
        tacc.append(val_acc)
        print('validation r2: {:.5f}'.format(val_acc))
        np.save(tacc_dir, tacc)

        if (e + 1) % 100 == 0:
            model.decay_lr()
            print('weight decayed')

for i in range(8, 9):
    train(load_old=False, target=i, device='cpu', num_epochs=30)

    
