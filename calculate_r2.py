import torch
from torch_geometric.loader import DataLoader


def calculate(dataset, model, target, device='cpu'):
    model.eval()
    data = DataLoader(dataset=dataset, batch_size=len(dataset))

    for d in data:
        y = d.y[:, target].to(device)
        pred = model(d.to(device)).reshape(len(y))
        return getr2(y, pred)
        

def getr2(y, pred):
    y_bar = torch.mean(y)

    ssr = torch.sum(torch.pow(y - pred, 2.0))
    sst = torch.sum(torch.pow(y - y_bar, 2.0))

    return float(1 - (ssr / sst))
