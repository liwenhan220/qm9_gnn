import csv

from torch import Tensor
from converter_v2 import *
from tqdm import tqdm
from torch_geometric.data import download_url
import torch
from torch_geometric.data import InMemoryDataset, download_url

url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"

class QM9(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['qm9.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        download_url(url, self.raw_dir)
        

    def atof(self, ls):
        for i in range(len(ls)):
            ls[i] = float(ls[i])
        return ls

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        with open(self.raw_paths[0]) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in tqdm(csv_reader, total = 133886):
                if line_count == 0:
                    line_count += 1
                else:
                    try:
                        mol = smiles_converter(row[1])
                        mol.y = Tensor([self.atof(row[2:])])
                        line_count += 1
                        data_list.append(mol)
                    except:
                        pass

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])