from converter_v2 import *
from gnn import *
from rdkit.Chem import Draw
import cv2

model_dir = 'gnn'
device = 'cpu'

TARGET = ['A','B','C','mu','alpha','homo','lumo','gap','r2','zpve','u0','u298','h298','g298','cv','u0_atom','u298_atom','h298_atom','g298_atom']

smiles = input('Pleases input molecule in SMILES format: ')
mol = Chem.MolFromSmiles(smiles)
img = Draw.MolToImage(mol)
print()
data = smiles_converter(smiles)
model = NN()
for i in range(3, 19):
    model.load_state_dict(torch.load(model_dir + '/nn_{}'.format(i), map_location=torch.device(device)))
    model.to(device=device)
    model.eval()
    with torch.no_grad():
        pred = model(data)
        print('{}: {}\n'.format(TARGET[i], pred[0][0]))

img.show()
# input('PRESS ANY KEY TO END')