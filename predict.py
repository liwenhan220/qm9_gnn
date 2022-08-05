from converter_v2 import *
from rdkit.Chem import Draw
import sys

# usage: python predict.py [model_type] [model_dir] [device] [target_low] [target_high]
# model_type: (gat/gnn/egnn)
# model_dir: Directory of your model
# device: cpu, cuda, mps, etc.
# target: 0 to 18

try:
    if len(sys.argv) != 6:
        raise NameError('')
    model_type = sys.argv[1]
    if model_type == 'gat':
        from gat import NN
    elif model_type == 'gnn':
        from gnn import NN
    elif model_type == 'egnn':
        from egnn import NN
    else:
        raise NameError('')

    model_dir = sys.argv[2]
    device = sys.argv[3]

    TARGET = ['A','B','C','mu','alpha','homo','lumo','gap','r2','zpve','u0','u298','h298','g298','cv','u0_atom','u298_atom','h298_atom','g298_atom']

    while True:
        smiles = input('Pleases input molecule in SMILES format: ')
        mol = Chem.MolFromSmiles(smiles)
        img = Draw.MolToImage(mol)
        print()
        data = smiles_converter(smiles)
        model = NN()
        for i in range(int(sys.argv[4]), int(sys.argv[5])+1):
            model.load_state_dict(torch.load(model_dir + '/nn_{}'.format(i), map_location=torch.device(device)))
            model.to(device=device)
            model.eval()
            with torch.no_grad():
                pred = model(data)
                print('{}: {}\n'.format(TARGET[i], pred[0][0]))

except NameError:
    print('\n')
    print('usage: python predict.py [model_type] [model_dir] [device] [target_low] [target_high]')
    print('\n')
    print('model_type: gat, gnn, egnn')
    print('model_dir: Directory of your model')
    print('device: cpu, cuda, mps, etc.')
    print('target: 0 to 18')
