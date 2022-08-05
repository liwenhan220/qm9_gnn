# Setup

First, install Pytorch and Pytorch Geometric from following links

https://pytorch.org/get-started/locally/

https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

Then, install RDKit package with `pip install rdkit`. RDKit Package will help converting molecules to Graphs and generating 3D coordinates.

# Training your model

Open Terminal in the folder and run `python train.py`. The usage is `python train.py [model_type] [save_dir] [device] [num_epochs] [target_low] [target_high]`.

model_type: (gat/gnn/egnn)

save_dir: Your model output directory

device: cpu, cuda, mps, etc.

target: 0 to 18

Example: `python train.py gat model_v1 cpu 30 3 4` will use gat model to learn properties 3 and 4 of molecules with cpu, and the model back up is saved to folder `model_v1`.

# Analyze
During training, you can analyze the learning curves with `draw.py`. The usage is `python draw.py [target] [model_dir] [graph]`.

# Using your model

Open Terminal in the folder and run `python train.py`. It should prompt the usage
