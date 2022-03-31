from argparse import Namespace
from rdkit import Chem
import torch

atom_type_max = 100
atom_f_dim = 133
atom_features_define = {
    'atom_symbol': list(range(atom_type_max)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'charity_type': [0, 1, 2, 3],
    'hydrogen': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],}

smile_changed = {}

def get_atom_features_dim():
    return atom_f_dim

def onek_encoding_unk(key,length):
    encoding = [0] * (len(length) + 1)
    index = length.index(key) if key in length else -1
    encoding[index] = 1

    return encoding

def get_atom_feature(atom):
    feature = onek_encoding_unk(atom.GetAtomicNum() - 1, atom_features_define['atom_symbol']) + \
           onek_encoding_unk(atom.GetTotalDegree(), atom_features_define['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), atom_features_define['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), atom_features_define['charity_type']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), atom_features_define['hydrogen']) + \
           onek_encoding_unk(int(atom.GetHybridization()), atom_features_define['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]
    return feature

class GraphOne:
    def __init__(self,smile,args):
        self.smile = smile
        self.atom_feature = []

        mol = Chem.MolFromSmiles(self.smile)
        self.atom_num = mol.GetNumAtoms()
        
        for i, atom in enumerate(mol.GetAtoms()):
            self.atom_feature.append(get_atom_feature(atom))
        self.atom_feature = [self.atom_feature[i] for i in range(self.atom_num)]
        
class GraphBatch:
    def __init__(self,graphs,args):
        smile_list = []
        for graph in graphs:
            smile_list.append(graph.smile)
        self.smile_list = smile_list
        self.smile_num = len(self.smile_list)
        self.atom_feature_dim = get_atom_features_dim()
        self.atom_no = 1
        self.atom_index = []

        atom_feature = [[0]*self.atom_feature_dim]
        for graph in graphs:
            atom_feature.extend(graph.atom_feature)
            self.atom_index.append((self.atom_no,graph.atom_num))
            self.atom_no += graph.atom_num

        self.atom_feature = torch.FloatTensor(atom_feature) 

    def get_feature(self):
        return self.atom_feature,self.atom_index

def create_graph(smile,args):
    graphs = []
    for one in smile:
        if one in smile_changed:
            graph = smile_changed[one]
        else:
            graph = GraphOne(one, args)
            smile_changed[one] = graph
        graphs.append(graph)
    return GraphBatch(graphs,args)
