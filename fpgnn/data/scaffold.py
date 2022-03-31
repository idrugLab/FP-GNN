from collections import defaultdict
import logging
import random
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from fpgnn.data import MoleDataSet

def generate_scaffold(mol,include_chirality=False):
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(mol,use_indices=False):
    scaffolds = defaultdict(set)
    for i, one in enumerate(mol):
        scaffold = generate_scaffold(one)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(one)

    return scaffolds


def scaffold_split(data,size,seed,log):
    assert sum(size) == 1

    # Split
    train_size, val_size, test_size = size[0] * len(data), size[1] * len(data), size[2] * len(data)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(data.mol(), use_indices=True)

    index_sets = list(scaffold_to_indices.values())
    big_index_sets = []
    small_index_sets = []
    for index_set in index_sets:
        if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
            big_index_sets.append(index_set)
        else:
            small_index_sets.append(index_set)
    random.seed(seed)
    random.shuffle(big_index_sets)
    random.shuffle(small_index_sets)
    index_sets = big_index_sets + small_index_sets

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    log.debug(f'Total scaffolds = {len(scaffold_to_indices):,} | '
              f'train scaffolds = {train_scaffold_count:,} | '
              f'val scaffolds = {val_scaffold_count:,} | '
              f'test scaffolds = {test_scaffold_count:,}')
    
    # Map from indices to data
    train = [data[i] for i in train]
    val = [data[i] for i in val]
    test = [data[i] for i in test]

    return MoleDataSet(train), MoleDataSet(val), MoleDataSet(test)
