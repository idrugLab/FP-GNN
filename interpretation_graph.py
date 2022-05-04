import numpy as np
import matplotlib.cm as cm
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
from fpgnn.model import get_atts_out
from fpgnn.tool import set_intergraph_argument, get_scaler, load_args, load_data, load_model
from fpgnn.data import MoleDataSet
from fpgnn.train import predict

def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol

def drawmol_bond(smile,smi_att,path):
    smi_att = np.array(smi_att)
    atom_num = len(smi_att[0])
    
    for i in range(atom_num):
        for j in range(i+1):
            smi_att[j][i] = abs(smi_att[j][i]) + abs(smi_att[i][j])
            smi_att[i][j] = 0
    
    min_value = smi_att.min(axis=(0,1))
    max_value = smi_att.max(axis=(0,1))
    norm=matplotlib.colors.Normalize(vmin=min_value, vmax=max_value+0.15)
    cmap=cm.get_cmap('Oranges')
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    mol = Chem.MolFromSmiles(smile)
    mol = mol_with_atom_index(mol)
    
    bond_list = []
    bond_colors = {}
    bond_no = np.nonzero(smi_att)
    
    for i in range(len(bond_no[0])):
        a1 = int(bond_no[0][i])
        a2 = int(bond_no[1][i])
        
        bond_color = smi_att[a1,a2]
        bond_color = plt_colors.to_rgba(bond_color)
        
        bond = mol.GetBondBetweenAtoms(a1,a2).GetIdx()
        bond_list.append(bond)
        bond_colors[bond] = bond_color
    
    drawer = rdMolDraw2D.MolDraw2DCairo(500,500)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer,mol, 
                             highlightBonds=bond_list,
                             highlightBondColors=bond_colors)
    output_name = str(smile)
    output_name = output_name.replace('/','%')
    output_name = output_name.replace('\\','%')
    if len(output_name) > 50:
        output_name = output_name[0:50]
    str1 = path + '/' + output_name + '.jpg'
    with open(str1, 'wb') as file:
        file.write(drawer.GetDrawingText())
        print(f'Produce the interpretation molecule graph in {str1}')

def interp_graph(args):
    print('Load args.')
    scaler = get_scaler(args.model_path)
    train_args = load_args(args.model_path)
    
    for key,value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    print('Load data.')
    test_data = load_data(args.predict_path,args)
    fir_data_len = len(test_data)
    all_data = test_data
    if fir_data_len == 0:
        raise ValueError('Data is empty.')
    
    smi_exist = []
    for i in range(fir_data_len):
        if test_data[i].mol is not None:
            smi_exist.append(i)
    test_data = MoleDataSet([test_data[i] for i in smi_exist])
    now_data_len = len(test_data)
    print('There are ',now_data_len,' smiles in total.')
    if fir_data_len - now_data_len > 0:
        print('There are ',fir_data_len - now_data_len, ' smiles invalid.')
    
    test_smile = test_data.smile()
    
    print('Load model')
    model = load_model(args.model_path,args.cuda,pred_args=args)
    test_pred = predict(model,test_data,args.batch_size,scaler)
    assert len(test_data) == len(test_pred)
    test_pred = np.array(test_pred)
    test_pred = test_pred.tolist()
    
    atts_out = get_atts_out()
    nhead = args.nheads + 1
    
    for i in range(now_data_len):
        smile = test_smile[i]
        smi_att = atts_out[(i+1) * nhead - 1]
        drawmol_bond(smile,smi_att,args.figure_path)

if __name__ == '__main__':
    args = set_intergraph_argument()
    interp_graph(args)