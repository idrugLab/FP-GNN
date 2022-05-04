from argparse import Namespace
import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from fpgnn.data import GetPubChemFPs, create_graph, get_atom_features_dim
import csv

atts_out = []

class FPN(nn.Module):
    def __init__(self,args):
        super(FPN, self).__init__()
        self.fp_2_dim=args.fp_2_dim
        self.dropout_fpn = args.dropout
        self.cuda = args.cuda
        self.hidden_dim = args.hidden_size
        self.args = args
        if hasattr(args,'fp_type'):
            self.fp_type = args.fp_type
        else:
            self.fp_type = 'mixed'
        
        if self.fp_type == 'mixed':
            self.fp_dim = 1489
        else:
            self.fp_dim = 1024
        
        if hasattr(args,'fp_changebit'):
            self.fp_changebit = args.fp_changebit
        else:
            self.fp_changebit = None
        
        self.fc1=nn.Linear(self.fp_dim, self.fp_2_dim)
        self.act_func = nn.ReLU()
        self.fc2 = nn.Linear(self.fp_2_dim, self.hidden_dim)
        self.dropout = nn.Dropout(p=self.dropout_fpn)
    
    def forward(self, smile):
        fp_list=[]
        for i, one in enumerate(smile):
            fp=[]
            mol = Chem.MolFromSmiles(one)
            
            if self.fp_type == 'mixed':
                fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
                fp_phaErGfp = AllChem.GetErGFingerprint(mol,fuzzIncrement=0.3,maxPath=21,minPath=1)
                fp_pubcfp = GetPubChemFPs(mol)
                fp.extend(fp_maccs)
                fp.extend(fp_phaErGfp)
                fp.extend(fp_pubcfp)
            else:
                fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fp.extend(fp_morgan)
            fp_list.append(fp)
                
        if self.fp_changebit is not None and self.fp_changebit != 0:
            fp_list = np.array(fp_list)
            fp_list[:,self.fp_changebit-1] = np.ones(fp_list[:,self.fp_changebit-1].shape)
            fp_list.tolist()
        
        fp_list = torch.Tensor(fp_list)

        if self.cuda:
            fp_list = fp_list.cuda()
        fpn_out = self.fc1(fp_list)
        fpn_out = self.dropout(fpn_out)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2(fpn_out)
        return fpn_out

class GATLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout_gnn, alpha, inter_graph, concat=True):
        super(GATLayer, self).__init__()
        self.dropout_gnn= dropout_gnn
        self.in_features = in_features 
        self.out_features = out_features
        self.alpha = alpha 
        self.concat = concat 
        self.dropout = nn.Dropout(p=self.dropout_gnn)
        self.inter_graph = inter_graph

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        if self.inter_graph is not None:
            self.atts_out = []
    
    def forward(self,mole_out,adj):
        atom_feature = torch.mm(mole_out, self.W) 
        N = atom_feature.size()[0]

        atom_trans = torch.cat([atom_feature.repeat(1, N).view(N * N, -1), atom_feature.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features) 
        e = self.leakyrelu(torch.matmul(atom_trans, self.a).squeeze(2)) 

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        if self.inter_graph is not None:
            att_out = attention
            if att_out.is_cuda:
                att_out = att_out.cpu()
            att_out = np.array(att_out)
            att_out[att_out<-10000] = 0
            att_out = att_out.tolist()
            atts_out.append(att_out)
        
        attention = nn.functional.softmax(attention, dim=1)
        attention = self.dropout(attention)
        output = torch.matmul(attention, atom_feature) 

        if self.concat:
            return nn.functional.elu(output)
        else:
            return output 


class GATOne(nn.Module):
    def __init__(self,args):
        super(GATOne, self).__init__()
        self.nfeat = get_atom_features_dim()
        self.nhid = args.nhid
        self.dropout_gnn = args.dropout_gat
        self.atom_dim = args.hidden_size
        self.alpha = 0.2
        self.nheads = args.nheads
        self.args = args
        self.dropout = nn.Dropout(p=self.dropout_gnn)
        
        if hasattr(args,'inter_graph'):
            self.inter_graph = args.inter_graph
        else:
            self.inter_graph = None
        
        self.attentions = [GATLayer(self.nfeat, self.nhid, dropout_gnn=self.dropout_gnn, alpha=self.alpha, inter_graph=self.inter_graph, concat=True) for _ in range(self.nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATLayer(self.nhid * self.nheads, self.atom_dim, dropout_gnn=self.dropout_gnn, alpha=self.alpha, inter_graph=self.inter_graph, concat=False)

    def forward(self,mole_out,adj):
        mole_out = self.dropout(mole_out)
        mole_out = torch.cat([att(mole_out, adj) for att in self.attentions], dim=1)
        mole_out = self.dropout(mole_out)
        mole_out = nn.functional.elu(self.out_att(mole_out, adj))
        return nn.functional.log_softmax(mole_out, dim=1)

class GATEncoder(nn.Module):
    def __init__(self,args):
        super(GATEncoder,self).__init__()
        self.cuda = args.cuda
        self.args = args
        self.encoder = GATOne(self.args)
    
    def forward(self,mols,smiles):
        atom_feature, atom_index = mols.get_feature()
        if self.cuda:
            atom_feature = atom_feature.cuda()
        
        gat_outs=[]
        for i,one in enumerate(smiles):
            adj = []
            mol = Chem.MolFromSmiles(one)
            adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
            adj = adj/1
            adj = torch.from_numpy(adj)
            if self.cuda:
                adj = adj.cuda()
            
            atom_start, atom_size = atom_index[i]
            one_feature = atom_feature[atom_start:atom_start+atom_size]
            
            gat_atoms_out = self.encoder(one_feature,adj)
            gat_out = gat_atoms_out.sum(dim=0)/atom_size
            gat_outs.append(gat_out)
        gat_outs = torch.stack(gat_outs, dim=0)
        return gat_outs

class GAT(nn.Module):
    def __init__(self,args):
        super(GAT,self).__init__()
        self.args = args
        self.encoder = GATEncoder(self.args)
        
    def forward(self,smile):
        mol = create_graph(smile, self.args)
        gat_out = self.encoder.forward(mol,smile)

        return gat_out

class FpgnnModel(nn.Module):
    def __init__(self,is_classif,gat_scale,cuda,dropout_fpn):
        super(FpgnnModel, self).__init__()
        self.gat_scale = gat_scale
        self.is_classif = is_classif
        self.cuda = cuda
        self.dropout_fpn = dropout_fpn
        if self.is_classif:
            self.sigmoid = nn.Sigmoid()

    def create_gat(self,args):
        self.encoder3 = GAT(args)
    
    def create_fpn(self,args):
        self.encoder2 = FPN(args)
    
    def create_scale(self,args):
        linear_dim = int(args.hidden_size)
        if self.gat_scale == 1:
            self.fc_gat = nn.Linear(linear_dim,linear_dim)
        elif self.gat_scale == 0:
            self.fc_fpn = nn.Linear(linear_dim,linear_dim)
        else:
            self.gat_dim = int((linear_dim*2*self.gat_scale)//1)
            self.fc_gat = nn.Linear(linear_dim,self.gat_dim)
            self.fc_fpn = nn.Linear(linear_dim,linear_dim*2-self.gat_dim)
        self.act_func = nn.ReLU()

    def create_ffn(self,args):
        linear_dim = args.hidden_size
        if self.gat_scale == 1:
            self.ffn = nn.Sequential(
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim, out_features=linear_dim, bias=True),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
                                     )
        elif self.gat_scale == 0:
            self.ffn = nn.Sequential(
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim, out_features=linear_dim, bias=True),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
                                     )

        else:
            self.ffn = nn.Sequential(
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim*2, out_features=linear_dim, bias=True),
                                     nn.ReLU(),
                                     nn.Dropout(self.dropout_fpn),
                                     nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)
                                     )
    
    def forward(self,input):
        if self.gat_scale == 1:
            output = self.encoder3(input)
        elif self.gat_scale == 0:
            output = self.encoder2(input)
        else:
            gat_out = self.encoder3(input)
            fpn_out = self.encoder2(input)
            gat_out = self.fc_gat(gat_out)
            gat_out = self.act_func(gat_out)
            
            fpn_out = self.fc_fpn(fpn_out)
            fpn_out = self.act_func(fpn_out)
            
            output = torch.cat([gat_out,fpn_out],axis=1)
        output = self.ffn(output)
        
        if self.is_classif and not self.training:
            output = self.sigmoid(output)
        
        return output

def get_atts_out():
    return atts_out

def FPGNN(args):
    if args.dataset_type == 'classification':
        is_classif = 1
    else:
        is_classif = 0
    model = FpgnnModel(is_classif,args.gat_scale,args.cuda,args.dropout)
    if args.gat_scale == 1:
        model.create_gat(args)
        model.create_ffn(args)
    elif args.gat_scale == 0:
        model.create_fpn(args)
        model.create_ffn(args)
    else:
        model.create_gat(args)
        model.create_fpn(args)
        model.create_scale(args)
        model.create_ffn(args)
    
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
    
    return model