from argparse import ArgumentParser, Namespace
import torch
from .tool import mkdir

def add_train_argument(p):
    p.add_argument('--data_path',type=str,
                   help='The path of input CSV file.')
    p.add_argument('--save_path',type=str,default='model_save',
                   help='The path to save output model.pt.,default is "model_save/"')
    p.add_argument('--log_path',type=str,default='log',
                   help='The dir of output log file.')
    p.add_argument('--dataset_type',type=str,choices=['classification', 'regression'],
                   help='The type of dataset.')
    p.add_argument('--is_multitask',type=int,default=0,
                   help='Whether the dataset is multi-task. 0:no  1:yes.')
    p.add_argument('--task_num',type=int,default=1,
                   help='The number of task in multi-task training.')
    p.add_argument('--split_type',type=str,choices=['random', 'scaffold'],default='random',
                   help='The type of data splitting.')
    p.add_argument('--split_ratio',type=float,nargs=3,default=[0.8,0.1,0.1],
                   help='The ratio of data splitting.[train,valid,test]')
    p.add_argument('--val_path',type=str,
                   help='The path of excess validation data.')
    p.add_argument('--test_path',type=str,
                   help='The path of excess testing data.')
    p.add_argument('--seed',type=int,default=0,
                   help='The random seed of model. Using in splitting data.')
    p.add_argument('--num_folds',type=int,default=1,
                   help='The number of folds in cross validation.')
    p.add_argument('--metric',type=str,choices=['auc', 'prc-auc', 'rmse'],default=None,
                   help='The metric of data evaluation.')
    p.add_argument('--epochs',type=int,default=30,
                   help='The number of epochs.')
    p.add_argument('--batch_size',type=int,default=50,
                   help='The size of batch.')
    p.add_argument('--fp_type',type=str,choices=['morgan','mixed'],default='mixed',
                   help='The type of fingerprints. Use "morgan" or "mixed".')
    p.add_argument('--hidden_size',type=int,default=300,
                   help='The dim of hidden layers in model.')
    p.add_argument('--fp_2_dim',type=int,default=512,
                   help='The dim of the second layer in fpn.')
    p.add_argument('--nhid',type=int,default=60,
                   help='The dim of the attentions in gnn.')
    p.add_argument('--nheads',type=int,default=8,
                   help='The number of the attentions in gnn.')
    p.add_argument('--gat_scale',type=float,default=0.5,
                   help='The ratio of gnn in model.')
    p.add_argument('--dropout',type=float,default=0.0,
                   help='The dropout of fpn and ffn.')
    p.add_argument('--dropout_gat',type=float,default=0.0,
                   help='The dropout of gnn.')

def add_predict_argument(p):
    p.add_argument('--predict_path', type=str,
                   help='The path of input CSV file to predict.')
    p.add_argument('--result_path', type=str,default='result.txt',
                   help='The path of output CSV file.')
    p.add_argument('--model_path', type=str,
                   help='The path of model.pt.')
    p.add_argument('--batch_size', type=int,default=50,
                   help='The size of batch.')

def add_hyper_argument(p):
    p.add_argument('--search_num', type=int,default=10,
                   help='The number of hyperparameters searching.')

def add_interfp_argument(p):
    p.add_argument('--log_path', type=str,default='log',
                   help='The path of log file.')

def add_intergraph_argument(p):
    p.add_argument('--predict_path', type=str,
                   help='The path of input CSV file to predict.')
    p.add_argument('--figure_path', type=str,default='figure',
                   help='The path of output figure file.')
    p.add_argument('--model_path', type=str,
                   help='The path of model.pt.')
    p.add_argument('--batch_size', type=int,default=50,
                   help='The size of batch.')

def set_train_argument():
    p = ArgumentParser()
    add_train_argument(p)
    args = p.parse_args()
    
    assert args.data_path
    assert args.dataset_type
    
    mkdir(args.save_path)
    
    if args.metric is None:
        if args.dataset_type == 'classification':
            args.metric = 'auc'
        elif args.dataset_type == 'regression':
            args.metric = 'rmse'

    if args.dataset_type == 'classification' and args.metric not in ['auc', 'prc-auc']:
        raise ValueError('Metric or data_type is error.')
    if args.dataset_type == 'regression' and args.metric not in ['rmse']:
        raise ValueError('Metric or data_type is error.')
    if args.fp_type not in ['mixed','morgan']:
        raise ValueError('Fingerprint type is error.')

    args.cuda = torch.cuda.is_available()
    args.init_lr = 1e-4
    args.max_lr = 1e-3
    args.final_lr = 1e-4
    args.warmup_epochs = 2.0
    args.num_lrs = 1
    
    return args

def set_predict_argument():
    p = ArgumentParser()
    add_predict_argument(p)
    args = p.parse_args()
    
    assert args.predict_path
    assert args.model_path
    
    args.cuda = torch.cuda.is_available()
    
    mkdir(args.result_path, isdir = False)
    
    return args

def set_hyper_argument():
    p = ArgumentParser()
    add_train_argument(p)
    add_hyper_argument(p)
    args = p.parse_args()
    
    assert args.data_path
    assert args.dataset_type
    
    mkdir(args.save_path)
    
    if args.metric is None:
        if args.dataset_type == 'classification':
            args.metric = 'auc'
        elif args.dataset_type == 'regression':
            args.metric = 'rmse'

    if args.dataset_type == 'classification' and args.metric not in ['auc', 'prc-auc']:
        raise ValueError('Metric or data_type is error.')
    if args.dataset_type == 'regression' and args.metric not in ['rmse']:
        raise ValueError('Metric or data_type is error.')
    if args.fp_type not in ['mixed','morgan']:
        raise ValueError('Fingerprint type is error.')

    args.cuda = torch.cuda.is_available()
    args.init_lr = 1e-4
    args.max_lr = 1e-3
    args.final_lr = 1e-4
    args.warmup_epochs = 2.0
    args.num_lrs = 1
    args.search_now = 0
    
    return args

def set_interfp_argument():
    p = ArgumentParser()
    add_predict_argument(p)
    add_interfp_argument(p)
    args = p.parse_args()
    
    assert args.predict_path
    assert args.model_path
    
    args.cuda = torch.cuda.is_available()
    args.fp_changebit = 0
    
    mkdir(args.result_path, isdir = False)
    
    return args

def set_intergraph_argument():
    p = ArgumentParser()
    add_intergraph_argument(p)
    args = p.parse_args()
    
    
    assert args.predict_path
    assert args.model_path
    
    args.cuda = torch.cuda.is_available()
    args.inter_graph = 1
    
    mkdir(args.figure_path, isdir = True)
    
    return args