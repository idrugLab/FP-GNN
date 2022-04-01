from argparse import ArgumentParser, Namespace
from hyperopt import fmin, tpe, hp
import numpy as np
import os
from copy import deepcopy
from fpgnn.tool import set_hyper_argument, set_log
from train import training

space = {
         'fp_2_dim':hp.quniform('fp_2_dim', low=300, high=600, q=50),
         'nhid':hp.quniform('nhid', low=40, high=80, q=5),
         'nheads':hp.quniform('nheads', low=2, high=8, q=1),
         'gat_scale':hp.quniform('gat_scale', low=0.2, high=0.8, q=0.1),
         'dropout':hp.quniform('dropout', low=0.0, high=0.6, q=0.05),
         'dropout_gat':hp.quniform('dropout_gat', low=0.0, high=0.6, q=0.05)
}


def fn(space):
    search_no = args.search_now
    log_name = 'train'+str(search_no)
    log = set_log(log_name,args.log_path)
    result_path = os.path.join(args.log_path, 'hyper_para_result.txt')
    
    list = ['fp_2_dim','nhid','nheads']
    for one in list:
        space[one] = int(space[one])
    hyperp = deepcopy(args)
    name_list = []
    change_args = []
    for key,value in space.items():
        name_list.append(str(key))
        name_list.append('-')
        name_list.append((str(value))[:5])
        name_list.append('-')
        setattr(hyperp,key,value)
    dir_name = "".join(name_list)
    dir_name = dir_name[:-1]
    hyperp.save_path = os.path.join(hyperp.save_path, dir_name)
    
    ave,std = training(hyperp,log)
    
    with open(result_path,'a') as file:
        file.write(str(space)+'\n')
        file.write('Result '+str(hyperp.metric)+' : '+str(ave)+' +/- '+str(std)+'\n')
    
    if ave is None:
        if hyperp.dataset_type == 'classification':
            ave = 0
        else:
            raise ValueError('Result of model is error.')
    
    args.search_now += 1
    
    if hyperp.dataset_type == 'classification':
        return -ave
    else:
        return ave

def hyper_searching(args):
    result_path = os.path.join(args.log_path, 'hyper_para_result.txt')
    
    result = fmin(fn,space,tpe.suggest,args.search_num)
    
    with open(result_path,'a') as file:
        file.write('Best Hyperparameters : \n')
        file.write(str(result)+'\n')
        

if __name__ == '__main__':
    args = set_hyper_argument()
    hyper_searching(args)
    
