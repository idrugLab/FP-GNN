from argparse import Namespace
from logging import Logger
import numpy as np
import os
from fpgnn.train import fold_train
from fpgnn.tool import set_log, set_train_argument, get_task_name, mkdir

def training(args,log):
    info = log.info
    
    seed_first = args.seed
    data_path = args.data_path
    save_path = args.save_path
    
    score = []
    
    for num_fold in range(args.num_folds):
        info(f'Seed {args.seed}')
        args.seed = seed_first + num_fold
        args.save_path = os.path.join(save_path,f'Seed_{args.seed}')
        mkdir(args.save_path)
        
        fold_score = fold_train(args,log)
        
        score.append(fold_score)
    score = np.array(score)
    
    info(f'Running {args.num_folds} folds in total.')
    if args.num_folds > 1:
        for num_fold, fold_score in enumerate(score):
            info(f'Seed {seed_first + num_fold} : test {args.metric} = {np.nanmean(fold_score):.6f}')
    score = np.nanmean(score, axis=1)
    score_ave = np.nanmean(score)
    score_std = np.nanstd(score)
    info(f'Average test {args.metric} = {score_ave:.6f} +/- {score_std:.6f}')
    
    if args.task_num > 1:
        for i,one_name in enumerate(args.task_names):
            info(f'Average test {one_name} {args.metric} = {np.nanmean(score[:, i]):.6f} +/- {np.nanstd(score[:, i]):.6f}')
    
    return score_ave,score_std

if __name__ == '__main__':
    args = set_train_argument()
    log = set_log('train',args.log_path)
    training(args,log)
