import csv
import numpy as np
from fpgnn.tool import set_interfp_argument, set_log, get_scaler, load_args, load_data, load_model, rmse
from fpgnn.train import predict
from fpgnn.data import MoleDataSet

def make_fp_interpretation(args,log):
    info = log.info
    info('Load args.')
    
    scaler = get_scaler(args.model_path)
    train_args = load_args(args.model_path)
    
    for key,value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    info('Load data.')
    test_data = load_data(args.predict_path,args)
    test_label = test_data.label()
    test_label = np.squeeze(np.array(test_label))
    
    result = []
    orig_score = 0
    if hasattr(args,'fp_type'):
        fp_type = args.fp_type
    else:
        fp_type = 'mixed'
    
    if fp_type == 'mixed':
        fp_length = 1490
    else:
        fp_length = 1025
    
    for fp_changebit in range(fp_length):# 0:nothing changed  1-x:changed bit
        args.fp_changebit = fp_changebit
        model = load_model(args.model_path,args.cuda,pred_args=args)
        model_pred = predict(model,test_data,args.batch_size,scaler)
        model_pred = np.array(model_pred)
        if fp_changebit == 0:
            info('Original fingerprint. Nothing changed.')
            orig_score = rmse(test_label,model_pred)
        else:
            info(f'Change fingerprint bit : {fp_changebit}')
            change_score = rmse(test_label,model_pred)
            res = orig_score - change_score
            info(f'Change Importance: {res}')
            result.append([fp_changebit,res])
    
    with open(args.result_path,'w',newline = '') as file:
        writer = csv.writer(file)
        
        line = ['No_of_Bit_Changed','Importance']
        writer.writerow(line)
        
        for i in range(len(result)):
            line = result[i]
            writer.writerow(line)

if __name__ == '__main__':
    args = set_interfp_argument()
    log = set_log('inter_fp',args.log_path)
    make_fp_interpretation(args,log)
